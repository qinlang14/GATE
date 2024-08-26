import torch
from typing import List
import torch.nn as nn
import numpy as np


def padding(inp:List[torch.Tensor], pad=0):

    new = nn.utils.rnn.pad_sequence(inp, batch_first=True, padding_value=pad)
    mask_matrix = torch.where(torch.sum(new,dim=-1)==0, 1, 0)

    return new, mask_matrix


def loss_function(result, rl=True):
    # result: T * (prob:[batch_size], reward:[batch_size])
    batch_size = result[0][0].shape[0]
    loss = []
    detail = []
    reward = torch.zeros(batch_size, device="cuda:0")
    if rl:
        gamma = 0.98
        for l in result[::-1]:
            prob, step_reward, node_nll, knowledge_nll = l

            reward = step_reward + gamma * reward
            walk_loss = 0.25*torch.sum(-prob * (reward+3)) / batch_size
            # node_nll = torch.sum(torch.exp(prob) * node_nll)/ self.batch_size
            node_nll = 0.75*node_nll / batch_size
            # knowledge_nll = torch.sum(torch.exp(prob) * knowledge_nll)/ self.batch_size
            knowledge_nll = 1.5*knowledge_nll / batch_size
            loss.append(walk_loss + node_nll + knowledge_nll)  # T
            detail.append([walk_loss.item(), node_nll.item(), knowledge_nll.item()])  # T,3

    else:
        for l in result[::-1]:
            step_reward, node_nll, knowledge_nll = l
            # node_nll = torch.sum(torch.exp(prob) * node_nll)/ self.batch_size
            node_nll = node_nll / batch_size
            # knowledge_nll = torch.sum(torch.exp(prob) * knowledge_nll)/ self.batch_size
            knowledge_nll = knowledge_nll / batch_size
            loss.append(node_nll + knowledge_nll)  # T
            detail.append([node_nll.item(), knowledge_nll.item()])  # T,3

    return loss, np.array(detail)


def topk_accuracy(label:List[List[str]], output:List[List[str]]):
    # top-1, top-5, top-10, all
    batch_size = len(label)
    topk_recall = np.zeros(4, dtype=float)

    for i in range(batch_size):
        sample_label = label[i]
        sample_pool = output[i]
        num_label = len(sample_label)
        for l in sample_label:
            if l in sample_pool:
                idx = sample_pool.index(l)
                for j, k in enumerate([1,5,10,1000]):
                    if idx < k:
                        topk_recall[j] += 1 / num_label

    return topk_recall


def reward_function(node_reward_base, knowledge_reward_base, pool_reward_base, base_poolsize, gold_k:List[List[str]], gold_n:List[str], nodes:List[str], pool:List[List[str]], raw_pool:List[List[str]]):
    batch_size = len(gold_k)
    node_reward = np.zeros(batch_size, dtype=float)
    knowledge_reward = np.zeros(batch_size, dtype=float)
    pool_reward = np.zeros(batch_size, dtype=float)
    reward = torch.zeros(batch_size, device="cuda:0")
    for i in range(batch_size):
        node_reward[i] = 2*node_reward_base if nodes[i] in gold_n else -node_reward_base

        gold_knowledge = np.array([2.0-0.1*(pool[i].index(g)) if g in pool[i] else -1 for g in gold_k[i]])
        locate = np.max(np.mean(gold_knowledge),-1)
        knowledge_reward[i] = knowledge_reward_base * locate

        pool_reward[i] = pool_reward_base * (locate/32) / (len(pool[i])/base_poolsize)

        reward[i] = node_reward[i]+knowledge_reward[i]+pool_reward[i]

    topk_acc = topk_accuracy(gold_k, raw_pool)

    return reward, topk_acc, np.array([node_reward, knowledge_reward, pool_reward])


def smooth_labels(input, smoothing_rate):
    """
    Smoothing labels function.
    :param input: input tensor of shape (batch_size, num_classes)
    :param smoothing_rate: the percentage of the maximum value to be smoothed out
    :return: smoothed labels tensor of the same shape as input
    """

    max_values, _ = torch.max(input, dim=1, keepdim=True)
    smooth_values = max_values * smoothing_rate
    smooth_labels = input * (1 - smoothing_rate)
    valid_label = torch.sum(input!=0, dim=1).unsqueeze(-1)
    valid_label = valid_label.masked_fill(valid_label==1, 2)
    smooth_labels = smooth_labels + (smooth_values / (valid_label - 1))
    smooth_labels = smooth_labels.masked_fill(input==0, 0)
    normalized_labels = smooth_labels / smooth_labels.sum(dim=1, keepdim=True)
    if True in torch.isnan(normalized_labels) or True in torch.isinf(normalized_labels):
        print()

    return normalized_labels


def nll_loss(label, output, scale=False):
    mask = output != 0
    output1 = torch.where(mask, torch.log(output+1e-8), output)
    output2 = torch.where(mask, torch.log(1-output+1e-8), output)
    result1 = -output1[mask] * label[mask]
    result2 = -output2[mask] * (1-label)[mask]
    if scale:
        result2 = result2 / (len(output[mask]) / label.shape[1])
    # check = (torch.sum(result1), torch.sum(result2))
    result = torch.sum(result1+result2, dim=-1)
    return result


def log_t(u, t):
    """Compute log_t for `u`."""
    if t == 1.0:
        return torch.log(u)
    else:
        return ((u ** (1.0 - t)) - 1.0) / (1.0 - t)

def exp_t(u, t):
    """Compute exp_t for `u`."""
    if t == 1.0:
        return torch.exp(u)
    else:
        return (torch.relu(1.0 + (1.0 - t) * u)) ** (1.0 / (1.0 - t))

def compute_normalization_fixed_point(activations, t, num_iters=5):
    """Returns the normalization value for each example (t > 1.0).
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    mu = torch.max(activations, dim=-1).values.view(-1, 1)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0
    i = 0
    while i < num_iters:
        i += 1
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)
        normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

    logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)

    return -log_t(1.0 / logt_partition, t) + mu

def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    if t < 1.0:
        return None # not implemented as these values do not occur in the authors experiments...
    else:
        return compute_normalization_fixed_point(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.
    Returns:
    A probabilities tensor.
    """
    if t == 1.0:
        normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
    else:
        normalization_constants = compute_normalization(activations, t, num_iters)

    return exp_t(activations - normalization_constants, t)

def bi_tempered_logistic_loss(activations, labels, t1, t2, num_iters=5):
    """Bi-Tempered Logistic Loss with custom gradient.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: A tensor with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.
    Returns:
    A loss tensor.
    """
    # with torch.autograd.set_detect_anomaly(True):
    probabilities = tempered_softmax(activations, t2, num_iters)

    temp1 = (log_t(labels + 1e-10, t1) - log_t(probabilities, t1)) * labels
    temp2 = (1 / (2 - t1)) * (torch.pow(labels, 2 - t1) - torch.pow(probabilities, 2 - t1))

    loss_values = temp1 - temp2
    if True in torch.isnan(loss_values) or True in torch.isinf(loss_values):
        print("Nan in loss!")
    return torch.sum(loss_values, dim=-1)
