import torch
import torch.nn as nn
import torch.nn.functional as F
from KnowledgeSelection.GATEModel.module import ModalityAttentionLayer
from KnowledgeSelection.GATEModel.model_utils import smooth_labels, bi_tempered_logistic_loss
import numpy as np
from typing import List


class NodeSelector(nn.Module):
    def __init__(self, hidden_dim, gvt, mask, propagation_rate):
        super().__init__()
        self.device = "cuda:0"
        self.gvt = gvt
        self.mask = mask
        self.propagation_rate = propagation_rate
        self.label_smoothing_rate = 0.20

        self.modality_attention_layer = ModalityAttentionLayer(int(hidden_dim / 2))

    def forward(self, batch_input, all_node_embedding, input_info, agent_state, current_input, train, label_smoothing_rate):
        nodes, adj, label = batch_input
        current_node, current_activation, current_score = current_input
        self.label_smoothing_rate = label_smoothing_rate
        agent_state, current_node, current_activation, current_score, prob, node_nll = \
        self.walk_step(nodes, adj, label, all_node_embedding, input_info, agent_state,
                       current_node, current_activation, current_score, train)

        return agent_state, current_node, current_activation, current_score, prob, node_nll

    # @profile
    def node_score(self, all_node_embedding: torch.Tensor, state: torch.Tensor, node_idx: torch.Tensor, nodes: np.ndarray,
                   adj: torch.Tensor, train: bool):
        # 以每一样本目前所在节点，找到其在邻接矩阵中的位置
        adj_idx = node_idx.unsqueeze(-1).expand(-1, -1, adj.size(-1))
        # 取出各样本将要用到的邻接矩阵中的行，即当前节点的后继情况
        neighbor_idx = adj.gather(dim=1, index=adj_idx).squeeze()
        # 每一样本按邻接取出所有后继的embedding，根据邻接关系构建新邻接矩阵
        step_adj = []
        step_embedding = []
        step_indices = [i.nonzero(as_tuple=True)[0] for i in neighbor_idx]
        len_step_i = len(step_indices)
        for idx, embedding in enumerate(all_node_embedding):
            # 样本的后继在embedding矩阵中的位置
            indices = step_indices[idx]
            # 取新的邻接矩阵
            rows = torch.index_select(adj[idx], dim=0, index=indices)
            cols = torch.index_select(rows, dim=1, index=indices)
            step_adj.append(cols)
            # 新的embedding矩阵
            step_embedding.append(embedding[indices].squeeze())
        step_embedding = nn.utils.rnn.pad_sequence(step_embedding, batch_first=True)
        # 与state拼接
        step_embedding = torch.cat([step_embedding, state.unsqueeze(1).expand(-1, step_embedding.shape[1], -1)], dim=2)
        max_n = step_embedding.shape[1]
        step_adj = torch.stack([F.pad(tensor, (0, max_n - tensor.size(1), 0, max_n - tensor.size(0))) for tensor in step_adj])
        step_mask = torch.zeros([len_step_i, max_n], device=self.device)
        for i in range(len_step_i):
            step_mask[i][0:len(step_indices[i])] = 1
        # 计算这一跳的局部得分
        activations = self.gvt(step_embedding, step_adj)
        activations = activations.masked_fill(step_mask == 0, self.mask)  # [B, max_len]
        score = F.softmax(activations, dim=-1)

        # 采样
        m = torch.distributions.Categorical(score)
        if train:
            idx = m.sample()
        else:
            idx = torch.max(score, dim=-1)[1]  # batch_size List[int]
        # idx = torch.max(score, dim=-1)[1]
        prob = m.log_prob(idx)
        # 确定这一跳的目的节点
        idx = idx.tolist()
        next_node = nodes[np.arange(len_step_i), [step_indices[i][idx[i]].item() for i in range(len_step_i)]]  # batch_size  np.ndarray[str]

        return next_node, activations, score, step_indices, prob

    # @profile
    def walk_step(self, nodes, adj, label, all_node_embedding, input_info: torch.Tensor, agent_state: torch.Tensor,
                  current_node: List[str], current_activation: torch.Tensor,
                  current_score: torch.Tensor, train):

        node_idx = torch.as_tensor([[np.where(nodes[i]==current_node[i])[0][0]] for i in range(len(current_node))], dtype=torch.long, device=self.device)
        node_state = all_node_embedding.gather(dim=1, index=node_idx.unsqueeze(-1).expand(-1, -1, all_node_embedding.size(-1))).squeeze()

        inp = torch.stack([input_info, agent_state, node_state], dim=1)
        state = self.modality_attention_layer(inp)
        next_node, activations, score, step_indices, prob = self.node_score(all_node_embedding, state, node_idx, nodes, adj, train)

        step_label = torch.zeros(label.shape, device=self.device)
        for i in range(step_label.shape[0]):
            step_label[i][0:len(step_indices[i])] = label[i][step_indices[i]]
        step_label = step_label[:, 0:activations.shape[1]]

        if torch.sum(current_score) != 0:
            current_score = (1 - self.propagation_rate) * current_score * torch.sum(current_score!=0, dim=1).unsqueeze(-1)
            score = self.propagation_rate * score * torch.sum(score!=0, dim=1).unsqueeze(-1)
        for i in range(current_score.shape[0]):
            current_score[i].scatter_add_(0, step_indices[i], score[i][:len(step_indices[i])])
        current_score = current_score / current_score.sum(dim=1, keepdim=True)

        step_label = smooth_labels(step_label, smoothing_rate=self.label_smoothing_rate)

        nll = torch.sum(bi_tempered_logistic_loss(activations=activations, labels=step_label, t1=0.8, t2=1.2))

        return state, next_node, current_activation, current_score, prob, nll