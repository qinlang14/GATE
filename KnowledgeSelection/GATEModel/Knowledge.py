import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from KnowledgeSelection.GATEModel.module import RepresentationLayer
from KnowledgeSelection.GATEModel.model_utils import padding, smooth_labels, bi_tempered_logistic_loss
import math
import numpy as np
from typing import List, Optional, Any


class KnowledgeSelector(nn.Module):
    def __init__(self, hidden_dim, base_poolsize, min_poolsize, environment, mask):
        super().__init__()

        self.device = "cuda:0"
        self.base_poolsize = base_poolsize
        self.min_poolsize = min_poolsize
        self.environment = environment
        self.mask = mask

        self.label_smoothing_rate = 0.15

        self.knowledge_layer = RepresentationLayer(in_features=hidden_dim, out_features=int(hidden_dim / 2))
        self.knowledge_score_layer = RepresentationLayer(in_features=1 * self.base_poolsize, out_features=self.base_poolsize)

    def forward(self, batch_input, agent_state, current_score, label_smoothing_rate):
        gold, nodes = batch_input
        self.label_smoothing_rate = label_smoothing_rate
        knowledge_pool, knowledge_nll, raw_kp, mean_k = self.knowledge_selection(gold, nodes, agent_state, current_score)

        return knowledge_pool, knowledge_nll, raw_kp, mean_k

    def knowledge_pad(self, inp, value=0):
        if inp.shape[1] < self.base_poolsize:
            inp = F.pad(inp, (0, self.base_poolsize - inp.shape[1], 0, 0), value=value)
        return inp

    def knowledge_score(self, candidate: List[torch.Tensor], multiplier: Optional[torch.Tensor] = None,
                        attention=None):
        candidate, mask_matrix = padding(candidate, pad=0)

        candidate = self.knowledge_layer(candidate)
        candidate = torch.sum(multiplier * candidate, dim=-1) / math.sqrt(multiplier.shape[-1])
        att, _ = padding(attention, pad=0)
        max_n = att.shape[1]
        att = self.knowledge_pad(att)
        att = self.knowledge_score_layer(att)[:, 0:max_n]
        candidate = att * candidate

        result = candidate.masked_fill(mask_matrix == 1, self.mask)
        score = torch.softmax(result, dim=-1)
        return result, score

    def knowledge_selection(self, gold, nodes, agent_state: torch.Tensor,
                            score_distribution: torch.Tensor) -> tuple[list[Any], Tensor, list[Any], Tensor]:
        batch_size = len(score_distribution)

        indices = [i.nonzero(as_tuple=True)[0] for i in score_distribution]
        score = [torch.index_select(score_distribution[i], dim=0, index=indices[i]) for i in range(batch_size)]
        length = torch.as_tensor([len(s) for s in score], device=self.device)
        score_var_reverse = torch.as_tensor([1 - torch.var(s, unbiased=False) for s in score], device=self.device)

        # 通过y = kx^2 + b 将 [1-max_var, 1]映射到[0, 1]，以此作为不同样本知识池大小的参考比例
        # k = (length**2) / (2*(length-1) - ((1/length)-1)**2)
        k = 1 / (1 - (length / (length + (1 / length) - 1)))
        b = 1 - k
        # pool_size_rate = k*(score_var_reverse**2) + b
        pool_size_rate = k * (1 / (score_var_reverse)) + b
        pool_size = torch.as_tensor(torch.floor(self.base_poolsize * pool_size_rate), dtype=torch.int16, device=self.device)
        # [batch_size]
        pool_size = pool_size.masked_fill(pool_size < self.min_poolsize, self.min_poolsize)
        score_sort, node_idx = torch.sort(score_distribution, descending=True, dim=1)
        node_idx = node_idx.cpu()

        embedding = []
        attention = []
        knowledge_pool = []
        for idx in range(batch_size):
            sample_nodes = nodes[idx][node_idx[idx]]
            sample_nodes = sample_nodes[sample_nodes != ""]
            sample_size = 0
            sample_knowledge = []
            i=0
            while sample_size < self.base_poolsize and i < len(sample_nodes):
                tmp = self.environment.get_knowledge_text([sample_nodes[i]])
                sample_knowledge.extend(tmp)
                sample_size += len(tmp[0])
                i += 1
            sample_nodes = sample_nodes[0:i]
            sample_score = score_sort[idx][0:i]
            if sample_size > self.base_poolsize:
                diff = len(sample_knowledge[-1]) - (sample_size - self.base_poolsize)
                sample_knowledge[-1] = sample_knowledge[-1][0:diff]

            sample_embedding = self.environment.get_knowledge_embedding(sample_nodes)
            sample_embedding = torch.cat([d["embedding"] for d in sample_embedding])[0:self.base_poolsize]

            raw_size = torch.as_tensor([len(n) for n in sample_knowledge], device=self.device)
            att = sample_score.repeat_interleave(raw_size)
            attention.append(att)
            sample_knowledge = sum(sample_knowledge, [])
            knowledge_pool.append(sample_knowledge)
            embedding.append(sample_embedding)

        activations, knowledge_score = self.knowledge_score(candidate=embedding,
                                                            multiplier=agent_state.unsqueeze(1),
                                                            attention=attention)
        label = torch.zeros(knowledge_score.shape, device=self.device)

        pool = []
        raw_pool = []
        k = torch.min(pool_size, torch.as_tensor([len(p) for p in knowledge_pool], device=self.device))
        mean_k = torch.mean(k.float())
        for i in range(batch_size):
            values, indices = torch.sort(knowledge_score[i], descending=True)
            indices = indices[0:len(knowledge_pool[i])].cpu()
            tmp = np.array(knowledge_pool[i])[indices].tolist()
            pool.append(tmp[0:k[i]])
            raw_pool.append(tmp)

            label[i][0:len(attention[i])] = attention[i] / 10
            for g in gold[i]:
                if g in knowledge_pool[i]:
                    label[i][knowledge_pool[i].index(g)] += 1.0

        label = smooth_labels(label, smoothing_rate=self.label_smoothing_rate)
        nll = torch.sum(bi_tempered_logistic_loss(activations=activations, labels=label, t1=0.8, t2=1.2))

        return pool, nll, raw_pool, mean_k