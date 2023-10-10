import torch
import torch.nn as nn
import torch.nn.functional as F
from module import RepresentationLayer
from utils import padding, smooth_labels, bi_tempered_logistic_loss
import math
import numpy as np
from typing import List, Optional, Tuple

class KnowledgeSelector(nn.Module):
    def __init__(self, hidden_dim, base_poolsize, min_poolsize, environment, mask):
        super().__init__()

        self.device = "cuda:0"
        self.hidden_dim = hidden_dim
        self.base_poolsize = base_poolsize
        self.min_poolsize = min_poolsize
        self.environment = environment
        self.mask = mask

        self.label_smoothing_rate = 0.15


        self.knowledge_layer = RepresentationLayer(in_features=self.hidden_dim, out_features=int(self.hidden_dim / 2))
        self.knowledge_score_layer = RepresentationLayer(in_features=1 * self.base_poolsize, out_features=self.base_poolsize)
        self.maxpool = nn.MaxPool1d(2)

    def forward(self, batch_input, agent_state, current_score, label_smoothing_rate):
        gold, nodes, keywords, utterance = batch_input
        self.label_smoothing_rate = label_smoothing_rate
        knowledge_pool, knowledge_nll, raw_kp, mean_k = self.knowledge_selection(gold, nodes, keywords, utterance, agent_state, current_score)

        return knowledge_pool, knowledge_nll, raw_kp, mean_k

    def knowledge_pad(self, inp, value=0):
        if inp.shape[1] < self.base_poolsize:
            inp = F.pad(inp, (0, self.base_poolsize - inp.shape[1], 0, 0), value=value)
        return inp

    def knowledge_score(self, candidate: List[torch.Tensor], multiplier: Optional[torch.Tensor] = None,
                        attention=None, keywords_rate=None):
        new, mask_matrix = padding(candidate, pad=0)

        new = self.knowledge_layer(new)
        # new = self.maxpool(new)
        new = torch.sum(multiplier * new, dim=-1) / math.sqrt(multiplier.shape[-1])
        att, _ = padding(attention, pad=0)
        max_n = att.shape[1]
        att = self.knowledge_pad(att)
        att = self.knowledge_score_layer(att)[:, 0:max_n]
        # keywords_rate = self.knowledge_pad(keywords_rate[1])
        # att = self.knowledge_score_layer(torch.cat([att, keywords_rate], dim=-1))[:, 0:max_n]
        new = att * new

        result = new.masked_fill(mask_matrix == 1, self.mask)
        score = torch.softmax(result, dim=-1)
        # score = torch.sigmoid(result)
        return result, score

    # @profile
    def fuzzy_match(self, keywords, utterance, pool, rate=1.0):
        assert len(keywords) == len(utterance) == len(pool)

        can_key = self.environment.get_key_token(keywords)
        can_utt = self.environment.get_utter_token(utterance)
        TP1 = []
        TP2 = []
        ref = [self.environment.get_knowledge_token(p) for p in pool]
        for i in range(len(can_key)):
            sample_tp1 = []
            sample_tp2 = []
            k = can_key[i]
            l_k = len(k)
            if l_k==0:
                l_k = 1
            u = can_utt[i]
            l_u = len(u)
            for r in ref[i]:
                sample_tp1.append(0.8 + rate * (len(k&r) / l_k))
                sample_tp2.append(0.8 + rate * (len(u&r) / l_u))
            TP1.append(sample_tp1)
            TP2.append(sample_tp2)
        TP1 = [torch.as_tensor(s1, device=self.device) for s1 in TP1]
        TP2 = [torch.as_tensor(s2, device=self.device) for s2 in TP2]
        keys,_ = padding(TP1, pad=0)
        utters,_ = padding(TP2, pad=0)

        return keys, utters

    def knowledge_selection(self, gold, nodes, keywords, utterance, agent_state: torch.Tensor,
                            score_distribution: torch.Tensor) -> Tuple[List[List[str]], torch.Tensor, List[List[str]]]:
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

        # keywords_rate = self.fuzzy_match(keywords, utterance, knowledge_pool)
        activations, knowledge_score = self.knowledge_score(candidate=embedding,
                                                            multiplier=agent_state.unsqueeze(1),
                                                            attention=attention)
        label = torch.zeros(knowledge_score.shape, device=self.device)

        pool = []
        raw_pool = []
        k = torch.min(pool_size, torch.as_tensor([len(p) for p in knowledge_pool], device=self.device))
        mean_k = torch.mean(k.float())
        for i in range(batch_size):
            # tmp_pool = {}
            # values, indices = torch.topk(knowledge_score[i], k[i])
            # indices = indices.cpu()
            # pool.append(np.array(knowledge_pool[i])[indices].tolist())
            #
            # values, indices = torch.sort(knowledge_score[i], descending=True)
            # indices = indices[0:len(knowledge_pool[i])].cpu()
            # raw_pool.append(np.array(knowledge_pool[i])[indices].tolist())
            values, indices = torch.sort(knowledge_score[i], descending=True)
            indices = indices[0:len(knowledge_pool[i])].cpu()
            tmp = np.array(knowledge_pool[i])[indices].tolist()
            pool.append(tmp[0:k[i]])
            # pool.append(tmp[0:25])
            raw_pool.append(tmp)

            label[i][0:len(attention[i])] = attention[i] / 10
            for g in gold[i]:
                if g in knowledge_pool[i]:
                    label[i][knowledge_pool[i].index(g)] += 1.0

        label = smooth_labels(label, smoothing_rate=self.label_smoothing_rate)
        # tmp = torch.max(label, dim=1)
        # nll = nll_loss(label, knowledge_score, scale=True)
        nll = torch.sum(bi_tempered_logistic_loss(activations=activations, labels=label, t1=0.8, t2=1.2))

        return pool, nll, raw_pool, mean_k