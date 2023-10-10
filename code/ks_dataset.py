from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import numpy as np

class Example4RL:
    def __init__(self,
                 Index: int,
                 Embedding: List[torch.Tensor],
                 Gold_K: List[str],
                 Gold_N: str,
                 Root: str,
                 Nodes: List[str],
                 Adj: Dict,
                 Label: List,
                 Keywords: str,
                 Utterance: str,
                 History: str,
                 Response: str):

        self.Index = Index
        self.Embedding = Embedding
        self.Gold_K = Gold_K
        self.Gold_N = Gold_N
        self.Root = Root
        self.Nodes = Nodes
        self.Adj = Adj
        self.Label = Label
        self.Keywords = Keywords
        self.Utterance = Utterance
        self.History = History
        self.Response = Response


def dialogue_process_for_KS(dataset, split, num):
    dialogue = []
    with open("{}/{}.json".format(dataset, split), "r", encoding='utf-8') as f:
        text = json.load(f)["data"][0:num]
    # embedding = torch.load("{}/{}_embedding.pth".format(dataset, split), map_location=torch.device('cpu'))[0:num]
    embedding = torch.load("{}/{}_embedding.pth".format(dataset, split), map_location=torch.device('cuda:0'))[0:num]
    assert len(text) == len(embedding)
    for i in tqdm(range(len(text))):
        dialogue.append(
            Example4RL(Index=i + 1, Embedding=embedding[i], Gold_N=text[i]["Gold_Node"], Gold_K=text[i]["Gold_Knowledge"],
                       Root=text[i]["Root"], Nodes=text[i]["Nodes"], Adj=text[i]["Adj_Matrix"], Label=text[i]["Node_Label"],
                       Keywords=text[i]["KeyWords"], Utterance=text[i]["Utterance"], History=text[i]["History"],
                       Response=text[i]["Response"]))
    return dialogue

class KSDataset(Dataset):
    def __init__(self, opt):
        self.dialogue = dialogue_process_for_KS(opt["dataset"], opt["split"], opt["num"])

    def __len__(self):
        return len(self.dialogue)

    def __getitem__(self, idx):
        sample = self.dialogue[idx]
        return sample


def dialogue_collate(batch, device, train, predict=False, rollout=1):
    embedding = []
    gold_k = []
    gold_n = []
    root = []
    nodes = []
    adj = []
    label = []
    keywords = []
    utterance = []
    if predict:
        history = []
        response = []
    for example in batch:
        embedding.append(example.Embedding)
        gold_k.append(example.Gold_K)
        gold_n.append(example.Gold_N)
        root.append(example.Root)
        nodes.append(example.Nodes)
        adj.append(torch.sparse_csr_tensor(crow_indices=example.Adj["indptr"], col_indices=example.Adj["indices"],
                                           values=len(example.Adj["indices"])*[1], device=device, dtype=torch.int))
        label.append(torch.as_tensor(example.Label, device=device))
        keywords.append(example.Keywords)
        utterance.append(example.Utterance)
        if predict:
            history.append(example.History)
            response.append(example.Response)

    embedding = torch.stack(embedding, dim=0)       # tensor [B * 3 * H]

    max_n = max([len(n) for n in nodes])
    padded_nodes = []
    for arr in nodes:
        padded_arr = np.pad(arr, (0, max_n - len(arr)), constant_values='')
        padded_nodes.append(np.char.array(padded_arr))
    nodes = np.vstack(padded_nodes)     # np.array [B * max_len]

    adj_matrix = torch.zeros((len(adj), max_n, max_n), dtype=torch.int, device=device)  # tensor [B * max_len * max_len]    max_len: nodes
    for i, a in enumerate(adj):
        adj_matrix[i, :a.shape[0], :a.shape[1]] = a.to_dense()
        # adj_matrix[i, :a.shape[0], :a.shape[1]] = torch.ones(a.shape)

    label = nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=0)  # tensor [B * max_len]    max_len: nodes
    keywords = np.array(keywords)
    utterance = np.array(utterance)

    if train:
        embedding = embedding.repeat(rollout,1,1)
        gold_k = rollout * gold_k
        gold_n = rollout * gold_n
        root = rollout*root
        keywords = np.tile(keywords, rollout)
        utterance = np.tile(utterance, rollout)
        nodes = np.tile(nodes, (rollout,1))
        adj_matrix = adj_matrix.repeat(rollout,1,1)
        label = label.repeat(rollout,1)
    if predict:
        return {"embedding": embedding, "gold_k": gold_k, "gold_n": gold_n, "root": root, "nodes": nodes, "adj": adj_matrix, "label": label,
                "keywords": keywords, "utterance": utterance, "history": history, "response": response}
    else:
        return {"embedding": embedding, "gold_k": gold_k, "gold_n": gold_n, "root": root, "nodes": nodes, "adj": adj_matrix, "label": label,
                "keywords": keywords, "utterance": utterance}