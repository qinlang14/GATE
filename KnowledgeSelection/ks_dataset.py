from typing import List, Dict, TypedDict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import read_json, data_name_to_path, absolute_path
from functools import partial
from tqdm import tqdm
import numpy as np


class Sample(TypedDict):
    Index: int
    Embedding: List[torch.Tensor]
    History: str
    Utterance: str
    Keywords: str
    Gold_Node: str
    Gold_Knowledge: List[str]
    Response: str
    Root: str
    Nodes: List[str]
    Adj_Matrix: Dict[str, List[int]]
    Label: List[float]


class KSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


class KSLoader:
    def __init__(self, opt):
        self.opt = opt
        self.data_path = data_name_to_path(opt["data_name"])
        self.loader_dict = {}
        self.loader_length = {}

    def text_embedding_combine(self, json_path, embed_path):
        data = []
        text = read_json(json_path)["data"]
        embedding = torch.load(absolute_path(embed_path), map_location=torch.device('cuda:0'))
        assert len(text) == len(embedding)

        # Extract a portion of the data for procedure testing. size:(0,1]
        length = int(self.opt["size"] * len(text))
        text = text[0:length]
        embedding = embedding[0:length]

        for i in tqdm(range(length)):
            data.append(Sample(Index=i + 1, Embedding=embedding[i], Gold_Node=text[i]["Gold_Node"],
                               Gold_Knowledge=text[i]["Gold_Knowledge"], Root=text[i]["Root"],
                               Nodes=text[i]["Nodes"], Adj_Matrix=text[i]["Adj_Matrix"],
                               Label=text[i]["Node_Label"], Keywords=text[i]["Keywords"],
                               Utterance=text[i]["Utterance"], History=text[i]["History"],
                               Response=text[i]["Response"]))

        return data

    def dialogue_collate(self, batch, device, train, rollout=1):
        embedding, gold_k, gold_n, root, nodes, adj, label, keywords, utterance, history, response = [[] for _ in range(11)]
        for example in batch:
            embedding.append(example["Embedding"])
            gold_k.append(example["Gold_Knowledge"])
            gold_n.append(example["Gold_Node"])
            root.append(example["Root"])
            nodes.append(example["Nodes"])
            adj.append(torch.sparse_csr_tensor(crow_indices=example["Adj_Matrix"]["indptr"],
                                               col_indices=example["Adj_Matrix"]["indices"],
                                               values=len(example["Adj_Matrix"]["indices"]) * [1],
                                               device=device, dtype=torch.int))
            label.append(torch.as_tensor(example["Label"], device=device))
            keywords.append(example["Keywords"])
            utterance.append(example["Utterance"])
            history.append(example["History"])
            response.append(example["Response"])

        embedding = torch.stack(embedding, dim=0)  # tensor [B * 3 * H]

        max_n = max([len(n) for n in nodes])
        padded_nodes = []
        for arr in nodes:
            padded_arr = np.pad(arr, (0, max_n - len(arr)), constant_values='')
            padded_nodes.append(np.char.array(padded_arr))
        nodes = np.vstack(padded_nodes)  # np.array [B * max_len]

        # tensor [B * max_len * max_len]    max_len: nodes
        adj_matrix = torch.zeros((len(adj), max_n, max_n), dtype=torch.int, device=device)

        for i, a in enumerate(adj):
            adj_matrix[i, :a.shape[0], :a.shape[1]] = a.to_dense()

        # tensor [B * max_len]    max_len: nodes
        label = nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=0)

        keywords = np.array(keywords)
        utterance = np.array(utterance)

        if train:
            embedding = embedding.repeat(rollout, 1, 1)
            gold_k = rollout * gold_k
            gold_n = rollout * gold_n
            root = rollout * root
            keywords = np.tile(keywords, rollout)
            utterance = np.tile(utterance, rollout)
            nodes = np.tile(nodes, (rollout, 1))
            adj_matrix = adj_matrix.repeat(rollout, 1, 1)
            label = label.repeat(rollout, 1)

        return {"embedding": embedding, "gold_k": gold_k, "gold_n": gold_n, "root": root, "nodes": nodes,
                "adj": adj_matrix, "label": label,
                "keywords": keywords, "utterance": utterance, "history": history, "response": response}

    def get_loader(self, topic_split=True, train=True):
        print(f"Get dataloaders for {self.opt['data_name']}.")
        if not topic_split and self.opt["data_name"] == "WoW":
            print("WoW only has Topic Split as Original Split.")
            return None

        for json_path, embed_path in self.data_path:
            if topic_split and "Normal" in json_path:
                # skip OpendialKG normal splits if use topic split
                continue
            data = self.text_embedding_combine(json_path, embed_path)
            dataset = KSDataset(data)
            if "train" in json_path and train:
                loader = DataLoader(dataset, batch_size=self.opt["batch_size"], shuffle=True,
                                    collate_fn=partial(self.dialogue_collate, device='cuda:0', train=True,
                                                       rollout=self.opt["rollouts"]))
            else:
                loader = DataLoader(dataset, batch_size=self.opt["rollouts"]*self.opt["batch_size"], shuffle=False,
                                    collate_fn=partial(self.dialogue_collate, device='cuda:0', train=False))

            # e.g. key = "test_unseen"
            key = json_path.split(".")[0].split("/")[-1]
            self.loader_dict[key] = loader
            self.loader_length[key] = len(dataset)

        key_order = ["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"]
        if not topic_split:
            key_order = ["train", "valid", "test"]
        self.loader_dict = {key: self.loader_dict[key] for key in key_order}

        return self.loader_dict, self.loader_length
