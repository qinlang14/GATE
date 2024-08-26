import numpy as np
import torch
from typing import List, Union
import torch.nn as nn
import torch.nn.functional as F
from utils import read_json, absolute_path


class Environment:
    def __init__(self, opt):
        path_prefix = f"Data/{opt['data_name']}/Preprocess/Intermediate/"
        self.knowledge_base = read_json(absolute_path(path_prefix + "knowledge_base.json"))
        self.knowledge_embedding = torch.load(absolute_path(path_prefix+"knowledge_embedding.pth"), map_location=opt["device"])

    def get_knowledge_text(self, entity: Union[List[str], np.ndarray]) -> List[List[str]]:
        text = []
        for e in entity:
            text.append(self.knowledge_base[e])
        return text

    def get_knowledge_embedding(self, entity: Union[List[str], np.ndarray]) -> np.ndarray:
        embedding = []
        for e in entity:
            embedding.append(self.knowledge_embedding[e])
        return np.array(embedding)

class ModalityAttentionLayer(nn.Module):
    def __init__(self, hidden_dim = 768):
        super(ModalityAttentionLayer, self).__init__()
        self.in_dim = hidden_dim
        self.out_dim = 8
        self.linear = nn.Linear(in_features=self.in_dim, out_features=self.out_dim, bias=True)

    def forward(self, inp):
        x = self.linear(inp)
        a = torch.tanh(x)
        alpha_m = F.softmax(a, dim=1)
        x_bar = torch.einsum('bih,bif->bhf', alpha_m, inp)

        return torch.mean(x_bar, dim=1)


class RepresentationLayer(nn.Module):
    def __init__(self, in_features = 768, out_features = 384):
        super(RepresentationLayer, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        self.linear2 = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.layernorm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(p=0.25)
        self.activation = nn.LeakyReLU(negative_slope=0.21)

    def forward(self, x):
        x = self.linear1(x)+x
        x = self.layernorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation(x)

        return x


class InputLayer(nn.Module):
    def __init__(self, in_features=768, out_features=384):
        super(InputLayer, self).__init__()
        self.modality = ModalityAttentionLayer(hidden_dim=in_features)
        self.maxpool = nn.MaxPool1d(2)
        self.sharelayer = nn.Linear(in_features, int(in_features/2))
        self.utterlayer = nn.Linear(in_features, int(out_features/2))
        self.keywlayer = nn.Linear(in_features, int(out_features/2))
        self.activation = nn.LeakyReLU(negative_slope=0.21)

    def forward(self, inp):
        mod = self.sharelayer(self.modality(inp))
        mod = self.activation(mod)
        utt = self.sharelayer(inp[:,1,:].squeeze())
        utt = self.activation(utt)
        key = self.sharelayer(inp[:,2,:].squeeze())
        key = self.activation(key)
        utt_mod = self.utterlayer(torch.cat([utt, mod], dim=-1))
        utt_mod = self.activation(utt_mod)
        key_mod = self.keywlayer(torch.cat([key, mod],dim=-1))
        key_mod = self.activation(key_mod)

        return torch.cat([utt_mod, key_mod], dim=-1)