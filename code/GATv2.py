import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionV2Layer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.25,
                 leaky_relu_negative_slope: float = 0.21,
                 share_weights: bool = False,
                 precision:str = "32"):
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        if precision == "16":
            self.mask = -6e4
        else:
            self.mask = -1e6
        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=2)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        batch_size = h.shape[0]
        n_nodes = h.shape[1]
        g_l = self.linear_l(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        g_l_repeat = g_l.repeat(1,n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=1)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(batch_size, n_nodes, n_nodes, self.n_heads, self.n_hidden)
        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == n_nodes
        assert adj_mat.shape[3] == 1 or adj_mat.shape[3] == self.n_heads
        e = e.masked_fill(adj_mat == 0, self.mask)
        a = self.softmax(e)
        a = self.dropout(a)
        attn_res = torch.einsum('bijh,bjhf->bihf', a, g_r)
        if self.is_concat:
            return attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=2)


class GATv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, precision):
        super(GATv2, self).__init__()

        self.num_heads = num_heads
        self.layer1 = GraphAttentionV2Layer(in_features=in_dim, out_features=hidden_dim, n_heads=num_heads, precision=precision)
        self.layer2 = GraphAttentionV2Layer(in_features=hidden_dim, out_features=hidden_dim, n_heads=num_heads, precision=precision)
        self.layer3 = GraphAttentionV2Layer(in_features=hidden_dim, out_features=hidden_dim, n_heads=num_heads, precision=precision)
        self.layer4 = nn.Linear(in_features=hidden_dim, out_features=out_dim)
        self.layernorm1 = nn.LayerNorm(in_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj:torch.Tensor, glob = False):
        # adj = adj.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
        adj = adj.unsqueeze(-1)
        x = self.layer1(x, adj)
        x = self.layernorm2(x)
        x = F.dropout(x, p=0.25)
        x = self.layer2(x, adj) + x
        x = self.layernorm2(x)

        if glob:
            return x
        else:
            x = F.dropout(x, p=0.25)
            x = self.layer3(x, adj) + x
            x = self.layernorm2(x)
            x = F.dropout(x, p=0.25)
            x = self.layer4(x).squeeze()
            x = F.leaky_relu(x, negative_slope=0.21)

            return x