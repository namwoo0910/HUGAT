import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GATConv
import dgl


# import utils


class HUGAT(nn.Module):
    def __init__(self, g, meta_paths, in_size, out_size, num_heads, dropout):
        super().__init__()
        self.han = HAN(meta_paths, in_size, 64, out_size, num_heads, dropout)

    def forward(self, g, h):
        self.emb = self.han.forward(g, h)
        return self.emb  # , self.emb_chk, self.emb_src, self.emb_dst, self.emb_geo

    def get_mob_loss(self, g, h, src_matrix, dst_matrix):
        self.emb, _ = self.han.forward(g, h)
        prediction_src, prediction_dst = self.predict_distribution(self.emb, self.emb)
        src_loss = -torch.multiply(src_matrix, prediction_src).sum(-1).sum(-1)
        dst_loss = -torch.multiply(dst_matrix, prediction_dst).sum(-1).sum(-1)
        return src_loss + dst_loss

    def predict_distribution(self, emb_src, emb_dst):
        src = pairwise_inner_product(emb_src, emb_dst).T
        p = nn.LogSoftmax(dim=1)
        prediction_src = p(src)
        dst = pairwise_inner_product(emb_dst, emb_src).T
        q = nn.LogSoftmax(dim=1)
        prediction_dst = q(dst)
        return prediction_src, prediction_dst

    def get_chk_loss(self, g, h, chk_hell):
        self.emb,_ = self.han.forward(g, h)
        chk_distribution = F.softmax(self.emb, dim=1)
        inner = Hellinger_pairwise(chk_distribution, chk_distribution)
        criterion_chk = nn.MSELoss()
        chk_loss = criterion_chk(inner, chk_hell)
        return chk_loss

    def get_land_loss(self, g, h, geo_hell):
        self.emb,_ = self.forward(g, h)
        land_distribution = F.softmax(self.emb, dim=1)
        inner = Hellinger_pairwise(land_distribution, land_distribution)
        criterion_land = nn.MSELoss()
        land_loss = criterion_land(inner, geo_hell)
        return land_loss

    def get_loss(self, g, h, src_matrix, dst_matrix, chk_hell, geo_hell):
        mob_loss = 0.1 * self.get_mob_loss(g, h, src_matrix, dst_matrix)
        chk_loss = 0.3 * self.get_chk_loss(g, h, chk_hell)
        land_loss = 0.6 * self.get_land_loss(g, h, geo_hell)
        return mob_loss + chk_loss + land_loss



def Hellinger_pairwise(a, b):
    hellinger_distance = ((1/2)**(1/2)) * torch.cdist(torch.sqrt(a), torch.sqrt(b), p=2)
    return hellinger_distance

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)
        self.lyr_norm = nn.LayerNorm(hidden_size * num_heads[-1])
        #self.importance = self.layers[-1].semantic_attention.beta
        #self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[-2],out_size, 1, dropout))
                                           # (N, K, D)
    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)#self.predict(self.lyr_norm(h))

def pairwise_inner_product(a, b):
    n, _ = list(a.size())
    b_ = torch.unsqueeze(b, 0)
    b_ = torch.tile(b_, [n, 1, 1])
    b_ = b_.permute(1, 0, 2)
    inner_product = torch.multiply(b_, a)
    inner_product = torch.sum(inner_product, axis=-1)
    return inner_product


class HANLayer(nn.Module):
    """
    HAN layer.

    Dimensions
    ---------
    N : number of nodes
    D : dimension of output feature
    M : cardinality of meta-pahts
    K : number of Multi-heads

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            # concatenate
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))  # (N, D*K)
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        emb= self.semantic_attention(semantic_embeddings)  # (N, D * K)
        return emb

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        '''
        q.T x tanh(Wz+b)
        '''
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)     # z : (N, M, D*K)
        )                                              # project(z) : (N, M, 1)

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        return (beta * z).sum(1)                    # (N, D * K)
