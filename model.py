import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GCN(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass,
                 dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()

        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            pyg_nn.GCNConv(nfeat, nhidden, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhidden))
        for _ in range(nlayer - 2):
            self.convs.append(
                pyg_nn.GCNConv(nhidden, nhidden, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(nhidden))

        self.convs.append(
            pyg_nn.GCNConv(nhidden, nclass, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_idx):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_idx)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_idx)
        return x
