import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv.gcn_conv import *


def prune_threshold(x, threshold=1e-3):
    idx_0 = torch.norm(x, dim=1)/x.shape[1] < threshold
    x[idx_0] = 0
    return x, idx_0


def prune_topk(x, k=0.2):
    num_0 = int(x.shape[0] * k)
    x_norm = torch.norm(x, dim=1)
    _, idx_0 = torch.topk(x_norm, num_0)
    x[idx_0] = 0
    return x, idx_0


class GCNConvThr(pyg_nn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(GCNConvThr, self).__init__(*args, **kwargs)
        self.threshold = 4e-3
        self.idx_keep = None
        self.prune = False

        # self.register_propagate_forward_pre_hook(self.prune_edge)
        self.register_message_forward_hook(self.get_edge_rm)
        # self.register_aggregate_forward_hook(self.propagate_forward_print)
        # self.register_propagate_forward_hook(self.propagate_forward_print)

    def prune_edge(self, module, inputs): # -> None or inputs
        '''hook(self, (edge_index, size, kwargs))
        Called before propagate().
            E.g. in GCNConv, after `edge_index, edge_weight = gcn_norm()` and
            `x = self.lin(x)`

        Args:
        if not is_sparse(edge_index):
            edge_index [2, m]: start and end index of each edge
            edge_weight [m, F]: value of each edge pending distribute and message
        if is_sparse(edge_index):
            edge_index [n, n]: weighted (normalized) adj matrix
        '''
        edge_index, size, kwargs = inputs
        x, edge_weight = kwargs['x'], kwargs['edge_weight']

        if self.idx_keep is not None:
            edge_index = edge_index[:, self.idx_keep]
            edge_weight = edge_weight[self.idx_keep]
        return edge_index, size, {'x': x, 'edge_weight': edge_weight}

    def get_edge_rm(self, module, inputs, output): # -> None or output
        '''res = hook(self, ({'x_j', 'edge_weight'}, ), output)
        Applicable only if not is_sparse(edge_index)
        Called in propagate(), after `out = self.message(**msg_kwargs)`
            E.g. in GCNConv, after normalization: `edge_weight.view(-1, 1) * edge_index`

        Args:
            inputs
                x_j [m, F]
                edge_weight [m]
            output [m, F]: message value of each edge after message and pending aggregate
        '''
        # print(inputs, inputs[0]['x_j'].shape)
        # print(output, output.shape)
        # print(output.view(-1).cpu().histogram(bins=20)[0], end=' ')
        # print(torch.norm(output, dim=1).cpu())
        if self.prune:
            if self.training:
                idx_0 = torch.norm(output, dim=1)/output.shape[1] < self.threshold
                output[idx_0] = 0
                self.idx_keep = torch.where(~idx_0)[0]
                print(f"  keep: {self.idx_keep.shape}")
            else:
                idx_0 = torch.ones(output.shape[0], dtype=torch.bool)
                idx_0[self.idx_keep] = False
                output[idx_0] = 0
                print(f"Infer: {self.idx_keep.shape}")

    def propagate_forward_print(self, module, inputs, output):
        '''hook(self, (edge_index, size, kwargs), out)'''
        print(inputs[0], inputs[0].shape, inputs[1])
        print(output, output.shape)

    # def reset_parameters(self):
    #     super().reset_parameters()
    #     self.lin.reset_parameters()
    #     pyg_nn.inits.zeros(self.bias)
    #     self._cached_edge_index = None
    #     self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, prune: bool = False):
        self.prune = prune
        # out = super().forward(self, x, edge_index, edge_weight)

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        if self.prune:
            edge_index = edge_index[:, self.idx_keep]
            if edge_weight is not None:
                edge_weight = edge_weight[self.idx_keep]

        return out, edge_index


class GCN(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass,
                 dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()

        cached = False
        add_self_loops = True
        improved = False
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConvThr(nfeat, nhidden, cached=cached, normalize=not save_mem,
                       add_self_loops=add_self_loops, improved=improved))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhidden))
        for _ in range(nlayer - 2):
            self.convs.append(
                GCNConvThr(nhidden, nhidden, cached=cached, normalize=not save_mem,
                           add_self_loops=add_self_loops, improved=improved))
            self.bns.append(nn.BatchNorm1d(nhidden))

        self.convs.append(
            GCNConvThr(nhidden, nclass, cached=cached, normalize=not save_mem,
                       add_self_loops=add_self_loops, improved=improved))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_idx, prune=False):
        for i, conv in enumerate(self.convs[:-1]):
            x, edge_idx = conv(x, edge_idx, prune=prune)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.convs[-1](x, edge_idx, prune=prune)
        return x
