import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GCNConvThr(pyg_nn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(GCNConvThr, self).__init__(*args, **kwargs)
        self.threshold = 1e-3
        self.edge_keep = None
        self.cnt = 0

        self.register_propagate_forward_pre_hook(self.prune_edge)
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

        if self.edge_keep is not None and self.cnt > 40:
            edge_index = edge_index[:, self.edge_keep]
            edge_weight = edge_weight[self.edge_keep]
        return edge_index, size, {'x': x, 'edge_weight': edge_weight}

    def get_edge_rm(self, module, inputs, output): # -> None or output
        '''res = hook(self, ({'x_j', 'edge_index'}, ), output)
        Applicable only if not is_sparse(edge_index)
        Called in propagate(), after `out = self.message(**msg_kwargs)`
            E.g. in GCNConv, after normalization: `edge_weight.view(-1, 1) * edge_index`

        Args:
            output [m, F]: message value of each edge after message and pending aggregate
        '''
        # print(inputs, inputs[0]['x_j'].shape)
        # print(output, output.shape)
        # print(output.view(-1).cpu().histogram(bins=20)[0])
        idx_0 = torch.norm(output, dim=1)/output.shape[1] < self.threshold
        self.edge_keep = torch.where(~idx_0)[0]
        self.cnt += 1
        if self.cnt > 40:
            output[idx_0] = 0
        num_0 = output.shape[0] - len(self.edge_keep)
        # print(num_0, num_0/output.shape[0])

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


class GCN(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass,
                 dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()

        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConvThr(nfeat, nhidden, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhidden))
        for _ in range(nlayer - 2):
            self.convs.append(
                GCNConvThr(nhidden, nhidden, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(nhidden))

        self.convs.append(
            GCNConvThr(nhidden, nclass, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

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
