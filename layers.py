import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv.gcn_conv import *

from utils.logger import ThrLayerLogger
from prunes import ThrInPruningMethod, prune


class GCNConvThr(pyg_nn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(GCNConvThr, self).__init__(*args, **kwargs)
        self.threshold_a = 2e-3
        self.threshold_w = 1e-4
        self.idx_keep = None
        self.logger_a = ThrLayerLogger()
        self.logger_w = ThrLayerLogger()

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
        if self.training:
            idx_0 = torch.norm(output, dim=1)/output.shape[1] < self.threshold_a
            idx_0[self.idx_lock] = False
            output[idx_0] = 0
            self.idx_keep = torch.where(~idx_0)[0]
        # else:
        #     idx_0 = torch.ones(output.shape[0], dtype=torch.bool)
        #     idx_0[self.idx_keep] = False
        #     idx_0[self.idx_lock] = False
        #     output[idx_0] = 0

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
                edge_weight: OptTensor = None, node_lock: OptTensor = None):
        """Args:
            x (Tensor [n, F_in]): node feature matrix
        """
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

        if self.training:
            '''Shape:
                threshold_wi [F_in]
                self.lin.weight [F_out, F_in]
            '''
            threshold_wi = self.threshold_w / (torch.norm(x, dim=0)/x.shape[0])
            ThrInPruningMethod.apply(self.lin, 'weight', threshold_wi)
            x = self.lin(x)

            self.logger_w.numel_before = self.lin.weight.numel()
            self.logger_w.numel_after = torch.sum(self.lin.weight != 0).item()
        else:
            x = self.lin(x)
            if prune.is_pruned(self.lin):
                prune.remove(self.lin, 'weight')

        self.idx_lock = torch.where(edge_index[1].unsqueeze(0) == torch.tensor(node_lock, device=edge_index.device).unsqueeze(1))[1]
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        if self.training:
            self.logger_a.numel_before = edge_index.shape[1]
            self.logger_a.numel_after = self.idx_keep.shape[0]

            edge_index = edge_index[:, self.idx_keep]
            if edge_weight is not None:
                edge_weight = edge_weight[self.idx_keep]

            print(f"  A: {self.logger_a}, W: {self.logger_w}")

        return out, edge_index


class GINConvRaw(pyg_nn.GINConv):
    def __init__(self, in_channels: int, out_channels: int,
                 eps: float = 0., train_eps: bool = False, **kwargs):
        nn_default = pyg_nn.MLP(
            [in_channels, out_channels],
        )
        super(GINConvRaw, self).__init__(nn_default, eps, train_eps, **kwargs)


class GATv2ConvRaw(pyg_nn.GATv2Conv):
    def __init__(self, in_channels: int, out_channels: int, depth: int,
                 heads: int = 1, concat: bool = True, **kwargs):
        heads = 1 if depth == 0 else heads
        concat = (depth > 0)
        if concat:
            out_channels = out_channels // heads
        super(GATv2ConvRaw, self).__init__(in_channels, out_channels, heads, concat, **kwargs)


layer_dict = {
    'gcn': pyg_nn.GCNConv,
    'gcn_thr': GCNConvThr,
    'gin': GINConvRaw,
    'gat': GATv2ConvRaw,
}
