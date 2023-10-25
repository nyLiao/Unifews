import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv.gcn_conv import *
from torch_geometric.nn.conv.gatv2_conv import *

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

        # Edge pruning
        if self.training:
            mask_0 = torch.norm(output, dim=1)/output.shape[1] < self.threshold_a
            mask_0[self.idx_lock] = False
            output[mask_0] = 0
            self.idx_keep = torch.where(~mask_0)[0]
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
                edge_weight: OptTensor = None,
                node_lock: OptTensor = None, verbose: bool = False):
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

        # Weight pruning
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

        self.idx_lock = torch.where(edge_index[1].unsqueeze(0) == node_lock.to(edge_index.device).unsqueeze(1))[1]
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        # Edge removal for next layer
        if self.training:
            self.logger_a.numel_before = edge_index.shape[1]
            self.logger_a.numel_after = self.idx_keep.shape[0]

            edge_index = edge_index[:, self.idx_keep]
            # if edge_weight is not None:
            #     edge_weight = edge_weight[self.idx_keep]

            if verbose:
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


class GATv2ConvThr(GATv2ConvRaw):
    def __init__(self, *args, **kwargs):
        super(GATv2ConvThr, self).__init__(*args, **kwargs)
        self.threshold_a = 5e-4
        self.threshold_w = 2e-5
        self.idx_keep = None
        self.logger_a = ThrLayerLogger()
        self.logger_w = ThrLayerLogger()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                node_lock: OptTensor = None, verbose: bool = False):
        H, C = self.heads, self.out_channels
        x_l: OptTensor = None
        x_r: OptTensor = None
        assert x.dim() == 2

        # Weight pruning
        if self.training:
            threshold_wi = self.threshold_w / (torch.norm(x, dim=0)/x.shape[0])
            ThrInPruningMethod.apply(self.lin_l, 'weight', threshold_wi)
            ThrInPruningMethod.apply(self.lin_r, 'weight', threshold_wi)

            x_l = self.lin_l(x).view(-1, H, C)
            x_r = x_l if self.share_weights else self.lin_r(x).view(-1, H, C)

            self.logger_w.numel_before = self.lin_l.weight.numel() + self.lin_r.weight.numel()
            self.logger_w.numel_after = torch.sum(self.lin_l.weight != 0).item() + torch.sum(self.lin_r.weight != 0).item()
        else:
            x_l = self.lin_l(x).view(-1, H, C)
            x_r = x_l if self.share_weights else self.lin_r(x).view(-1, H, C)

            if prune.is_pruned(self.lin_l):
                prune.remove(self.lin_l, 'weight')
                prune.remove(self.lin_r, 'weight')

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError()

        self.idx_lock = torch.where(edge_index[1].unsqueeze(0) == node_lock.to(edge_index.device).unsqueeze(1))[1]
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias

        # Edge removal for next layer
        if self.training:
            self.logger_a.numel_before = edge_index.shape[1]
            self.logger_a.numel_after = self.idx_keep.shape[0]

            edge_index = edge_index[:, self.idx_keep]
            # if edge_attr is not None:
            #     edge_attr = edge_attr[self.idx_keep]

            if verbose:
                print(f"  A: {self.logger_a}, W: {self.logger_w}")

        return out, edge_index

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        """Args:
            x_j, x_i (Tensor [m, H, C])
            self.att (Tensor [1, H, C])
            alpha (Tensor [m, H])
        """
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)

        # Edge pruning
        if self.training:
            threshold_aj = self.threshold_a / (torch.norm(x_j, dim=[1,2])/x_j.shape[1]/x_j.shape[2])
            mask_0 = torch.norm(alpha, dim=1)/alpha.shape[1] < threshold_aj
            mask_0[self.idx_lock] = False
            alpha[mask_0] = 0
            self.idx_keep = torch.where(~mask_0)[0]

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


layer_dict = {
    'gcn': pyg_nn.GCNConv,
    'gcn_thr': GCNConvThr,
    'gin': GINConvRaw,
    'gat': GATv2ConvRaw,
    'gat_thr': GATv2ConvThr,
}
