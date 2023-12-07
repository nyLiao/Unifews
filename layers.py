import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv.gcn_conv import *
from torch_geometric.nn.conv.gatv2_conv import *

from utils.logger import ThrLayerLogger
from utils.prunes import ThrInPrune, prune


def identity_n_norm(edge_index, edge_weight=None, num_nodes=None,
                    rnorm=None, diag=1., dtype=torch.float32):
    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)
        if diag is not None:
            edge_index = torch_sparse.fill_diag(edge_index, diag)
        if rnorm is not None:
            # TODO: r-norm
            deg = torch_sparse.sum(adj_t, dim=1)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
            adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))
        return edge_index

    if isinstance(edge_index, Tensor):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        if diag is not None:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, diag, num_nodes)
        if rnorm is None:
            if edge_weight is None:
                return edge_index
        else:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
            row, col = edge_index[0], edge_index[1]
            idx = col
            deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight

    raise NotImplementedError()

# ==========
class GCNConvRaw(pyg_nn.GCNConv):
    def forward(self, x: Tensor, edge_tuple: Tuple, **kwargs):
        (edge_index, edge_weight) = edge_tuple
        return super().forward(x, edge_index, edge_weight)


class GCNConvThr(pyg_nn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(GCNConvThr, self).__init__(*args, **kwargs)
        self.threshold_a = 2e-3
        self.threshold_w = 1e-2
        self.idx_keep = None
        self.logger_a = ThrLayerLogger()
        self.logger_w = ThrLayerLogger()
        self.prune_lst = [self.lin]

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
                x_j [m, F]: node feature mapped by edge source nodes
                edge_weight [m]
            output [m, F]: message value of each edge after message and pending aggregate
        '''
        x_j = inputs[0]['x_j']
        # TODO: infer output much smaller: check batchnorm
        import numpy as np
        hist, bins = output.view(-1).cpu().detach().histogram(bins=10)
        hist = hist.numpy()/output.numel()
        np.set_printoptions(precision=2, suppress=True, formatter={'float': '{: 0.2f}'.format})
        # print(f' % {hist}')
        # print(f' | {bins.numpy()}')
        print(x_j.shape, x_j.norm(dim=0).cpu().detach().numpy(), x_j.norm(dim=1).cpu().detach().numpy())

        # Edge pruning
        # TODO: change to relative to x norm
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

    def forward(self, x: Tensor, edge_tuple: Tuple,
                node_lock: OptTensor = None, verbose: bool = False):
        """Args:
            x (Tensor [n, F_in]): node feature matrix
        """
        (edge_index, edge_weight) = edge_tuple
        # Weight pruning
        if self.training:
            '''Shape:
                threshold_wi [F_in]
                self.lin.weight [F_out, F_in]
            '''
            threshold_wi = self.threshold_w / torch.norm(x, dim=0)
            ThrInPrune.apply(self.lin, 'weight', threshold_wi)
            x = self.lin(x)

            self.logger_w.numel_before = self.lin.weight.numel()
            self.logger_w.numel_after = torch.sum(self.lin.weight != 0).item()
        else:
            x = self.lin(x)
            # if prune.is_pruned(self.lin):
            #     prune.remove(self.lin, 'weight')

        # print(x.shape, x.norm(dim=0).cpu().detach().numpy(), x.norm(dim=1).cpu().detach().numpy())
        # Lock edges ending at node_lock
        self.idx_lock = torch.where(edge_index[1].unsqueeze(0) == node_lock.to(edge_index.device).unsqueeze(1))[1]
        # Lock self-loop edges
        idx_diag = torch.where(edge_index[0] == edge_index[1])[0]
        self.idx_lock = torch.cat((self.idx_lock, idx_diag))
        self.idx_lock = torch.unique(self.idx_lock)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out = out + self.bias

        # Edge removal for next layer
        if self.training:
            self.logger_a.numel_before = edge_index.shape[1]
            self.logger_a.numel_after = self.idx_keep.shape[0]
            if verbose:
                print(f"  A: {self.logger_a}, W: {self.logger_w}")

            edge_index = edge_index[:, self.idx_keep]
            edge_weight = edge_weight[self.idx_keep]

        return out, (edge_index, edge_weight)


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
        self.prune_lst = [self.lin_l, self.lin_r]


class GATv2ConvThr(GATv2ConvRaw):
    def __init__(self, *args, **kwargs):
        super(GATv2ConvThr, self).__init__(*args, **kwargs)
        self.threshold_a = 5e-4
        self.threshold_w = 1e-2
        self.idx_keep = None
        self.logger_a = ThrLayerLogger()
        self.logger_w = ThrLayerLogger()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                node_lock: OptTensor = None, verbose: bool = False):
        """Shape:
            x: [n, F_in]
            self.lin_l.weight: [F_out, F_in]
            x_l: [n, H, F_out//H]
        """
        H, C = self.heads, self.out_channels
        x_l: OptTensor = None
        x_r: OptTensor = None
        assert x.dim() == 2

        # Weight pruning
        if self.training:
            threshold_wi = self.threshold_w / torch.norm(x, dim=0)
            ThrInPrune.apply(self.lin_l, 'weight', threshold_wi)
            ThrInPrune.apply(self.lin_r, 'weight', threshold_wi)
            x_l = self.lin_l(x).view(-1, H, C)
            x_r = x_l if self.share_weights else self.lin_r(x).view(-1, H, C)

            self.logger_w.numel_before = self.lin_l.weight.numel() + self.lin_r.weight.numel()
            self.logger_w.numel_after = torch.sum(self.lin_l.weight != 0).item() + torch.sum(self.lin_r.weight != 0).item()
        else:
            x_l = self.lin_l(x).view(-1, H, C)
            x_r = x_l if self.share_weights else self.lin_r(x).view(-1, H, C)
            # if prune.is_pruned(self.lin_l):
            #     prune.remove(self.lin_l, 'weight')
            #     prune.remove(self.lin_r, 'weight')

        # Lock edges ending at node_lock
        self.idx_lock = torch.where(edge_index[1].unsqueeze(0) == node_lock.to(edge_index.device).unsqueeze(1))[1]
        # Lock self-loop edges
        idx_diag = torch.where(edge_index[0] == edge_index[1])[0]
        self.idx_lock = torch.cat((self.idx_lock, idx_diag))
        self.idx_lock = torch.unique(self.idx_lock)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, size=None)

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
            if verbose:
                print(f"  A: {self.logger_a}, W: {self.logger_w}")

            edge_index = edge_index[:, self.idx_keep]
            # if edge_attr is not None:
            #     edge_attr = edge_attr[self.idx_keep]

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
    'gcn': GCNConvRaw,
    'gcn_thr': GCNConvThr,
    'gin': GINConvRaw,
    'gat': GATv2ConvRaw,
    'gat_thr': GATv2ConvThr,
}
