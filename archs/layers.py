import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv.gcn_conv import *
from torch_geometric.nn.conv.gatv2_conv import *

from utils.logger import LayerNumLogger
from .prunes import ThrInPrune, prune


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
class ConvThr(nn.Module):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super(ConvThr, self).__init__(*args, **kwargs)
        self.threshold_a = thr_a
        self.threshold_w = thr_w
        self.idx_keep = torch.Tensor([])
        self.prune_lst = []
        """
            pruneall: regrow and fully prune
            pruneinc: keep previous pruning and apply prune
            keep: keep previous pruning
            full: use full without pruning
        """
        self.scheme_a = 'full'
        self.scheme_w = 'full'

        # self.register_aggregate_forward_hook(self.propagate_forward_print)
        # self.register_propagate_forward_hook(self.propagate_forward_print)

    def propagate_forward_print(self, module, inputs, output):
        """hook(self, (edge_index, size, kwargs), out)"""
        print(inputs[0], inputs[0].shape, inputs[1])
        print(output, output.shape)

    def get_idx_lock(self, edge_index, node_lock):
        # <<<<<<<<<< performance sensitive
        # Lock edges ending at node_lock
        idx_lock = torch.tensor([], dtype=torch.int32).to(edge_index.device)
        bs = int(2**28 / edge_index.shape[1])
        for i in range(0, node_lock.shape[0], bs):
            batch = node_lock[i:min(i+bs, node_lock.shape[0])].to(edge_index.device)
            idx_lock = torch.cat((idx_lock, torch.where(edge_index[1].unsqueeze(0) == batch.unsqueeze(1))[1]))
        # Lock self-loop edges
        idx_diag = torch.where(edge_index[0] == edge_index[1])[0].to(idx_lock.device)
        idx_lock = torch.cat((idx_lock, idx_diag))
        return torch.unique(idx_lock)


class GCNConvRaw(pyg_nn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(GCNConvRaw, self).__init__(*args, **kwargs)
        self.logger_a = LayerNumLogger()        # sparsity of adjacancy matrix
        self.logger_w = LayerNumLogger()        # sparsity of weight matrix
        self.logger_in = LayerNumLogger()       # sparsity of node feature matrix
        self.logger_msg = LayerNumLogger()      # sparsity of message matrix

    def forward(self, x: Tensor, edge_tuple: Tuple, **kwargs):
        (edge_index, edge_weight) = edge_tuple
        self.logger_a.numel_after = edge_index.shape[1]
        self.logger_w.numel_after = self.lin.weight.numel()
        # self.logger_in.numel_before = x.numel()
        # self.logger_in.numel_after = torch.sum(x != 0).item()
        return super().forward(x, edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, (edge_index, edge_weight) = input
        x_out = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_index.shape[1]

        # Linear
        flops_bias = f_out if module.lin.bias is not None else 0
        # module.__flops__ += int(f_in * f_out * n * module.logger_n.ratio)
        module.__flops__ += int(f_in * f_out * n)
        module.__flops__ += flops_bias * n
        # Message
        # module.__flops__ += int(f_in * m * module.logger_m.ratio)
        module.__flops__ += f_in * m


class GCNConvThr(ConvThr, GCNConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super(GCNConvThr, self).__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.lin]

        # self.register_propagate_forward_pre_hook(self.prune_on_ew)
        self.register_message_forward_hook(self.prune_on_msg)

    def prune_on_ew(self, module, inputs): # -> None or inputs
        """hook(self, (edge_index, size, kwargs))
        Called before propagate().
            E.g. in GCNConv, after `edge_index, edge_weight = gcn_norm()` and
            `x = self.lin(x)`

        Apply pruning on edge_weight based on node feature norm
        Args:
        if not is_sparse(edge_index):
            edge_index [2, m]: start and end index of each edge
            edge_weight [m, F]: value of each edge pending distribute and message
        if is_sparse(edge_index):
            edge_index [n, n]: weighted (normalized) adj matrix
        """
        edge_index, size, kwargs = inputs
        x, edge_weight = kwargs['x'], kwargs['edge_weight']
        return edge_index, size, {'x': x, 'edge_weight': edge_weight}

    def prune_on_msg(self, module, inputs, output): # -> None or output
        """res = hook(self, ({'x_j', 'edge_weight'}, ), output)
        Applicable only if not is_sparse(edge_index)
        Called in propagate(), after `out = self.message(**msg_kwargs)`
            E.g. in GCNConv, after normalization: `edge_weight.view(-1, 1) * edge_index`

        Apply pruning on message based on message norm
        Args:
            inputs
                x_j [m, F]: node feature mapped by edge source nodes
                edge_weight [m]
            output [m, F]: message of each edge after message() and pending aggregate
        """
        # Edge pruning
        if self.scheme_a in ['pruneall', 'pruneinc']:
            if self.scheme_a == 'pruneinc':
                raise NotImplementedError()
                # <<<<<<<<<< NOTE: previous idx may change when l>1
                self.idx_keep = self.idx_keep.to(output.device)
                mask_0 = torch.ones(output.shape[0], dtype=torch.bool, device=output.device)
                mask_0[self.idx_keep] = False
                output[mask_0] = 0
            else:
                mask_0 = torch.zeros(output.shape[0], dtype=torch.bool, device=output.device)
            # self.logger_msg.numel_before = output.numel()
            # self.logger_msg.numel_after = torch.sum(output != 0).item()

            norm_feat_msg = torch.norm(output, dim=1)       # each entry accross all features
            norm_all_msg = torch.norm(norm_feat_msg, dim=None, p=1)/output.shape[0]
            mask_cmp = norm_feat_msg < self.threshold_a * norm_all_msg
            # mask_cmp = norm_feat < self.threshold_a * self.norm_all_node
            mask_0 = torch.logical_or(mask_0, mask_cmp)
            mask_0[self.idx_lock] = False
            output[mask_0] = 0
            self.idx_keep = torch.where(~mask_0)[0]
        # >>>>>>>>>>
        elif self.scheme_a == 'keep':
            # self.logger_msg.numel_before = output.numel()
            # self.logger_msg.numel_after = torch.sum(output != 0).item()
            mask_0 = torch.ones(output.shape[0], dtype=torch.bool)
            mask_0[self.idx_keep] = False
            mask_0[self.idx_lock] = False
            output[mask_0] = 0
        return output

    def forward(self, x: Tensor, edge_tuple: Tuple,
                node_lock: OptTensor = None, verbose: bool = False):
        """Shape:
            x (Tensor [n, F_in]): node feature matrix
            threshold_wi [F_in]
            self.lin.weight [F_out, F_in]
        """
        (edge_index, edge_weight) = edge_tuple
        # self.logger_in.numel_before = x.numel()
        # self.logger_in.numel_after = torch.sum(x != 0).item()

        # Weight pruning
        if self.scheme_w in ['pruneall', 'pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.lin):
                    prune.remove(self.lin, 'weight')
            norm_node_in = torch.norm(x, dim=0)
            norm_all_in = torch.norm(norm_node_in, dim=None)/x.shape[1]
            if norm_all_in > 1e-8:
                threshold_wi = self.threshold_w * norm_all_in / norm_node_in
                ThrInPrune.apply(self.lin, 'weight', threshold_wi)
            x = self.lin(x)
        elif self.scheme_w == 'keep':
            x = self.lin(x)
        elif self.scheme_w == 'full':
            raise NotImplementedError()
        self.logger_w.numel_before = self.lin.weight.numel()
        self.logger_w.numel_after = torch.sum(self.lin.weight != 0).item()

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        self.idx_lock = self.get_idx_lock(edge_index, node_lock)
        # self.norm_all_node = torch.norm(x, dim=None) / x.shape[0]
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out = out + self.bias

        # Edge removal for next layer
        if self.scheme_a in ['pruneall', 'pruneinc', 'keep']:
            self.logger_a.numel_before = edge_index.shape[1]
            self.logger_a.numel_after = self.idx_keep.shape[0]
            # if verbose:
            #     print(f"  A: {self.logger_a}, W: {self.logger_w}")

        # >>>>>>>>>>
            edge_index = edge_index[:, self.idx_keep]
            edge_weight = edge_weight[self.idx_keep]
        else:
            self.logger_a.numel_before = edge_index.shape[1]
            self.logger_a.numel_after = edge_index.shape[1]

        with torch.cuda.device(edge_index.device):
            torch.cuda.empty_cache()
        return out, (edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, _ = input
        x_out, (edge_index, edge_weight) = output
        f_in, f_out = x_in.shape[-1], x_out.shape[-1]
        n, m = x_in.shape[0], edge_index.shape[1]

        # Linear
        flops_bias = f_out if module.lin.bias is not None else 0
        module.__flops__ += int((f_in * f_out * module.logger_w.ratio + flops_bias) * n)
        # Message
        module.__flops__ += f_in * (m - n)


class GATv2ConvRaw(pyg_nn.GATv2Conv):
    def __init__(self, in_channels: int, out_channels: int, depth: int,
                 heads: int = 1, concat: bool = True, **kwargs):
        heads = 1 if depth == 0 else heads
        concat = (depth > 0)
        if concat:
            out_channels = out_channels // heads
        super(GATv2ConvRaw, self).__init__(in_channels, out_channels, heads, concat, **kwargs)
        self.logger_a = LayerNumLogger()
        self.logger_w = LayerNumLogger()
        self.logger_in = LayerNumLogger()
        self.logger_msg = LayerNumLogger()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, **kwargs):
        self.logger_a.numel_after = edge_index.shape[1]
        self.logger_w.numel_after = self.lin_l.weight.numel()
        if not self.share_weights:
            self.logger_w.numel_after += self.lin_r.weight.numel()
        return super().forward(x, edge_index, edge_weight)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, edge_index = input
        f_in, f_h, f_c = x_in.shape[-1], module.heads, module.out_channels
        n, m = x_in.shape[0], edge_index.shape[1]

        # Linear
        flops_lin = f_in * f_h * f_c * n
        if not module.share_weights:
            flops_lin *= 2
        module.__flops__ += flops_lin
        # Message
        flops_attn  = 2 * f_c * m               # relu & alpha
        flops_attn += 2 * m                     # softmax & attention
        flops_attn *= f_h
        module.__flops__ += flops_attn
        # Bias and concat
        if (module.bias is not None) and module.concat:
            module.__flops__ += f_h * f_c * n
        elif (module.bias is not None) and not module.concat:
            module.__flops__ += (f_c + 1) * n
        else:
            module.__flops__ += n


class GATv2ConvThr(ConvThr, GATv2ConvRaw):
    def __init__(self, *args, thr_a, thr_w, **kwargs):
        super(GATv2ConvThr, self).__init__(*args, thr_a=thr_a, thr_w=thr_w, **kwargs)
        self.prune_lst = [self.lin_l, self.lin_r]

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
        if self.scheme_w in ['pruneall', 'pruneinc']:
            if self.scheme_w == 'pruneall':
                if prune.is_pruned(self.lin_l):
                    prune.remove(self.lin_l, 'weight')
                    prune.remove(self.lin_r, 'weight')
            norm_node_in = torch.norm(x, dim=0)
            norm_all_in = torch.norm(norm_node_in, dim=None)/x.shape[1]
            if norm_all_in > 1e-8:
                threshold_wi = self.threshold_w * norm_all_in / norm_node_in
                ThrInPrune.apply(self.lin_l, 'weight', threshold_wi)
                if not self.share_weights:
                    ThrInPrune.apply(self.lin_r, 'weight', threshold_wi)
            x_l = self.lin_l(x).view(-1, H, C)
            x_r = x_l if self.share_weights else self.lin_r(x).view(-1, H, C)
        elif self.scheme_w == 'keep':
            x_l = self.lin_l(x).view(-1, H, C)
            x_r = x_l if self.share_weights else self.lin_r(x).view(-1, H, C)
        elif self.scheme_w == 'full':
            raise NotImplementedError()
        self.logger_w.numel_before  = self.lin_l.weight.numel()
        self.logger_w.numel_after  = torch.sum(self.lin_l.weight != 0).item()
        if not self.share_weights:
            self.logger_w.numel_before += self.lin_r.weight.numel()
            self.logger_w.numel_after += torch.sum(self.lin_r.weight != 0).item()

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        self.idx_lock = self.get_idx_lock(edge_index, node_lock)
        # edge_updater_type: (x: PairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)
        # propagate_type: (x: PairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha, edge_attr=edge_attr, size=None)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias

        # Edge removal for next layer
        if self.scheme_a in ['pruneall', 'pruneinc', 'keep']:
            self.logger_a.numel_before = edge_index.shape[1]
            self.logger_a.numel_after = self.idx_keep.shape[0]
            # if verbose:
            #     print(f"  A: {self.logger_a}, W: {self.logger_w}")

            edge_index = edge_index[:, self.idx_keep]
            if edge_attr is not None:
                edge_attr = edge_attr[self.idx_keep]
        else:
            self.logger_a.numel_before = edge_index.shape[1]
            self.logger_a.numel_after = edge_index.shape[1]

        with torch.cuda.device(edge_index.device):
            torch.cuda.empty_cache()
        return out, edge_index

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
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
        self.logger_msg.numel_before = x.numel()
        self.logger_msg.numel_after = torch.sum(x != 0).item()

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)

        # Edge pruning
        if self.scheme_a in ['pruneall', 'pruneinc']:
            if self.scheme_a == 'pruneinc':
                raise NotImplementedError()
                self.idx_keep = self.idx_keep.to(alpha.device)
                mask_0 = torch.ones(alpha.shape[0], dtype=torch.bool, device=alpha.device)
                mask_0[self.idx_keep] = False
                alpha[mask_0] = 0
            else:
                mask_0 = torch.zeros(alpha.shape[0], dtype=torch.bool, device=alpha.device)

            norm_feat_msg = torch.norm(x_j, dim=[1,2])
            norm_all_msg = torch.norm(norm_feat_msg, dim=None, p=1)/x_j.shape[0]
            threshold_aj = self.threshold_a * norm_all_msg / norm_feat_msg
            mask_cmp = torch.norm(alpha, dim=1)/alpha.shape[1] < threshold_aj
            mask_0 = torch.logical_or(mask_0, mask_cmp)
            mask_0[self.idx_lock] = False
            alpha[mask_0] = 0
            self.idx_keep = torch.where(~mask_0)[0]
        elif self.scheme_a == 'keep':
            mask_0 = torch.ones(alpha.shape[0], dtype=torch.bool, device=alpha.device)
            mask_0[self.idx_keep] = False
            mask_0[self.idx_lock] = False
            alpha[mask_0] = 0

        # self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    @classmethod
    def cnt_flops(cls, module, input, output):
        x_in, _ = input
        x_out, edge_index = output
        f_in, f_h, f_c = x_in.shape[-1], module.heads, module.out_channels
        n, m = x_in.shape[0], edge_index.shape[1]

        # Linear
        flops_lin = f_in * f_h * f_c * n
        if not module.share_weights:
            flops_lin *= 2
        flops_lin = int(flops_lin * module.logger_w.ratio)
        module.__flops__ += flops_lin
        # Message
        flops_attn  = 2 * f_c * (m - n)          # relu & alpha
        flops_attn += 2 * (m - n)            # softmax & attention
        flops_attn *= f_h
        module.__flops__ += flops_attn
        # Bias and concat
        if (module.bias is not None) and module.concat:
            module.__flops__ += f_h * f_c * n
        elif (module.bias is not None) and not module.concat:
            module.__flops__ += (f_c + 1) * n
        else:
            module.__flops__ += n


class GINConvRaw(pyg_nn.GINConv):
    def __init__(self, in_channels: int, out_channels: int,
                 eps: float = 0., train_eps: bool = False, **kwargs):
        nn_default = pyg_nn.MLP(
            [in_channels, out_channels],
        )
        super(GINConvRaw, self).__init__(nn_default, eps, train_eps, **kwargs)


class GCNIIConvRaw(pyg_nn.GCN2Conv):
    pass


# ==========
layer_dict = {
    'gcn': GCNConvRaw,
    'gcn_thr': GCNConvThr,
    'gin': GINConvRaw,
    'gcn2': GCNIIConvRaw,
    'gat': GATv2ConvRaw,
    'gat_thr': GATv2ConvThr,
}

flops_modules_dict = {
    GCNConvRaw: GCNConvRaw.cnt_flops,
    GCNConvThr: GCNConvThr.cnt_flops,
    GATv2ConvRaw: GATv2ConvRaw.cnt_flops,
    GATv2ConvThr: GATv2ConvThr.cnt_flops,
}
