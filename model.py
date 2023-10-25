import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv.gcn_conv import *
import torch.nn.utils.prune as prune


from utils.logger import ThrLayerLogger


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


class ThrInPruningMethod(prune.BasePruningMethod):
    """Prune by input-dimension thresholding.
    """
    PRUNING_TYPE = 'unstructured'
    def __init__(self, threshold):
        """Args:
            threshold (Tensor [F_in]): threshold for each input channel
        """
        self.threshold = threshold

    def compute_mask(self, t, default_mask):
        """Args:
            t (Tensor [F_out, F_in]): tensor to prune
        """
        assert self.threshold.shape == t.shape[1:]
        mask = default_mask.clone()
        mask[t.abs() < self.threshold] = 0
        return mask

    @classmethod
    def apply(cls, module, name, threshold):
        return super().apply(module, name, threshold=threshold)


class GCNConvThr(pyg_nn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(GCNConvThr, self).__init__(*args, **kwargs)
        self.threshold_a = 4e-3
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
            threshold_wi = self.threshold_w / (torch.norm(x, dim=0)/x.shape[0])
            ThrInPruningMethod.apply(self.lin, 'weight', threshold_wi)
            x = self.lin(x)

            self.logger_w.nele_before = self.lin.weight.numel()
            self.logger_w.nele_after = torch.sum(self.lin.weight != 0).item()
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
            self.logger_a.nele_before = edge_index.shape[1]
            self.logger_a.nele_after = self.idx_keep.shape[0]

            edge_index = edge_index[:, self.idx_keep]
            if edge_weight is not None:
                edge_weight = edge_weight[self.idx_keep]

            print(f"  A: {self.logger_a}, W: {self.logger_w}")

        return out, edge_index


class GCNThr(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass,
                 dropout=0.5, apply_thr=True):
        super(GCNThr, self).__init__()

        cached = False
        add_self_loops = True
        improved = False
        self.apply_thr = apply_thr
        Conv = GCNConvThr if apply_thr else GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(
            Conv(nfeat, nhidden, cached=cached, normalize=True,
                 add_self_loops=add_self_loops, improved=improved))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhidden))
        for _ in range(nlayer - 2):
            self.convs.append(
                Conv(nhidden, nhidden, cached=cached, normalize=True,
                     add_self_loops=add_self_loops, improved=improved))
            self.bns.append(nn.BatchNorm1d(nhidden))

        self.convs.append(
            Conv(nhidden, nclass, cached=cached, normalize=True,
                 add_self_loops=add_self_loops, improved=improved))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = True

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_idx, node_lock=[]):
        if self.apply_thr:
            # Layer inheritence of edge_idx
            for i, conv in enumerate(self.convs[:-1]):
                x, edge_idx = conv(x, edge_idx, node_lock=node_lock)
                if self.use_bn:
                    x = self.bns[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x, _ = self.convs[-1](x, edge_idx, node_lock=node_lock)
            return x
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_idx)
                if self.use_bn:
                    x = self.bns[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_idx)
            return x
