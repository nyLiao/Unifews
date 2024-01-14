import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from .layers import layer_dict


kwargs_default = {
    'gcn': {
        'cached': False,
        'add_self_loops': False,
        'improved': False,
        'normalize': False,
        'rnorm': 0.5,
        'diag': 1.,
    },
    'gin': {
        'eps': 0.0,
        'train_eps': False,
        'rnorm': None,
        'diag': 1.,
    },
    'gat': {
        'heads': 8,
        'concat': True,
        'share_weights': False,
        'add_self_loops': False,
        'rnorm': None,
        'diag': 1.,
    },
}


def state2module(model, param_name):
    parts = param_name.split('.')
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    return module


def set_attr(module, k, v):
    if hasattr(module, k):
        setattr(module, k, v)


class GNNThr(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass,
                 thr_a=0.0, thr_w=0.0, dropout: float = 0.0, layer: str = 'gcn',
                 **kwargs,):
        super(GNNThr, self).__init__()
        self.apply_thr = (layer.endswith('_thr'))
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()
        self.use_bn = True
        self.kwargs = kwargs

        # Select conv layer
        Conv = layer_dict[layer]
        # Set layer args
        for k, v in kwargs_default[layer.replace('_thr', '')].items():
            self.kwargs.setdefault(k, v)
        norm_kwargs = {'affine': True, 'track_running_stats': True, 'momentum': 0.9}
        thr_a = [thr_a] * nlayer if isinstance(thr_a, float) else thr_a
        thr_w = [thr_w] * nlayer if isinstance(thr_w, float) else thr_w

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(Conv(nfeat, nhidden,
                               depth=nlayer, thr_a=thr_a[0], thr_w=thr_w[0],
                               **self.kwargs))
        self.norms.append(nn.BatchNorm1d(nhidden, **norm_kwargs))
        for layer in range(1, nlayer - 1):
            self.convs.append(Conv(nhidden, nhidden,
                                   depth=nlayer-layer, thr_a=thr_a[layer], thr_w=thr_w[layer],
                                   **self.kwargs))
            self.norms.append(nn.BatchNorm1d(nhidden, **norm_kwargs))
        self.convs.append(Conv(nhidden, nclass,
                               depth=0, thr_a=thr_a[nlayer-1], thr_w=thr_w[nlayer-1],
                               **self.kwargs))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_idx, node_lock=torch.Tensor([]), verbose=False):
        if self.apply_thr:
            # Layer inheritence of edge_idx
            for i, conv in enumerate(self.convs[:-1]):
                x, edge_idx = conv(x, edge_idx, node_lock=node_lock, verbose=verbose)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x, _ = self.convs[-1](x, edge_idx, node_lock=node_lock, verbose=verbose)
            return x
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_idx)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x = self.convs[-1](x, edge_idx)
            return x

    def set_scheme(self, scheme_a, scheme_w):
        self.apply(lambda module: set_attr(module, 'scheme_a', scheme_a))
        self.apply(lambda module: set_attr(module, 'scheme_w', scheme_w))

    def remove(self):
        for conv in self.convs:
            if hasattr(conv, 'prune_lst'):
                for module in conv.prune_lst:
                    if prune.is_pruned(module):
                        prune.remove(module, 'weight')

    def get_numel(self):
        numel_a = sum([conv.logger_a.numel_after for conv in self.convs])
        numel_w = sum([conv.logger_w.numel_after for conv in self.convs])
        return numel_a/1e3, numel_w/1e3

    @classmethod
    def batch_counter_hook(cls, module, input, output):
        module.__batch_counter__ += 1
