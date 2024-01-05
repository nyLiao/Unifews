import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from layers import layer_dict


kwargs_default = {
    'gcn': {
        'cached': False,
        'add_self_loops': True,
        'improved': False,
        'normalize': True,
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
        'add_self_loops': True,
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


class GNNThr(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass,
                 dropout: float = 0.0,
                 layer: str = 'gcn',
                 **kwargs,):
        super(GNNThr, self).__init__()
        self.apply_thr = (layer.endswith('_thr'))
        self.dropout = nn.Dropout(p=dropout)
        self.act = F.relu
        self.use_bn = True
        self.kwargs = kwargs

        # Select conv layer
        Conv = layer_dict[layer]
        # Set layer args
        for k, v in kwargs_default[layer.replace('_thr', '')].items():
            self.kwargs.setdefault(k, v)
        norm_kwargs = {'affine': True, 'track_running_stats': True, 'momentum': 0.9}

        self.convs = nn.ModuleList()
        self.convs.append(Conv(nfeat, nhidden, depth=nlayer, **self.kwargs))
        self.norms = nn.ModuleList()
        self.norms.append(nn.BatchNorm1d(nhidden, **norm_kwargs))

        for layer in range(1, nlayer - 1):
            self.convs.append(Conv(nhidden, nhidden, depth=nlayer-layer, **self.kwargs))
            self.norms.append(nn.BatchNorm1d(nhidden, **norm_kwargs))
        self.convs.append(Conv(nhidden, nclass, depth=0, **self.kwargs))

    def remove(self):
        for conv in self.convs:
            if hasattr(conv, 'prune_lst'):
                for module in conv.prune_lst:
                    if prune.is_pruned(module):
                        prune.remove(module, 'weight')

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
