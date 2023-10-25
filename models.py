import torch.nn as nn
import torch.nn.functional as F

from layers import layer_dict


kwargs_default = {
    'gcn': {
        'cached': False,
        'add_self_loops': True,
        'improved': False,
        'normalize': True,
    },
    'gin': {
        'eps': 0.0,
        'train_eps': False,
    },
    'gat': {
        'heads': 8,
        'concat': True,
    },
}

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

        # Select conv layer
        Conv = layer_dict[layer]
        # Set layer args
        for k, v in kwargs_default[layer.replace('_thr', '')].items():
            kwargs.setdefault(k, v)

        self.convs = nn.ModuleList()
        self.convs.append(Conv(nfeat, nhidden, depth=nlayer, **kwargs))
        self.norms = nn.ModuleList()
        self.norms.append(nn.BatchNorm1d(nhidden))

        for layer in range(1, nlayer - 1):
            self.convs.append(Conv(nhidden, nhidden, depth=nlayer-layer, **kwargs))
            self.norms.append(nn.BatchNorm1d(nhidden))
        self.convs.append(Conv(nhidden, nclass, depth=0, **kwargs))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_idx, node_lock=[]):
        if self.apply_thr:
            # Layer inheritence of edge_idx
            for i, conv in enumerate(self.convs[:-1]):
                x, edge_idx = conv(x, edge_idx, node_lock=node_lock)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x, _ = self.convs[-1](x, edge_idx, node_lock=node_lock)
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
