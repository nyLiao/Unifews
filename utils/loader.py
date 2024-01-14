import os
import sys
import gc
import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from .data_processor import DataProcess, DataProcess_inductive


def load_edgelist(datastr: str, datapath: str="./data/",
                   inductive: bool=False, multil: bool=False,
                   seed: int=0, **kwargs):
    # Inductive or transductive data processor
    dp = DataProcess(datastr, path=datapath, seed=seed)
    dp.input(['adjnpz', 'labels', 'attr_matrix'])
    if inductive:
        dpi = DataProcess_inductive(datastr, path=datapath, seed=seed)
        dpi.input(['adjnpz', 'attr_matrix'])
    else:
        dpi = dp
    # Get index
    if (datastr.startswith('cora') or datastr.startswith('citeseer') or datastr.startswith('pubmed')):
        dp.calculate(['idx_train'])
    else:
        dp.input(['idx_train', 'idx_val', 'idx_test'])
    idx = {'train': torch.LongTensor(dp.idx_train),
           'val':   torch.LongTensor(dp.idx_val),
           'test':  torch.LongTensor(dp.idx_test)}

    # Get label
    if multil:
        dp.calculate(['labels_oh'])
        dp.labels_oh[dp.labels_oh < 0] = 0
        labels = torch.LongTensor(dp.labels_oh).float()
    else:
        dp.labels[dp.labels < 0] = 0
        labels = torch.LongTensor(dp.labels.flatten())
    labels = {'train': labels[idx['train']],
              'val':   labels[idx['val']],
              'test':  labels[idx['test']]}

    # Get edge index
    dp.calculate(['edge_idx'])
    adj = {'test':  torch.from_numpy(dp.edge_idx).long()}
    if inductive:
        dpi.calculate(['edge_idx'])
        adj['train'] = torch.from_numpy(dpi.edge_idx).long()
    else:
        adj['train'] = adj['test']
    # Get node attributes
    feat = dp.attr_matrix
    feati = dpi.attr_matrix if inductive else feat
    # scaler = StandardScaler(with_mean=False)
    # scaler.fit(feati)
    # feat = scaler.transform(feat)
    # feati = scaler.transform(feati)
    feat = {'test': torch.FloatTensor(feat)}
    feat['train'] = torch.FloatTensor(feati)

    # Get graph property
    n, m = dp.n, dp.m
    nfeat, nclass = dp.nfeat, dp.nclass
    if seed >= 15:
        print(dp)
    return adj, feat, labels, idx, nfeat, nclass
