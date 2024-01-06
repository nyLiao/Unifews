# -*- coding:utf-8 -*-
""" Convert OGB data
Author: nyLiao
File Created: 2021-12-01
"""
import os
import numpy as np
from ogb.nodeproppred import NodePropPredDataset

from data_processor import *


class DataProcess_OGB(DataProcess):
    def __init__(self, name, path='../data/', rrz=0.5, seed=0) -> None:
        super().__init__(name, path=path, rrz=rrz, seed=seed)

    @property
    def n_train(self):
        if self.idx_train is None:
            self.input(['labels'])
        return len(self.idx_train)

    def fetch(self):
        dataset = NodePropPredDataset(name=self.name, root=self.path)
        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0]

        row, col = graph['edge_index'][0], graph['edge_index'][1]
        row, col = np.concatenate([row, col], axis=0), np.concatenate([col, row], axis=0)
        deg = np.bincount(row)
        idx_zero = np.where(deg == 0)[0]
        if len(idx_zero) > 0:
            print(f"Warning: removing {len(idx_zero)} isolated nodes: {idx_zero}!")

        # remove isolated nodes
        nodelst = deg.nonzero()[0]
        idxnew = np.full(graph['num_nodes'], -1)
        idxnew[nodelst] = np.arange(len(nodelst))
        self._n = len(nodelst)
        row, col = idxnew[row], idxnew[col]
        self.adj_matrix = edgeidx2adj(row, col, self.n)
        self._m = self.adj_matrix.nnz

        self.attr_matrix = graph['node_feat'][nodelst]
        assert (labels.ndim==2 and labels.shape[1]==1) or labels.ndim==1, "label shape error"
        self.labels = labels.flatten()[nodelst]

        self.idx_train = idxnew[idx_train]
        self.idx_train = self.idx_train[self.idx_train > -1]
        self.idx_val = idxnew[idx_val]
        self.idx_val = self.idx_val[self.idx_val > -1]
        self.idx_test = idxnew[idx_test]
        self.idx_test = self.idx_test[self.idx_test > -1]


# ====================
if __name__ == '__main__':
    ds = DataProcess_OGB('ogbn-arxiv', path='/share/data/dataset/OGB')
    ds.fetch()
    ds.calculate(['deg'])
    print(ds)
    ds.output(['adjtxt', 'adjnpz', 'adjl', 'attr_matrix', 'deg', 'labels', ])
