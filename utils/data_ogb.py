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

        self._n = graph['num_nodes']
        row, col = graph['edge_index'][0], graph['edge_index'][1]
        row, col = np.concatenate([row, col], axis=0), np.concatenate([col, row], axis=0)
        ones = np.ones(len(row), dtype=np.int8)
        self.adj_matrix = sp.coo_matrix(
            (ones, (row, col)),
            shape=(self.n, self.n))
        self.adj_matrix = self.adj_matrix.tocsr()
        self.adj_matrix.setdiag(0)
        self.adj_matrix.eliminate_zeros()
        self.adj_matrix.data = np.ones(len(self.adj_matrix.data), dtype=np.int8)
        self._m = self.adj_matrix.nnz

        self.calculate(['deg'])
        idx_zero = np.where(self.deg == 0)[0]
        if len(idx_zero) > 0:
            print(f"Warning: {len(idx_zero)} isolated nodes found: {idx_zero}!")

        self.attr_matrix = graph['node_feat']
        assert (labels.ndim==2 and labels.shape[1]==1) or labels.ndim==1, "label shape error"
        self.labels = labels.flatten()

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test


# ====================
if __name__ == '__main__':
    ds = DataProcess_OGB('ogbn-arxiv', path='/share/data/dataset/OGB')
    ds.fetch()
    ds.calculate(['deg'])
    print(ds)
    ds.output(['adjtxt', 'adjnpz', 'adjl', 'attr_matrix', 'deg', 'labels', ])
