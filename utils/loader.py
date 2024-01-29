import os
import sys
import gc
import copy
from dotmap import DotMap
import numpy as np
# from sklearn.preprocessing import StandardScaler
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from .data_processor import DataProcess, DataProcess_inductive, matstd_clip
from precompute.prop import A2Prop

np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "% 9.3e" % x))
torch.set_printoptions(linewidth=160, edgeitems=5)


def dmap2dct(chnname: str, dmap: DotMap, processor: DataProcess):
    typedct = {'sgc': -2, 'gbp': -3,
               'sgc_agp': 0, 'gbp_agp': 1,
               'sgc_thr': 2, 'gbp_thr': 3,}
    ctype = chnname.split('_')[0]

    dct = {}
    dct['type'] = typedct[chnname]
    dct['hop'] = dmap.hop
    dct['dim'] = processor.nfeat
    dct['delta'] = dmap.delta if type(dmap.delta) is float else 1e-5
    dct['alpha'] = dmap.alpha if (type(dmap.alpha) is float and not (ctype == 'sgc')) else 0
    dct['rra'] = (1 - dmap.rrz) if type(dmap.rrz) is float else 0
    dct['rrb'] = dmap.rrz if type(dmap.rrz) is float else 0
    return dct

# ==========
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
    # dp.adj_matrix.setdiag(0)
    # dp.adj_matrix.eliminate_zeros()
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


def load_embedding(datastr: str, algo: str, algo_chn: DotMap,
                   datapath: str="./data/",
                   inductive: bool=False, multil: bool=False,
                   seed: int=0, **kwargs):
    # Inductive or transductive data processor
    dp = DataProcess(datastr, path=datapath, seed=seed)
    dp.input(['labels', 'attr_matrix', 'deg'])
    if inductive:
        dpi = DataProcess_inductive(datastr, path=datapath, seed=seed)
        dpi.input(['attr_matrix', 'deg'])
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
    # Get graph property
    n, m = dp.n, dp.m
    nfeat, nclass = dp.nfeat, dp.nclass
    if seed >= 15:
        print(dp)

    # Get node attributes
    py_a2prop = A2Prop()
    py_a2prop.load(os.path.join(datapath, datastr), m, n, seed)
    chn = dmap2dct(algo, DotMap(algo_chn), dp)

    feat = dp.attr_matrix.transpose().astype(np.float32, order='C')

    # deg_b = np.power(np.maximum(dp.deg, 1e-12), chn['rrb'])
    # idx_zero = np.where(deg_b == 0)[0]
    # assert idx_zero.size == 0, f"Isolated nodes found: {idx_zero}"
    # deg_b[idx_zero] = 1
    # feat /= deg_b
    macs_pre, time_pre = py_a2prop.compute(1, [chn], feat)
    # feat *= deg_b
    if not ('_'  in algo):
        if seed >= 15:
            print(f"[Pre] MACs Real: {macs_pre/1e9}, full: {chn['hop'] * m * nfeat/1e9}, ", end='')
            print(f"numA Real: {macs_pre/nfeat/1e3}, full: {chn['hop'] * m /1e3}")
        # macs_pre = chn['hop'] * m * nfeat

    feat = feat.transpose()
    feat = matstd_clip(feat, idx['train'], with_mean=True)
    feats = {'val':  torch.FloatTensor(feat[idx['val']]),
             'test': torch.FloatTensor(feat[idx['test']])}
    feats['train'] = torch.FloatTensor(feat[idx['train']])
    del feat
    gc.collect()
    # print(feats['train'].shape)
    return feats, labels, idx, nfeat, nclass, macs_pre/1e9, time_pre
