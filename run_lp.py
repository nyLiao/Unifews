import os
import random
import argparse
import numpy as np
import ptflops
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.datasets as datasets
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling, remove_self_loops, to_undirected

from utils.logger import Logger, ModelLogger, prepare_opt
import utils.metric as metric
from archs import identity_n_norm, flops_modules_dict
import archs.models as models


np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "% 9.3e" % x))
torch.set_printoptions(linewidth=160, edgeitems=5)

# ========== Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--seed', type=int, default=11, help='Random seed.')
parser.add_argument('-v', '--dev', type=int, default=0, help='Device id.')
parser.add_argument('-c', '--config', type=str, default='ogbl-collab', help='Config file name.')
parser.add_argument('-m', '--algo', type=str, default=None, help='Model name')
parser.add_argument('-n', '--suffix', type=str, default='', help='Save name suffix.')
parser.add_argument('-a', '--thr_a', type=float, default=None, help='Threshold of adj.')
parser.add_argument('-w', '--thr_w', type=float, default=None, help='Threshold of weight.')
parser.add_argument('-l', '--layer', type=int, default=None, help='Layer.')
args = prepare_opt(parser)

args.data = args.data + '-lp'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.dev >= 0:
    with torch.cuda.device(args.dev):
        torch.cuda.manual_seed(args.seed)

if not ('_'  in args.algo):
    args.thr_a, args.thr_w = 0.0, 0.0
flag_run = f"{args.seed}-{args.thr_a:.1e}-{args.thr_w:.1e}"
logger = Logger(args.data, args.algo, flag_run=flag_run)
logger.save_opt(args)
model_logger = ModelLogger(logger,
                patience=args.patience,
                cmp='max',
                prefix='model'+args.suffix,
                storage='state_ram' if args.data in ['ogbl-collab'] else 'state_gpu')
stopwatch = metric.Stopwatch()

# ========== Load
if args.data[:-3] in ['ogbl-collab']:
    from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
    ds = PygLinkPropPredDataset(root='./data/pl', name=args.data[:-3])
    evaluator = Evaluator(name=args.data[:-3])
    split_edge = ds.get_edge_split()
    data_train = ds[0]

    adj, adj_li, adj_weight, feat, labels = {}, {}, {}, {}, {}
    edge_index = split_edge['train']['edge'].T
    lb = torch.ones(edge_index.size(1), dtype=torch.float)
    adj_li['train'], labels['train'] = remove_self_loops(edge_index, lb)
    adj['train'], adj_weight['train'] = data_train.edge_index, data_train.edge_weight.flatten().float()
    for sp, spd in zip(['val', 'test'], ['valid', 'test']):
        data = split_edge[spd]
        edge_index = data['edge'].T
        lb = torch.ones(edge_index.size(1), dtype=torch.float)
        edge_index, lb = remove_self_loops(edge_index, lb)
        adj_li[sp] = torch.cat([edge_index, data['edge_neg'].T], dim=-1)
        labels[sp] = torch.cat([lb, torch.zeros(data['edge_neg'].size(0))], dim=0)
        edge_index, lb = to_undirected(edge_index, data['weight'].flatten().float())
        pre = {'val': 'train', 'test': 'val'}
        adj[sp] = torch.cat([adj[pre[sp]], edge_index], dim=-1)
        adj_weight[sp] = torch.cat([adj_weight[pre[sp]], lb], dim=0)
    feat['train'] = feat['test'] = data_train.x
    nfeat, nclass = ds.num_features, 1
else:
    ds = datasets.Planetoid(root='./data/pl', name=args.data[:-3],
                            transform=T.Compose([
                                # AddRemainingSelfLoops(fill_value=1.),
                                T.RandomLinkSplit(
                                    num_val=0.05, num_test=0.1, is_undirected=True,
                                    add_negative_train_samples=False)
                            ]),)
    adj, adj_li, adj_weight, feat, labels = {}, {}, {}, {}, {}
    for sp, data in zip(['train', 'val', 'test'], ds[0]):
        edge_index = remove_self_loops(data.edge_index)[0]
        adj[sp], labels[sp] = edge_index, data.edge_label
        adj_li[sp] = data.edge_label_index
        adj_weight[sp] = None
    feat['train'] = feat['test'] = data.x
    evaluator = None
    nfeat, nclass = ds.num_features, 1

model = models.GNNLPThr(nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
                        thr_a=args.thr_a, thr_w=args.thr_w, dropout=args.dropout, layer=args.algo)
model.reset_parameters()
diag = 1.0
adj['train'] = identity_n_norm(adj['train'], edge_weight=adj_weight['train'], num_nodes=feat['train'].shape[0],
                    rnorm=model.kwargs['rnorm'], diag=diag)
adj['val'] = identity_n_norm(adj['val'], edge_weight=adj_weight['val'], num_nodes=feat['train'].shape[0],
                    rnorm=model.kwargs['rnorm'], diag=diag)
if logger.lvl_config > 1:
    print(type(model).__name__, args.algo, args.thr_a, args.thr_w)
if logger.lvl_config > 2:
    print(model)
model_logger.register(model, save_init=False)
if args.dev >= 0:
    model = model.cuda(args.dev)

# ========== Train helper
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=1e-4, patience=15, verbose=False)
loss_fn = nn.BCEWithLogitsLoss()


def train(x, edge_idx, edge_laebl_idx, y, epoch, verbose=False):
    model.train()
    if epoch < args.epochs//2:
        model.set_scheme('pruneall', 'pruneall')
    else:
        # model.set_scheme('pruneinc', 'pruneinc')
        model.set_scheme('pruneall', 'pruneinc')
    x, y = x.cuda(args.dev), y.cuda(args.dev)
    if isinstance(edge_idx, tuple):
        edge_idx = (edge_idx[0].cuda(args.dev), edge_idx[1].cuda(args.dev))
    else:
        edge_idx = edge_idx.cuda(args.dev)
    edge_laebl_idx = edge_laebl_idx.cuda(args.dev)
    stopwatch.reset()

    stopwatch.start()
    optimizer.zero_grad()
    z = model(x, edge_idx, node_lock=torch.Tensor([]), verbose=verbose)
    stopwatch.pause()

    if isinstance(edge_idx, tuple):
        edge_idx = edge_idx[0]
    neg_edge_index = negative_sampling(
        edge_index=edge_idx, num_nodes=x.size(0),
        num_neg_samples=edge_laebl_idx.size(1), method='sparse')
    edge_label_index = torch.cat([edge_laebl_idx, neg_edge_index], dim=-1,)
    edge_label = torch.cat([y, y.new_zeros(neg_edge_index.size(1))], dim=0)

    stopwatch.start()
    z_i, z_j = z[edge_label_index[0]], z[edge_label_index[1]]
    output = model.decode(z_i, z_j).view(-1)
    loss = loss_fn(output, edge_label)
    loss.backward()
    optimizer.step()
    stopwatch.pause()

    return loss.item(), stopwatch.time


def eval(x, edge_idx, edge_laebl_idx, y, verbose=False):
    model.eval()
    model.set_scheme('keep', 'keep')
    # model.set_scheme('full', 'keep')
    x, y = x.cuda(args.dev), y.cuda(args.dev)
    if isinstance(edge_idx, tuple):
        edge_idx = (edge_idx[0].cuda(args.dev), edge_idx[1].cuda(args.dev))
    else:
        edge_idx = edge_idx.cuda(args.dev)
    stopwatch.reset()

    with torch.no_grad():
        stopwatch.start()
        z = model(x, edge_idx, node_lock=torch.Tensor([]), verbose=verbose)
        z_i, z_j = z[edge_laebl_idx[0]], z[edge_laebl_idx[1]]
        output = model.decode(z_i, z_j).view(-1).sigmoid()
        stopwatch.pause()

    if evaluator is None:
        res = roc_auc_score(y.cpu().numpy(), output.cpu().numpy())
    else:
        pos_idx, neg_idx = y.nonzero().view(-1), (1-y).nonzero().view(-1)
        res = evaluator.eval({
            'y_pred_pos': output[pos_idx],
            'y_pred_neg': output[neg_idx],
        })['hits@50']
    return res, stopwatch.time, None, None


def cal_flops(x, edge_idx, verbose=False):
    return 0
    model.eval()
    model.set_scheme('keep', 'keep')
    x = x.cuda(args.dev)
    if isinstance(edge_idx, tuple):
        edge_idx = (edge_idx[0].cuda(args.dev), edge_idx[1].cuda(args.dev))
    else:
        edge_idx = edge_idx.cuda(args.dev)

    handle = model.register_forward_hook(models.GNNThr.batch_counter_hook)
    model.__batch_counter_handle__ = handle
    macs, nparam = ptflops.get_model_complexity_info(model, (1,1,1),
                        input_constructor=lambda _: {'x': x, 'edge_idx': edge_idx},
                        custom_modules_hooks=flops_modules_dict,
                        as_strings=False, print_per_layer_stat=verbose, verbose=verbose)
    return macs/1e9


# ========== Train
# print('-' * 20, flush=True)
with torch.cuda.device(args.dev):
    torch.cuda.empty_cache()
time_tol, macs_tol = metric.Accumulator(), metric.Accumulator()
epoch_conv, acc_best = 0, 0

for epoch in range(1, args.epochs+1):
    verbose = epoch % 1 == 0 and (logger.lvl_log > 0)
    loss_train, time_epoch = train(x=feat['train'], edge_idx=adj['train'],
                                   edge_laebl_idx=adj_li['train'], y=labels['train'],
                                   epoch=epoch, verbose=verbose)
    time_tol.update(time_epoch)
    acc_val, _, _, _ = eval(x=feat['train'], edge_idx=adj['val'],
                            edge_laebl_idx=adj_li['val'], y=labels['val'],)
    scheduler.step(acc_val)
    macs_epoch = cal_flops(x=feat['train'], edge_idx=adj['train'],)
    macs_tol.update(macs_epoch)

    if verbose:
        res = f"Epoch:{epoch:04d} | train loss:{loss_train:.4f}, val acc:{acc_val:.4f}, time:{time_tol.val:.4f}, macs:{macs_tol.val:.4f}"
        if logger.lvl_log > 1:
            logger.print(res)

    # Log convergence
    acc_best = model_logger.save_best(acc_val, epoch=epoch)
    if model_logger.is_early_stop(epoch=epoch):
        # pass
        break     # Enable to early stop
    else:
        epoch_conv = max(0, epoch - model_logger.patience)

# ========== Test
# print('-' * 20, flush=True)
for sp in ['train', 'val']:
    del adj[sp], adj_li[sp], adj_weight[sp], labels[sp]
model = model_logger.load('best')
if args.dev >= 0:
    model = model.cuda(args.dev)
with torch.cuda.device(args.dev):
    torch.cuda.empty_cache()

adj['test'] = identity_n_norm(adj['test'], edge_weight=None, num_nodes=feat['test'].shape[0],
                    rnorm=model.kwargs['rnorm'], diag=model.kwargs['diag'])
acc_test, time_test, outl, labl = eval(x=feat['test'], edge_idx=adj['test'],
                                       edge_laebl_idx=adj_li['test'], y=labels['test'],)
# mem_ram, mem_cuda = metric.get_ram(), metric.get_cuda_mem(args.dev)
# num_param, mem_param = metric.get_num_params(model), metric.get_mem_params(model)
macs_test = cal_flops(x=feat['test'], edge_idx=adj['test'],)
numel_a, numel_w = model.get_numel()

# ========== Log
if logger.lvl_config > 0:
    print(f"[Val] best acc: {acc_best:0.5f} (epoch: {epoch_conv}/{epoch}), [Test] best acc: {acc_test:0.5f}", flush=True)
if logger.lvl_config > 1:
    print(f"[Train] time: {time_tol.val:0.4f} s (avg: {time_tol.avg*1000:0.1f} ms), MACs: {macs_tol.val:0.3f} G (avg: {macs_tol.avg:0.1f} G)")
    print(f"[Test]  time: {time_test:0.4f} s, MACs: {macs_test:0.4f} G, Num adj: {numel_a:0.3f} k, Num weight: {numel_w:0.3f} k")
    # print(f"RAM: {mem_ram:.3f} GB, CUDA: {mem_cuda:.3f} GB, Num params: {num_param:0.4f} M, Mem params: {mem_param:0.4f} MB")
if logger.lvl_config > 2:
    logger_tab = Logger(args.data, args.algo, flag_run=flag_run, dir=('./save', args.data))
    logger_tab.file_log = logger_tab.path_join('log.csv')
    hstr, cstr = logger_tab.str_csv(data=args.data, algo=args.algo, seed=args.seed, thr_a=args.thr_a, thr_w=args.thr_w,
                                    acc_test=acc_test, conv_epoch=epoch_conv, epoch=epoch,
                                    time_train=time_tol.val, macs_train=macs_tol.val,
                                    time_test=time_test, macs_test=macs_test, numel_a=numel_a, numel_w=numel_w)
    logger_tab.print_header(hstr, cstr)
