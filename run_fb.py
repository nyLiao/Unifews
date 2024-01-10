import os
import random
import argparse
import numpy as np
import ptflops

import torch
import torch.nn as nn
import torch.optim as optim

from utils.logger import Logger, ModelLogger, prepare_opt
from utils.loader import load_edgelist
import utils.metric as metric
from archs import identity_n_norm, flops_modules_dict
import archs.models as models


np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "% 9.3e" % x))
torch.set_printoptions(linewidth=160, edgeitems=5)

# ========== Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--seed', type=int, default=11, help='Random seed.')
parser.add_argument('-c', '--config', type=str, default='./config/cora.json', help='Config file path.')
parser.add_argument('-v', '--dev', type=int, default=1, help='Device id.')
parser.add_argument('-n', '--suffix', type=str, default='', help='Save name suffix')
args = prepare_opt(parser)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.dev >= 0:
    with torch.cuda.device(args.dev):
        torch.cuda.manual_seed(args.seed)

flag_run = str(args.seed)
logger = Logger(args.data, args.algo, flag_run=flag_run)
if args.seed > 20:
    print(args.toDict())
    logger.save_opt(args)
model_logger = ModelLogger(logger, patience=args.patience, cmp='max',
                           prefix='model'+args.suffix, storage='state_gpu')
stopwatch = metric.Stopwatch()

# ========== Load
adj, feat, labels, idx, nfeat, nclass = load_edgelist(datastr=args.data, datapath=args.path,
                inductive=args.inductive, multil=args.multil, seed=args.seed)

model = models.GNNThr(nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
                    dropout=args.dropout, layer=args.algo)
model.reset_parameters()
adj['train'] = identity_n_norm(adj['train'], edge_weight=None, num_nodes=feat['train'].shape[0],
                    rnorm=model.kwargs['rnorm'], diag=model.kwargs['diag'])
if args.seed > 15:
    print(type(model).__name__)
if args.seed > 20:
    print(model)
model_logger.register(model, save_init=False)
if args.dev >= 0:
    model = model.cuda(args.dev)

# ========== Train helper
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=1e-4, patience=30, verbose=False)
loss_fn = nn.BCEWithLogitsLoss() if args.multil else nn.CrossEntropyLoss()


def train(x, edge_idx, y, idx_split, verbose=False):
    model.train()
    x, y = x.cuda(args.dev), y.cuda(args.dev)
    if isinstance(edge_idx, tuple):
        edge_idx = (edge_idx[0].cuda(args.dev), edge_idx[1].cuda(args.dev))
    else:
        edge_idx = edge_idx.cuda(args.dev)
    stopwatch.reset()

    stopwatch.start()
    optimizer.zero_grad()
    output = model(x, edge_idx, node_lock=torch.Tensor([]), verbose=verbose)[idx_split]
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    stopwatch.pause()

    return loss.item(), stopwatch.time


def eval(x, edge_idx, y, idx_split, verbose=False):
    model.eval()
    x, y = x.cuda(args.dev), y.cuda(args.dev)
    if isinstance(edge_idx, tuple):
        edge_idx = (edge_idx[0].cuda(args.dev), edge_idx[1].cuda(args.dev))
    else:
        edge_idx = edge_idx.cuda(args.dev)
    calc = metric.F1Calculator(nclass)
    stopwatch.reset()

    with torch.no_grad():
        stopwatch.start()
        output = model(x, edge_idx, node_lock=idx_split, verbose=verbose)[idx_split]
        stopwatch.pause()

        output = output.cpu().detach()
        ylabel = y.cpu().detach()
        if args.multil:
            output = torch.where(output > 0, torch.tensor(1, device=output.device), torch.tensor(0, device=output.device))
        else:
            output = output.argmax(dim=1)
        calc.update(ylabel, output)

        output = output.numpy()
        ylabel = ylabel.numpy()

    res = calc.compute(('macro' if args.multil else 'micro'))
    return res, stopwatch.time, output, y


def get_flops(x, edge_idx, idx_split, verbose=False):
    model.eval()
    model.apply(models.cnting_flops)
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
    model.apply(models.cnted_flops)
    return macs/1e9


# ========== Train
# print('-' * 20, flush=True)
# print('Start training...')
with torch.cuda.device(args.dev):
    torch.cuda.empty_cache()
time_tol, macs_tol = metric.Accumulator(), metric.Accumulator()
epoch_conv, acc_best = 0, 0

for epoch in range(1, args.epochs+1):
    verbose = epoch % 1 == 0 and (args.seed >= 10)
    loss_train, time_epoch = train(x=feat['train'], edge_idx=adj['train'],
                                   y=labels['train'], idx_split=idx['train'],
                                   verbose=verbose)
    time_tol.update(time_epoch)
    acc_val, _, _, _ = eval(x=feat['train'], edge_idx=adj['train'],
                            y=labels['val'], idx_split=idx['val'])
    scheduler.step(acc_val)
    macs_epoch = get_flops(x=feat['train'], edge_idx=adj['train'], idx_split=idx['train'])
    macs_tol.update(macs_epoch)

    if verbose:
        res = f"Epoch:{epoch:04d} | train loss:{loss_train:.4f}, val acc:{acc_val:.4f}, time:{time_tol.val:.4f}, macs:{macs_tol.val:.4f}"
        if args.seed > 20:
            logger.print(res)
        else:
            print(res)
    # Early stop if converge
    acc_best = model_logger.save_best(acc_val, epoch=epoch)
    epoch_conv = epoch
    if model_logger.is_early_stop(epoch=epoch):
        break

# ========== Test
# print('-' * 20, flush=True)
model = model_logger.load()
if args.dev >= 0:
    model = model.cuda(args.dev)
with torch.cuda.device(args.dev):
    torch.cuda.empty_cache()

adj['test'] = identity_n_norm(adj['test'], edge_weight=None, num_nodes=feat['test'].shape[0],
                    rnorm=model.kwargs['rnorm'], diag=model.kwargs['diag'])
acc_test, time_test, outl, labl = eval(x=feat['test'], edge_idx=adj['test'],
                                       y=labels['test'], idx_split=idx['test'])
mem_ram, mem_cuda = metric.get_ram(), metric.get_cuda_mem(args.dev)
num_param, mem_param = metric.get_num_params(model), metric.get_mem_params(model)
macs_test = get_flops(x=feat['test'], edge_idx=adj['test'], idx_split=idx['test'])

# ========== Log
if args.seed >= 5:
    print(f"[Val] best acc: {acc_best:0.4f} (epoch: {epoch_conv}/{epoch}), [Test] best acc: {acc_test:0.4f}", flush=True)
if args.seed >= 15:
    print(f"[Train] time: {time_tol.val:0.4f} s (avg: {time_tol.avg*1000:0.1f} ms), MACs: {macs_tol.val:0.4f} G (avg: {macs_tol.avg:0.1f} G)")
    print(f"[Test]  time: {time_test:0.4f} s, MACs: {macs_test:0.4f} G, RAM: {mem_ram:.3f} GB, CUDA: {mem_cuda:.3f} GB")
    print(f"Num params: {num_param:0.4f} M, Mem params: {mem_param:0.4f} MB")
if args.seed >= 25:
    logger_tab = Logger(args.data, args.algo, flag_run=flag_run, dir=('./save', args.data))
    logger_tab.file_log = logger_tab.path_join('log.csv')
    hstr, cstr = logger_tab.str_csv(data=args.data, algo=args.algo, seed=flag_run,
                                    acc_test=acc_test, conv_epoch=epoch_conv, epoch=epoch,
                                    time_train=time_tol.val, macs_train=macs_tol.val,
                                    time_test=time_test, macs_test=macs_test, mem_ram=mem_ram, mem_cuda=mem_cuda,
                                    num_param=num_param, mem_param=mem_param)
    logger_tab.print_header(hstr, cstr)
