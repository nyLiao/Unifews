import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils.logger import Logger, ModelLogger, prepare_opt
from utils.loader import load_edgelist
import utils.metric as metric
import model


np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "% 9.3e" % x))
torch.set_printoptions(linewidth=160, edgeitems=5)

# ========== Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--seed', type=int, default=7, help='Random seed.')
parser.add_argument('-c', '--config', type=str, default='./config/cora.json', help='Config file path.')
parser.add_argument('-v', '--dev', type=int, default=0, help='Device id.')
parser.add_argument('-n', '--suffix', type=str, default='', help='Save name suffix')
args = prepare_opt(parser)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.dev >= 0:
    torch.cuda.manual_seed(args.seed)

flag_run = str(args.seed)
logger = Logger(args.data, args.algo, flag_run=flag_run)
if args.seed > 7:
    print(args.toDict())
    logger.save_opt(args)
else:
    print(args.chn.toDict())
model_logger = ModelLogger(logger, patience=args.patience, cmp='max',
                           prefix='model'+args.suffix, storage='model_gpu')
stopwatch = metric.Stopwatch()

# ========== Load
adj, feat, labels, idx, nfeat, nclass = load_edgelist(datastr=args.data, datapath=args.path,
                inductive=args.inductive, multil=args.multil, seed=args.seed)

model = model.GCNThr(nlayer=args.layer, nfeat=nfeat, nhidden=args.hidden, nclass=nclass,
                     dropout=args.dropout, apply_thr=True)
# if args.seed == 7:
#     print(type(model).__name__)
if args.seed > 7:
    print(model)
model_logger.register(model, save_init=False)
if args.dev >= 0:
    model = model.cuda(args.dev)

# ========== Train helper
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=1e-4, patience=30, verbose=False)
loss_fn = nn.BCEWithLogitsLoss() if args.multil else nn.CrossEntropyLoss()


def train(x, edge_idx, y, idx_split):
    model.train()
    x, edge_idx, y = x.cuda(args.dev), edge_idx.cuda(args.dev), y.cuda(args.dev)
    stopwatch.reset()

    stopwatch.start()
    optimizer.zero_grad()
    output = model(x, edge_idx, node_lock=[])[idx_split]
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    stopwatch.pause()

    return loss.item(), stopwatch.time


def eval(x, edge_idx, y, idx_split):
    model.eval()
    x, edge_idx, y = x.cuda(args.dev), edge_idx.cuda(args.dev), y.cuda(args.dev)
    calc = metric.F1Calculator(nclass)
    stopwatch.reset()
    # n = feat['train'].shape[0]
    # node_lock = torch.LongTensor(np.arange(n))

    with torch.no_grad():
        stopwatch.start()
        output = model(x, edge_idx, node_lock=idx_split)[idx_split]
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


# ========== Train
# print('-' * 20, flush=True)
# print('Start training...')
torch.cuda.empty_cache()
time_train = 0
conv_epoch, acc_best = 0, 0

for epoch in range(args.epochs):
    loss_train, time_epoch = train(x=feat['train'], edge_idx=adj['train'],
                                   y=labels['train'], idx_split=idx['train'])
    time_train += time_epoch
    acc_val, _, _, _ = eval(x=feat['train'], edge_idx=adj['train'],
                            y=labels['val'], idx_split=idx['val'])
    # acc_val = epoch
    scheduler.step(acc_val)

    if (epoch+1) % 1 == 0 and (args.seed >= 7):
        res = f"Epoch:{epoch:04d} | train loss:{loss_train:.4f}, val acc:{acc_val:.4f}, cost:{time_train:.4f}"
        print(res)
        # logger.print(res)
    # Early stop if converge
    acc_best = model_logger.save_best(acc_val, epoch=epoch)
    conv_epoch = epoch + 1
    if model_logger.is_early_stop(epoch=epoch):
        break

# ========== Test
# print('-' * 20, flush=True)
model = model_logger.load()
if args.dev >= 0:
    model = model.cuda(args.dev)
torch.cuda.empty_cache()

acc_test, time_test, outl, labl = eval(x=feat['test'], edge_idx=adj['test'],
                                       y=labels['test'], idx_split=idx['test'])

if args.seed >= 7:
    mem_ram, mem_cuda = metric.get_ram(), metric.get_cuda_mem(args.dev)
    print(f"[Val] best acc: {acc_best:0.4f}, [Test] best acc: {acc_test:0.4f}", flush=True)
    # print(f"[Train] time cost: {time_train:0.4f}, Best epoch: {conv_epoch}, Epoch avg: {time_train*1000 / (epoch+1):0.1f}")
    # print(f"[Test]  time cost: {time_test:0.4f}, RAM: {mem_ram / 2**20:.3f} GB, CUDA: {mem_cuda / 2**30:.3f} GB")
    # print(f"Num params (M): {metric.get_num_params(model):0.4f}, Mem params (MB): {metric.get_mem_params(model):0.4f}")
