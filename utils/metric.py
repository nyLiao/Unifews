import time
import resource
import numpy as np
import torch
import torch.nn as nn


class F1Calculator(object):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.TP = torch.zeros(self.num_classes)
        self.FP = torch.zeros(self.num_classes)
        self.FN = torch.zeros(self.num_classes)

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            y_true = nn.functional.one_hot(y_true, num_classes=self.num_classes)
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
            y_pred = nn.functional.one_hot(y_pred, num_classes=self.num_classes)
        self.TP += (y_true * y_pred).sum(dim=0).cpu()
        self.FP += ((1 - y_true) * y_pred).sum(dim=0).cpu()
        self.FN += (y_true * (1 - y_pred)).sum(dim=0).cpu()

    def compute(self, average: str=None):
        eps = 1e-10
        if average == 'micro':
            # For multi-class classification, F1 micro is equivalent to accuracy
            f1 = 2 * self.TP.float().sum() / (2 * self.TP.sum() + self.FP.sum() + self.FN.sum() + eps)
            return f1.item()
        elif average == 'macro':
            f1 = 2 * self.TP.float() / (2 * self.TP + self.FP + self.FN + eps)
            return f1.mean().item()
        else:
            raise ValueError('average must be "micro" or "macro"')


class Stopwatch(object):
    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def pause(self) -> float:
        """Pause clocking and return elapsed time"""
        self.elapsed_sec += time.time() - self.start_time
        self.start_time = None
        return self.elapsed_sec

    def lap(self) -> float:
        """No pausing, return elapsed time"""
        return time.time() - self.start_time + self.elapsed_sec

    def reset(self):
        self.start_time = None
        self.elapsed_sec = 0

    @property
    def time(self) -> float:
        return self.elapsed_sec


class Accumulator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0

    def update(self, val: float, count: int=1):
        self.val += val
        self.count += count
        return self.val

    @property
    def avg(self) -> float:
        return self.val / self.count


def get_ram() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def get_cuda_mem(dev) -> float:
    return torch.cuda.max_memory_allocated(dev) / 2**30


def get_num_params(model: nn.Module) -> float:
    num_paramst = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    num_params = sum([param.nelement() for param in model.parameters()])
    num_bufs = sum([buf.nelement() for buf in model.buffers()])
    # return num_paramst/1e6, num_params/1e6, num_bufs/1e6
    return num_paramst/1e6


def get_mem_params(model: nn.Module) -> float:
    mem_paramst = sum([param.nelement()*param.element_size() for param in model.parameters() if param.requires_grad])
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    # return mem_paramst/(1024**2), mem_params/(1024**2), mem_bufs/(1024**2)
    return (mem_params+mem_bufs)/(1024**2)
