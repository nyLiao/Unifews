import torch
import torch.nn.utils.prune as prune


def prune_threshold(x, threshold=1e-3):
    idx_0 = torch.norm(x, dim=1)/x.shape[1] < threshold
    x[idx_0] = 0
    return x, idx_0


def prune_topk(x, k=0.2):
    num_0 = int(x.shape[0] * k)
    x_norm = torch.norm(x, dim=1)
    _, idx_0 = torch.topk(x_norm, num_0)
    x[idx_0] = 0
    return x, idx_0


class ThrInPrune(prune.BasePruningMethod):
    """Prune by input-dimension thresholding.
    """
    PRUNING_TYPE = 'structured'
    def __init__(self, threshold, dim=0):
        """Args:
            threshold (Tensor [F_in]): threshold for each input channel
        """
        self.threshold = threshold
        self.dim = dim      # prune on dim=0, i.e. keep dim=1 unchanged

    def compute_mask(self, t, default_mask):
        """Args:
            t (Tensor [F_out, F_in]): tensor to prune
        """
        assert self.threshold.shape == t.shape[1:]
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        mask[t.abs() < self.threshold] = 0
        return mask

    @classmethod
    def apply(cls, module, name, threshold):
        return super().apply(module, name, threshold=threshold)


class ThrProdPrune(prune.BasePruningMethod):
    """Prune by thresholding weight-input product.
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, threshold):
        """Args:
            threshold (Scalar | Tensor [F_in]): threshold for each input channel
        """
        self.threshold = threshold

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask[t.abs() < self.threshold] = 0
        return mask

    @classmethod
    def apply(cls, module, name, threshold, x):
        """Args:
            w (Tensor [F_out, F_in]): weight tensor
            x (Tensor [N, F_in]): input tensor
        """
        w = getattr(module, name)
        assert w.shape[1] == x.shape[1]
        score = w.abs() * torch.norm(x, dim=0)
        return super().apply(module, name, threshold=threshold, importance_scores=score)
