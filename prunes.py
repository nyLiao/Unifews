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


class ThrInPruningMethod(prune.BasePruningMethod):
    """Prune by input-dimension thresholding.
    """
    PRUNING_TYPE = 'unstructured'
    def __init__(self, threshold):
        """Args:
            threshold (Tensor [F_in]): threshold for each input channel
        """
        self.threshold = threshold

    def compute_mask(self, t, default_mask):
        """Args:
            t (Tensor [F_out, F_in]): tensor to prune
        """
        assert self.threshold.shape == t.shape[1:]
        mask = default_mask.clone()
        mask[t.abs() < self.threshold] = 0
        return mask

    @classmethod
    def apply(cls, module, name, threshold):
        return super().apply(module, name, threshold=threshold)
