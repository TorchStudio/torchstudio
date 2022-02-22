from torch import Tensor
from typing import Callable, Optional
import torch.nn as nn

class SmoothMAE(nn.SmoothL1Loss):
    """Smooth Mean Absolute Error regression loss (based on SmoothL1Loss)
    https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
    Target values should be float numbers.
    Behaves as L1 when the absolute value is higher than Beta (default: 1),
    and behaves like L2 when the absolute value is lower than Beta.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        beta (float, optional): Specifies the threshold at which to change between L1 and L2 loss.
            The value must be non-negative. Default: 1.0
    """
    def __init__(self, reduction: str = 'mean', beta: float = 1.0):
        super().__init__(size_average=None, reduce=None, reduction=reduction, beta=beta)
