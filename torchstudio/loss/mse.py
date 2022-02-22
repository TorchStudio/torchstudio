from torch import Tensor
from typing import Callable, Optional
import torch.nn as nn

class MeanSquareError(nn.MSELoss):
    """Mean Square Error regression loss (based on MSELoss)
    https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    Target values should be float numbers.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__(size_average=None, reduce=None, reduction=reduction)
