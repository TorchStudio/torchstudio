from torch import Tensor
from typing import Callable, Optional
import torch
import torch.nn as nn

class BinaryCrossEntropy(nn.Module):
    """Binary Cross Entropry classification loss (based on BCELoss and BCEWithLogits)
    https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    Target value(s) must be 0 or 1. Can be used for binary classification or
    multi-label multi-class binary classification when multiple values are provided.

    Args:
        output_type (string): Type of output. Can be 'auto', 'logits' or 'sigmoid'
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    """
    def __init__(self, output_type: str = 'auto', weight: Optional[Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.output_type=output_type
        self.bceloss=nn.BCELoss(weight, size_average=None, reduce=None, reduction=reduction)
        self.bcelosswithlogits=nn.BCEWithLogitsLoss(weight, size_average=None, reduce=None, reduction=reduction)

    def forward(self, input, target):
        if self.output_type=='auto':
            min_value=torch.min(input)
            max_value=torch.max(input)
            if min_value<0 or max_value>1: #no sigmoid was applied
                self.output_type='logits'
            else:
                self.output_type='sigmoid'

        if self.output_type=='logits':
            return self.bcelosswithlogits.forward(input,target)
        else:
            return self.bceloss.forward(input,target)


