from torch import Tensor
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    """Cross Entropy classification loss (based on NLLLoss and CrossEntropyLoss)
    https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    You need a log softmax as your last layer. Target value must be a class number.
    Can be used for multi-class classification.

    Args:
        output_type (string): Type of output. Can be 'auto', 'logits', 'softmax' or 'logsoftmax'
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
    """
    def __init__(self, output_type: str = 'auto', weight: Optional[Tensor] = None, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
        self.output_type=output_type
        self.nllloss=nn.NLLLoss(weight, size_average=None, ignore_index=ignore_index, reduce=None, reduction=reduction)
        self.crossentropyloss=nn.CrossEntropyLoss(weight, size_average=None, ignore_index=ignore_index, reduce=None, reduction=reduction)


    def forward(self, input, target):
        if self.output_type=='auto':
            min_value=torch.min(input)
            max_value=torch.max(input)
            if min_value<0 and max_value>0: #logits, no logsoftmax or softmax was applied
                self.output_type='logits'
            if min_value>=0 and max_value<=1: #softmax was applied
                self.output_type='softmax'
            if min_value<0 and max_value<=0: #logsoftmax was applied
                self.output_type='logsoftmax'

        if self.output_type=='logits':
            return self.crossentropyloss.forward(input,target)
        elif self.output_type=='logsoftmax':
            return self.nllloss.forward(input,target)
        else:
            return self.nllloss.forward(torch.log(input),target)
