from torchstudio.modules import Metric
import torch.nn.functional as F
import torch

class Precision(Metric):
    """Precision for binary and multilabel data
    Precision = TP / (TP + FP): https://en.wikipedia.org/wiki/Precision_and_recall
    NB: For multiclass data, precision=accuracy:
        https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
        https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
        https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

    Args:
        normalize: if set to True, normalize predictions with sigmoid or softmax before calculating the accuracy
    """
    def __init__(self, normalize: bool = False):
        self.normalize = normalize
        self.reset()

    def update(self, preds, target):
        if len(preds.shape)==len(target.shape)+1 and preds.shape[0]==target.shape[0]:
            if self.normalize:
                preds=F.softmax(preds, dim=1)
            tp = torch.sum(torch.eq(torch.argmax(preds, dim=1), target).view(-1))
            tpfp = tp.shape[0]
        elif preds.shape==target.shape:
            if self.normalize:
                preds=F.sigmoid(preds)
            tpfp = torch.sum(torch.greater(preds, .5))
            tp = torch.sum(torch.bitwise_and(tpfp, torch.greater(target, .5)))
        else:
            raise ValueError("prediction and target have different shapes or aren't compatible with multiclass prediction")

        self.tpfp += tpfp
        self.tp += tp
    def compute(self):
        return self.tp / self.tpfp

    def reset(self):
        self.tpfp = 0
        self.tp = 0
