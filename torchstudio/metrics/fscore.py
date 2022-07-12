from torchstudio.modules import Metric
import torch.nn.functional as F
import torch
import math

class FScore(Metric):
    """FScore for binary and multilabel data
    FScore = (1+beta^2)*(precision*recall)/(beta^2*precision+recall): https://en.m.wikipedia.org/wiki/F-score
    NB: For multiclass data, fscore=accuracy:
        https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
        https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
        https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

    Args:
        beta: beta factor for the F score. 1.0 gives a F1 score.
        normalize: if set to True, normalize predictions with sigmoid or softmax before calculating the accuracy
    """
    def __init__(self, beta: float = 1.0, normalize: bool = False):
        self.beta_square = math.pow(beta,2)
        self.normalize = normalize
        self.reset()

    def update(self, preds, target):
        if len(preds.shape)==len(target.shape)+1 and preds.shape[0]==target.shape[0]:
            if self.normalize:
                preds=F.softmax(preds, dim=1)
            tp = torch.sum(torch.eq(torch.argmax(preds, dim=1), target).view(-1))
            tpfp = torch.tensor(tp.shape[0])
            tpfn = torch.tensor(tp.shape[0])
        elif preds.shape==target.shape:
            if self.normalize:
                preds=F.sigmoid(preds)
            tpfn = torch.sum(torch.greater(target, .5))
            tpfp = torch.sum(torch.greater(preds, .5))
            tp = torch.sum(torch.bitwise_and(tpfn, torch.greater(target, .5)))
        else:
            raise ValueError("prediction and target have different shapes or aren't compatible with multiclass prediction")

        self.tpfn += tpfn
        self.tpfp += tpfp
        self.tp += tp
    def compute(self):
        precision = self.tp.float() / self.tpfp.float()
        recall = self.tp.float() / self.tpfn.float()
        fscore = (1.0+self.beta_square)*(precision*recall)/(self.beta_square*precision+recall)
        return fscore

    def reset(self):
        self.tpfn = 0
        self.tpfp = 0
        self.tp = 0

