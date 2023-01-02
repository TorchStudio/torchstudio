from torchstudio.modules import Metric
import torch.nn.functional as F
import torch

class Accuracy(Metric):
    """Accuracy for binary, multiclass, multilabel and regression data

    Args:
        threshold: error threshold below which predictions are considered accurate (not used in multiclass)
        normalize: if set to True, normalize predictions with sigmoid or softmax before calculating the accuracy
    """
    def __init__(self, threshold: float = 0.01, normalize: bool = False):
        self.threshold = threshold
        self.normalize = normalize
        self.reset()

    def update(self, preds, target):
        if len(preds.shape)==len(target.shape)+1 and preds.shape[0]==target.shape[0]:
            if self.normalize:
                preds=F.softmax(preds, dim=1)
            correct = torch.eq(torch.argmax(preds, dim=1), target).view(-1)
        elif preds.shape==target.shape:
            if self.normalize:
                preds=F.sigmoid(preds)
            correct = torch.less(torch.abs(preds-target.float()),self.threshold).view(-1)
        else:
            raise ValueError("prediction and target have different shapes or aren't compatible with multiclass prediction")

        self.num_correct += torch.sum(correct)
        self.num_samples += torch.tensor(correct.shape[0])

    def compute(self):
        if self.num_samples == 0:
            raise ValueError("Accuracy must have at least one sample before it can be computed.")
        return self.num_correct.float() / self.num_samples.float()

    def reset(self):
        self.num_correct = 0
        self.num_samples = 0
