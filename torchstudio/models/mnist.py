import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    """MNIST Classifier as described here:
    https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
    https://github.com/pytorch/examples/blob/master/mnist/main.py

    To be used with the MNIST dataset (in torchvision.datasets)

    Args:
        use_dropouts (bool): Whether to use dropout nodes or not
    """
    def __init__(self, use_dropouts:bool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.use_dropouts=use_dropouts

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.use_dropouts:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.use_dropouts:
            x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
