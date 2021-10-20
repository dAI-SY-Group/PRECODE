import numpy as np
from torch import nn

from src.VariationalBottleneck import VariationalBottleneck

class MLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3,32,32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

class VBMLP(nn.Module):
    def __init__(self, width=1024, num_classes=10, data_shape=(3,32,32)):
        super().__init__()
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(np.prod(data_shape), width)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(width, width)
        self.VB = VariationalBottleneck((width,))
        self.l3 = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.flat(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.VB(x)
        x = self.l3(x)
        return x

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()
    