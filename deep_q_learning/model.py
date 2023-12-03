import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, n):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n * n, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n * n * 2)  # Output size: n*n*2 (for each element in the matrix, two possible actions)

    def forward(self, x):
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
