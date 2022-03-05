import torch.nn as nn


class MLP(nn.Module): 
    def __init__(self, in_features=75):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.head = nn.Linear(32, 2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.head(x)
        return x