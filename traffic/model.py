import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features=77, out_features=35):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.head = nn.Linear(32, out_features)
        self.act = nn.GELU()

    def forward(self, x):
        # input x: B, x_cols
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.head(x)
        return x # B, y_cols