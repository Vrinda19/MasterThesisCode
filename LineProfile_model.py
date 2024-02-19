from .layers import *
import torch.nn as nn
from torch.nn import Module


class CustomModel4(nn.Module):
    def __init__(self, input_dim, cls):
        super(CustomModel4, self).__init__()
        self.feature_x = nn.Sequential(
            # nn.GaussianNoise(0.01),
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=4, stride=1),
            nn.Flatten(),
        )

        self.feature_x_x = nn.Sequential(
            # nn.GaussianNoise(0.01),
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=4, stride=1),
            nn.Flatten(),
        )

        self.feature_f = nn.Sequential(nn.Flatten())
        self.combined_features = nn.Sequential()

        self.fc = nn.Sequential(
            nn.Linear(130160, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, cls)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.feature_x(x)
        x2 = self.feature_x_x(x)
        x6 = self.feature_f(x)
        combined_features = torch.cat((x1, x2, x6), dim=1)
        # print(x.shape)
        fc_x = self.fc(combined_features)
        return fc_x
