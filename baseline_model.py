import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG_CNN_Baseline(nn.Module):
    def __init__(self, n_channels, n_timepoints, n_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 7), padding=(1, 3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 7), padding=(1, 3))

        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.3)

        # calculate feature size after convs
        self.feature_dim = 32 * (n_channels // 4) * (n_timepoints // 4)

        self.fc1 = nn.Linear(self.feature_dim, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)  # (B, 1, C, T)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
