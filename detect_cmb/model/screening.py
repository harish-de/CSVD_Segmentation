import torch.nn as nn
import torch.nn.functional as F

class screening(nn.Module):
    def __init__(self):
        super(screening, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5, 3), stride=(1, 1, 1), bias=True)
        self.max1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=True)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 1), stride=(1, 1, 1), bias=True)
        self.fc1 = nn.Conv3d(in_channels=64, out_channels=150, kernel_size=(2, 2, 2), stride=(1, 1, 1), bias=True)
        self.fc2 = nn.Conv3d(in_channels=150, out_channels=2, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=True)
        self.drop = nn.Dropout3d(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x