import torch.nn as nn
import torch.nn.functional as F

class discrimination(nn.Module):
    def __init__(self):
        super(discrimination, self).__init__()
        self.zero_pad = nn.ConstantPad3d
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32,
                               kernel_size=(7,7,5), stride=(1, 1, 1), bias=False)
        self.max1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(5,5,3), stride=(1, 1, 1), bias=False)

        self.fc1 = nn.Linear(in_features=64*3*3*4, out_features=500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 2)


    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1,64*3*3*4)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x
