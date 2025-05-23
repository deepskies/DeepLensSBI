import torch
import torch.nn as nn
import torch.nn.functional as F


class SummaryNet(nn.Module):
    """
    Summary Network module
    """

    def __init__(self, out_features):
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm2d(128)

        # Maxpool layer that reduces 32x32 image to 4x4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # out_features correspond to 4*(Number of summary parameters)
        self.fc = nn.Linear(in_features=128 * 4 * 4, out_features=out_features)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)

        x = (self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(self.bn4(F.relu(self.conv4(x))))

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.pool(self.bn6(F.relu(self.conv6(x))))

        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc(x))
        return x
