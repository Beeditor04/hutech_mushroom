import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MiniAlexNet, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Convolutional Layer 3 (Sửa lại)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=3, stride=2, padding=1)  # Giữ kernel size 3, stride 2
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Fully Connected Layers with Dropout
        self.fc1 = nn.Linear(256 * 6 * 6, 512)  # Điều chỉnh theo kích thước đầu vào
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x