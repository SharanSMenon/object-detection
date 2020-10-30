import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBase(nn.Module):
    """An implementation of VGG"""
    def __init__(self):
        super(VGGBase, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    def forward(self, image: torch.Tensor):
        # [N, 3, 300, 300] <- Starting Size
        out = F.relu(self.conv1_1(image)) # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out)) # (N, 64, 300, 300)
        out = self.pool1(out) # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out)) # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out)) # (N, 128, 150, 150)
        out = self.pool2(out) # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out)) # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out)) # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out)) # (N, 256, 75, 75)
        out = self.pool3(out) # (N, 256, 38, 38)
        
        out = F.relu(self.conv4_1(out)) # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out)) # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out)) # (N, 512, 38, 38)
        conv4_3_features = out
        out = self.pool4(out) # (N, 512, 19, 19)

        out = F.relu(self.conv4_1(out)) # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out)) # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out)) # (N, 512, 19, 19)
        out = self.pool5(out) # (N, 512, 19, 19) <- Does not reduce dimensions

        out = F.relu(self.conv6(out)) # (N, 1024, 19, 19)
        conv7_features = self.conv7(out) # (N, 1024, 19, 19)

        return conv4_3_features, conv7_features