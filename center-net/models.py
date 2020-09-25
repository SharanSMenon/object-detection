import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """A Basic ResNet Block"""
    def __init__(self, inplanes, planes, stride=1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet34(nn.Module):
    """Modified version of ResNet 34 for CenterNet"""
    def __init__(self):
        self.inplanes = 64
        layers = [3, 4, 6, 3]
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(*[
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            ])
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

class DeConvNet(nn.Module):
    def __init__(self):
        super(DeConvNet, self).__init__()

        self.layer1 = self._make_deconv_block(512, 256)
        self.layer2 = self._make_deconv_block(256, 128)
        self.layer3 = self._make_deconv_block(128, 64)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)
    
    def _make_deconv_block(self, in_ch, out_ch):
        return nn.Sequential(*[
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, in_ch, 4, 2, 1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        ])

class CenterNet(nn.Module):
    """Some Information about CenterNet"""
    def __init__(self):
        super(CenterNet, self).__init__()
        self.resnet = ResNet34()
        self.deconvnet = DeConvNet()



    def forward(self, x):

        x = self.resnet(x)
        x = self.deconvnet(x)

        return x