import torch
import torch.nn as nn
import torch.nn.functional as F

class SimAM(nn.Module):
    """
    Simple Attention Module (SimAM) - Zero parameter 3D attention.
    Calculates attention weights using an energy function based on spatial variance.
    Paper: "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks"
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        return x * self.activaton(y)

class BasicBlock(nn.Module):
    """Standard ResNet Basic Block integrated with SimAM."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Injecting the Zero-Parameter Attention Module
        self.simam = SimAM()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.simam(out) # Apply SimAM before the skip connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetSimAM(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2, dropout_rate=0.3):
        super(ResNetSimAM, self).__init__()
        self.in_planes = 64

        # Modified for 1-channel input (LFCCs/Spectrograms) instead of 3-channel RGB
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x expected shape: (Batch, 1, Freq_Bins, Time_Frames)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        
        logits = self.fc(out)
        return logits

def resnet18_simam(num_classes=2, dropout_rate=0.3):
    """Constructs a ResNet-18 model with SimAM."""
    return ResNetSimAM(BasicBlock, [2, 2, 2, 2], num_classes, dropout_rate)