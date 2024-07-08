import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'prelu':
            self.act = nn.PReLU()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation != 'no':
            x = self.act(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = nn.InstanceNorm2d(output_size)

        self.activation = activation
        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'prelu':
            self.act = nn.PReLU()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.deconv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv2d(x)
