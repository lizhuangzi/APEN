import torch.nn as nn
import torch
import torch.nn.functional as F


def swish(x):
    return x * F.sigmoid(x)
class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))
class srnet(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(srnet, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor//2):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)
    def forward(self, x):
        x = swish(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks-4):
            y = self.__getattr__('residual_block' + str(i+1))(y)
        srf1=y
        srf2 = self.__getattr__('residual_block' + str(self.n_residual_blocks - 3))(srf1)
        srf3 = self.__getattr__('residual_block' + str(self.n_residual_blocks - 2))(srf2)
        srf4 = self.__getattr__('residual_block' + str(self.n_residual_blocks - 1))(srf3)
        srf5 = self.__getattr__('residual_block' + str(self.n_residual_blocks))(srf4)
        x = self.bn2(self.conv2(srf5)) + x
        for i in range(self.upsample_factor//2):
            x = self.__getattr__('upsample' + str(i+1))(x)

        sr=self.conv3(x)
        srf=torch.cat([srf1, srf2, srf3, srf4, srf5], 1)

        return sr,srf








