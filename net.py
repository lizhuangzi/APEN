import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import math
vgg16 = models.vgg16(pretrained=True)
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

class vgg(nn.Module):
    def __init__(self,num_classes):
        super(vgg, self).__init__()
        self.features = vgg16.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.model_name = 'vgg16'
    def forward(self, x, y):
        c1=self.features[0:26](x)
        c2=self.features[26:28](c1)
        c3=self.features[28:30](c2)
        feature = self.features[30](c3)
        f = feature.view(feature.size(0), -1)
        if type(y)!= int:
            f=f+y
        c = self.classifier(f)
        return c,c1,c2,c3


class attNet(nn.Module):
    def __init__(self):
        super(attNet, self).__init__()
        self.cov1=nn.Conv2d(320, 64, 3, stride=1, padding=1)
        self.att = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=0, bias=True),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(64, 1, 1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.cov1(x)
        x = self.att(x)
        return x


class srcorrectNet(nn.Module):
    def __init__(self):
        super(srcorrectNet, self).__init__()
        self.fea = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        )
    def forward(self, x):
        feature=self.fea(x)
        f = feature.view(feature.size(0), -1)
        return f





if __name__ == '__main__':
    print (vgg16)

