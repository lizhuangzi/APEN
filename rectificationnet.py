import torch.nn as nn

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