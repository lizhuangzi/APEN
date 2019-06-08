import torch.nn as nn

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