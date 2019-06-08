import torch.nn as nn
from torchvision import models

vgg16 = models.vgg16(pretrained=True)
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