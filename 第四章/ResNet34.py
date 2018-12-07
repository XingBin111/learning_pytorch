import torch as t
from torch import nn
from torch.nn import functional as F

# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcuts=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcuts

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )                                                               # 224 --> 56

        self.layer1 = self._make_layers(64, 64, 3)
        self.layer2 = self._make_layers(64, 128, 4, 2)                  # 56 --> 28
        self.layer3 = self._make_layers(128, 256, 6, 2)                 # 28 --> 14
        self.layer4 = self._make_layers(256, 512, 3, 2)                 # 14 --> 7

        self.fc = nn.Linear(512, num_classes)

    def _make_layers(self, inchannel, outchannel, block_num, stride=1):
        shortcuts = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcuts))

        for i in range(block_num-1):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)

        x = x.view(x.size(0), -1)
        return self.fc(x)

model = ResNet()
input = t.randn(1, 3, 224, 224)
o = model(input)

# from torchvision import models
# model1 = models.resnet34()
# o1 = model1(input)