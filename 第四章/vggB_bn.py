import torch.nn as nn
import torch as t


class vggB_bn(nn.Module):
    def __init__(self, num_classes=1000):
        super(vggB_bn, self).__init__()
        self.features = nn.Sequential(
            self._make_layers(3, 64),
            self._make_layers(64, 128),
            self._make_layers(128, 256),
            self._make_layers(256, 512),
            self._make_layers(512, 512)
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 7 * 7 * 512)
        return self.fc(x)

    def _make_layers(self, inchannel, outchannel):
        _seq = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(outchannel, outchannel, 3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        return _seq


model = vggB_bn()
x = t.randn(1, 3, 224, 224)
o = model.forward(x)
