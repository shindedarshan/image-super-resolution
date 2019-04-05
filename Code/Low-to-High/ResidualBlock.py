import torch.nn as nn
from Convolution import conv3x3

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False, upsample=False, nobn = False):
        super(ResidualBlock, self).__init__()
        self.upsample = upsample
        self.downsample = downsample
        self.nobn = nobn

        if self.upsample:
            self.conv1 = nn.ConvTranspose2d(inplanes, planes, 4, 2, 1)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)

        if not self.nobn:
            self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.downsample:
            self.conv2 =nn.Sequential(nn.AvgPool2d(2,2), conv3x3(planes, planes))
        else:
            self.conv2 = conv3x3(planes, planes)

        if not self.nobn:
            self.bn2 = nn.BatchNorm2d(planes)

        if inplanes != planes or self.upsample or self.downsample:
            if self.upsample:
                self.skip = nn.ConvTranspose2d(inplanes, planes, 4, 2, 1)
            elif self.downsample:
                self.skip = nn.Sequential(nn.AvgPool2d(2,2), nn.Conv2d(inplanes, planes, 1, 1))
            else:
                self.skip = nn.Conv2d(inplanes, planes, 1, 1, 0)
        else:
            self.skip = None

        self.stride = stride

    def forward(self, x):
        residual = x

        if not self.nobn:
            out = self.bn1(x)
            out = self.relu(out)
        else:
            out = self.relu(x)

        out = self.conv1(out)

        if not self.nobn:
            out = self.bn2(out)

        out = self.relu(out)
        out = self.conv2(out)

        if self.skip is not None:
            residual = self.skip(x)

        out += residual

        return out
