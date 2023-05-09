import torch
import torch.nn as nn
from torch.nn import functional as F

class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, *kwargs):
        super(Inception, self).__init__(*kwargs)

        #Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)

        #Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)

        #Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)

        #Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)


    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(F.relu(self.b4_1(x))))
        return torch.cat((b1, b2, b3, b4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
                self.stage1(),
                self.stage2(),
                self.inception_3(),
                self.inception_4(),
                self.inception_5(),
                self.classification(),
                nn.LazyLinear(num_classes)
                )


    def stage1(self):
        return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )

    def stage2(self):
        return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=1), 
                nn.ReLU(),
                nn.LazyConv2d(192, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )

    def inception_3(self):
        return nn.Sequential(
                Inception(64, (96, 128), (16, 32), 32),
                Inception(128, (128, 192), (32, 96), 64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )

    def inception_4(self):
        return nn.Sequential(
                Inception(192, (96, 208), (16, 48), 64),
                Inception(160, (112, 224), (24, 64), 64),
                Inception(128, (128, 256), (24, 64), 64),
                Inception(112, (144, 288), (32, 64), 64),
                Inception(256, (160, 320), (32, 128), 68)
                )

    def inception_5(self):
        return nn.Sequential(
                Inception(256, (160, 320), (32, 128), 128),
                Inception(384, (192, 384), (48, 128), 128)
                )

    def classification(self):
        return nn.Sequential(
                Inception(256, (168, 320), (32, 128), 128),
                Inception(384, (192, 384), (48, 128), 128),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten()
                )

    def forward(self, x):
        return self.net(x)



if __name__=='__main__':
    x = torch.rand([1,3,224,224])
    y = GoogLeNet()(x)
    print(y.shape)
