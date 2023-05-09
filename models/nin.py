import torch
from torch import nn

def nin_block(channels, kernel_size, stride, padding):
    return nn.Sequential(
            nn.LazyConv2d(channels, kernel_size, stride, padding), nn.ReLU(),
            nn.LazyConv2d(channels, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(channels, kernel_size=1), nn.ReLU()
    )

class NiN(nn.Module):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
                nin_block(96, kernel_size=11, stride=4, padding=0),
                nn.MaxPool2d(3, stride=2),
                nin_block(256, kernel_size=5, stride=1, padding=2),
                nn.MaxPool2d(3, stride=2),
                nin_block(384, kernel_size=2, stride=1, padding=1),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, stride=1, padding=1),
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten()
                )

    def forward(self,x):
        return self.net(x)


if __name__ == '__main__':
    x = torch.rand([1,3,256,256])
    y = NiN()(x)
    print(y.shape)
