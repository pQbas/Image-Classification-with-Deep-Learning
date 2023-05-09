import torch
from torch import nn


def vgg_block(num_convs, channels, conv1=False, LRN=False):
    layers = []
    for it in range(num_convs):
        layers.append(nn.LazyConv2d(channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def backbone(arch):
    conv_blks = []
    for (num_convs, channels) in arch:
        conv_blks.append(vgg_block(num_convs, channels))
    return nn.Sequential(*conv_blks)


class VGG(nn.Module):
    def __init__(self, config, num_classes=10, lr=0.1):
        super(VGG, self).__init__()
        self.backbone = None
        self.arch = None
        
        if config == 'A':
            self.arch = ((1,64),(1,128),(2,256),(2,512),(2,512))
        
        if config == 'B':
            self.arch = ((2,64),(2,128),(2,256),(2,512),(2,512))
        
        if config == 'D':
            self.arch = ((2,64),(2,128),(3,256),(3,512),(3,512))
        
        if config == 'E':
            self.arch = ((2,64),(2,128),(4,256),(4,512),(4,512))

        if self.arch is not None:
            self.backbone_ = backbone(self.arch)
            self.net = nn.Sequential(
                *self.backbone_, nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(num_classes)
            )
            print(f'Configuration "{config}" was selected for the backbone')
        
        else:
            print('WARNING: Any VGG configuration for the backbone was selected')
            exit()

    def forward(self,x):
        return self.net(x)


if __name__ == '__main__':

    #arch = ((1,64),(1,128),(2,256),(2,512),(2,512))
    x = torch.rand([1,3,256,256])
    y = VGG(config='B')(x)
    print(x.shape)
    print(y.shape)
