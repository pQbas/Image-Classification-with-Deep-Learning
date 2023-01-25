import torch
import torch.nn as nn


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv, self).__init__()
        self.filter_bank = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.filter_bank(x)
        return x


class pooling(nn.Module):
    def __init__(self, size, stride):
        super(pooling, self).__init__()
        self.pooling = torch.nn.MaxPool2d(kernel_size=size, stride=stride)

    def forward(self, x):
        x = self.pooling(x)
        return x


class lrn(nn.Module):
    def __init__(self, neighbors):
        super(lrn, self).__init__()
        self.lrn_ = nn.LocalResponseNorm(neighbors)

    def forward(self, x):
        x = self.lrn_(x)
        return x


class linear(nn.Module):
    def __init__(self, input, output):
        super(linear, self).__init__()
        self.linear_ = nn.Sequential(
            nn.Linear(input, output),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.linear_(x)
        return x


class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
    
    def forward(self, x):
        # first stage
        x = conv(3, 96, 11, 4, 2)(x)
        x = lrn(5)(x)
        x = pooling(3, 2)(x)
        
        # second stage
        x = conv(96, 256, 5, 1, 2)(x)
        x = lrn(5)(x)
        x = pooling(3, 2)(x)
        
        # third stage
        x = conv(256, 384, 3, 1, 1)(x)
        
        # fourth stage
        x = conv(384, 384, 3, 1, 1)(x)
        
        # fifth stage
        x = conv(384, 256, 3, 1, 1)(x)
        x = pooling(3, 3)(x)
        
        # six stage
        x = x.view(-1)
        x = linear(4096, 4096)(x)
        x = linear(4096, 1000)(x)
        x = nn.Softmax(0)(x)
        return x
        
        
if __name__=='__main__':
        
    model = alexnet()
    
    with torch.no_grad():
        x = torch.rand((1,3,224,224))
        y = model(x)
        print(y.shape)