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
        
        self.stage1 = nn.Sequential(
            conv(1, 96, 11, 4, 2),
            lrn(5),
            pooling(3, 2)
        )
        
        self.stage2 = nn.Sequential(
            conv(96, 256, 5, 1, 2),
            lrn(5),
            pooling(3, 2),
        )
        
        self.stage3 = conv(256, 384, 3, 1, 1)
        
        self.stage4 = conv(384, 384, 3, 1, 1)
        
        self.stage5 = nn.Sequential(
            conv(384, 256, 3, 1, 1),
            pooling(3, 3)
        )
        
        self.stage6 = nn.Sequential(
            linear(1024, 1024),
            linear(1024, 10),
            nn.Softmax(dim=1)
        )
        
        # intializing weights
        self.initialize_weights()
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = torch.flatten(x, 1)
        x = self.stage6(x)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                        
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 1)
                    
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        
        
if __name__=='__main__':
        
    model = alexnet()
    
    with torch.no_grad():
        x = torch.rand((32,1,128,128))
        y = model(x)
        print(y.shape)