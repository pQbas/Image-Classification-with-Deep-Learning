import torch
import torch.nn as nn
from torch.nn import functional as F

def stem():
    layers = [nn.LazyConv2d(32, 3, 2),
            nn.LazyConv2d(32, 3),
            nn.LazyConv2d(64, 3, 1, 1),
            nn.MaxPool2d(3, 2),
            nn.LazyConv2d(80, 1),
            nn.LazyConv2d(192, 3),
            nn.LazyConv2d(192, 3),
            nn.LazyConv2d(256, 3,2,1)]

    return nn.Sequential(*layers)

class inception_resnet_a(nn.Module):
    def __init__(self,**kwargs):
        super(inception_resnet_a, self).__init__(**kwargs)
        #Branch 1
        self.b1 = nn.LazyConv2d(32, 1)

        #Branch 2
        self.b2_1 = nn.LazyConv2d(32, 1)
        self.b2_2 = nn.LazyConv2d(32, 3, 1, 1)

        #Branch 3
        self.b3_1 = nn.LazyConv2d(32, 1)
        self.b3_2 = nn.LazyConv2d(32, 3, 1, 1)
        self.b3_3 = nn.LazyConv2d(32, 3, 1, 1)

        #Concatenation
        self.concat = nn.LazyConv2d(256,1)

    def forward(self,x):
        #Branch 1
        x1 = F.relu(self.b1(x))

        #Branch 2
        x2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))

        #Branch 3
        x3 = F.relu(self.b3_3(F.relu(self.b3_2(F.relu(self.b3_1(x))))))

        #Concatenation and dimensionality reduction
        x4 = F.relu(self.concat(torch.cat((x3,x2,x1),dim=1)))

        #Skip conection
        return F.relu(torch.add(x4,x))



class inception_resnet_b(nn.Module):
    def __init__(self, **kwargs):
        super(inception_resnet_b, self).__init__(**kwargs)
        #Branch 1
        self.b1 = nn.LazyConv2d(128, 1)

        #Branch 2
        self.b2_1 = nn.LazyConv2d(128, 1, 1, 1)
        self.b2_2 = nn.LazyConv2d(128, (1,7), 1, 1)
        self.b2_3 = nn.LazyConv2d(128, (7,1), 1, 1)

        #Joint
        self.joint = nn.LazyConv2d(896, 1)

    def forward(self, x):
        x1 = F.relu(self.b1(x))
        x2 = F.relu(self.b2_3(F.relu(self.b2_2(F.relu(self.b2_1(x))))))
        x3 = F.relu(self.joint(torch.cat((x1,x2), dim=1)))
        return F.relu(torch.add(x,x3))



class inception_resnet_c(nn.Module):
    def __init__(self, **kwargs):
        super(inception_resnet_c, self).__init__(**kwargs)
        #Branch 1
        self.b1 = nn.LazyConv2d(192, 1)

        #Branch 2
        self.b2_1 = nn.LazyConv2d(192, 1, 1, 1)
        self.b2_2 = nn.LazyConv2d(192, (1,3), 1)
        self.b2_3 = nn.LazyConv2d(192, (3,1), 1)

        #Joint
        self.joint = nn.LazyConv2d(1792, 1)

    def forward(self, x):
        x1 = F.relu(self.b1(x))
        x2 = F.relu(self.b2_3(F.relu(self.b2_2(F.relu(self.b2_1(x))))))
        x3 = F.relu(self.joint(torch.cat((x1,x2), dim=1)))
        return F.relu(torch.add(x,x3))


class reduction_module_a(nn.Module):
    def __init__(self, **kwargs):
        super(reduction_module_a, self).__init__(**kwargs)

        #branch1
        self.b1 = nn.MaxPool2d(3, 2)

        #branch2
        self.b2 = nn.LazyConv2d(384, 3, 2)

        #branch3
        self.b3_1 = nn.LazyConv2d(192, 1)
        self.b3_2 = nn.LazyConv2d(192, 3, 1, 1)
        self.b3_3 = nn.LazyConv2d(256, 3, 2)

        #filter concat
        self.filter_concat = nn.LazyConv2d(896,1)


    def forward(self, x):
        x1 = self.b1(x)
        x2 = F.relu(self.b2(x))
        x3 = F.relu(self.b3_3(F.relu(self.b3_2(F.relu(self.b3_1(x))))))
        x = F.relu(self.filter_concat(torch.cat((x1,x2,x3),dim=1)))
        return x


class reduction_module_b(nn.Module):
    def __init__(self, **kwargs):
        super(reduction_module_b, self).__init__(**kwargs)

        #branch1
        self.b1 = nn.MaxPool2d(3, 2)

        #branch2
        self.b2_1 = nn.LazyConv2d(256, 1)
        self.b2_2 = nn.LazyConv2d(384, 3, 2)

        #branch3
        self.b3_1 = nn.LazyConv2d(256, 1)
        self.b3_2 = nn.LazyConv2d(288, 3, 2)  

        #branch4
        self.b4_1 = nn.LazyConv2d(256, 1)
        self.b4_2 = nn.LazyConv2d(288, 3, 1, 1)
        self.b4_3 = nn.LazyConv2d(320, 3, 2)

        #Filter Concat
        self.filter_concat = nn.LazyConv2d(1792, 1)

    def forward(self, x):
        x1 = self.b1(x)
        x2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        x3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        x4 = F.relu(self.b4_3(F.relu(self.b4_2(F.relu(self.b4_1(x))))))
        x = F.relu(self.filter_concat(torch.cat((x1,x2,x3,x4),dim=1)))
        return x



class inception_resnet_v1(nn.Module):
    def __init__(self, n_incres_a=5, n_incres_b=10, n_incres_c=5):
        super(inception_resnet_v1, self).__init__()

        self.inceptionResNetA = []
        for _ in range(n_incres_a):
            self.inceptionResNetA.append(inception_resnet_a())
        self.inceptionResNetA = nn.Sequential(*self.inceptionResNetA)


        self.inceptionResNetB = []
        for _ in range(n_incres_b):
            self.inceptionResNetB.append(inception_resnet_b())
        self.inceptionResNetB = nn.Sequential(*self.inceptionResNetB)

        self.inceptionResNetC = []
        for _ in range(n_incres_c):
            self.inceptionResNetC.append(inception_resnet_c())
        self.inceptionResNetC = nn.Sequential(*self.inceptionResNetC)

        self.stem = stem()

        self.reductionA = reduction_module_a()
        self.reductionB = reduction_module_b()
        
        self.avg_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        self.dropout = nn.Dropout(0.8)
        self.classifier = nn.LazyLinear(1000)


    def forward(self, x):
        x = self.stem(x)
        x = self.inceptionResNetA(x)
        x = self.reductionA(x)
        x = self.inceptionResNetB(x)
        x = self.reductionB(x)
        x = self.inceptionResNetC(x)
        x = self.avg_pooling(x)
        x = self.dropout(x)
        x = F.softmax(self.classifier(x), dim=, dim=1)
        return x
        




if __name__ == '__main__':
    print('Hello InceptionResnet-V1')
    x = torch.rand([1, 3, 299, 299])
    y = inception_resnet_v1()(x)
    print(y.shape)
