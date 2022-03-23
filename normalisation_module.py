import torch
import torch.nn as nn
from collections import OrderedDict

class ConvRELU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvRELU, self).__init__()
        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), 
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.convrelu(x)

class Normalisation_Module_t1(nn.Module):
    def __init__(self, in_channels, features=[32, 64, 32, 1]): #16 #16 #16 #1
        super(Normalisation_Module_t1, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit2', ConvRELU(in_channels, features[0]))] 
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit3', ConvRELU(features[0], features[1]))] 
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit4', ConvRELU(features[1], features[2]))] 
                ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit5', ConvRELU(features[2], features[3]))] 
                ))),
        ]))

    def forward(self, x):
        z = x
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return x + z

class Normalisation_Module_t1ce(nn.Module):
    def __init__(self, in_channels, features=[32, 64, 32, 1]):
        super(Normalisation_Module_t1ce, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit2', ConvRELU(in_channels, features[0]))] 
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit3', ConvRELU(features[0], features[1]))] 
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit4', ConvRELU(features[1], features[2]))] 
                ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit5', ConvRELU(features[2], features[3]))] 
                ))),
        ]))

    def forward(self, x):
        z = x
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return x + z

        


class Normalisation_Module_t2(nn.Module):
    def __init__(self, in_channels, features=[32, 64, 32, 1]):
        super(Normalisation_Module_t2, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit2', ConvRELU(in_channels, features[0]))] 
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit3', ConvRELU(features[0], features[1]))] 
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit4', ConvRELU(features[1], features[2]))] 
                ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit5', ConvRELU(features[2], features[3]))] 
                ))),
        ]))

    def forward(self, x):
        z = x
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return x + z


class Normalisation_Module_flair(nn.Module):
    def __init__(self, in_channels, features=[32, 64, 32, 1]):
        super(Normalisation_Module_flair, self).__init__()

        self.layers = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit2', ConvRELU(in_channels, features[0]))] 
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit3', ConvRELU(features[0], features[1]))] 
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit4', ConvRELU(features[1], features[2]))] 
                ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit5', ConvRELU(features[2], features[3]))] 
                ))),
        ]))

    def forward(self, x):
        z = x
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return x + z