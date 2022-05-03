import torch
import torch.nn as nn
from collections import OrderedDict

class ConvRELU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvRELU, self).__init__()
        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), 
            #nn.ReLU(inplace=True)
            RBF(1)
        )
    
    def forward(self, x):
        return self.convrelu(x)

class Normalisation_Module_t1(nn.Module):
    def __init__(self, in_channels, features=[16, 16, 1]): #16 #16 #1
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

        ]))

    def forward(self, x):
        z = x
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return x + z

class Normalisation_Module_t1ce(nn.Module):
    def __init__(self, in_channels, features=[16, 16, 1]):
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
        ]))

    def forward(self, x):
        z = x
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return x + z

        


class Normalisation_Module_t2(nn.Module):
    def __init__(self, in_channels, features=[16, 16, 1]):
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
        ]))

    def forward(self, x):
        z = x
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return x + z


class Normalisation_Module_flair(nn.Module):
    def __init__(self, in_channels, features=[16, 16, 1]):
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
        ]))

    def forward(self, x):
        z = x
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return x + z


class RBF(nn.Module):
    r""" Applies function

        $$ y = \exp(-0.5 \cdot \beta \cdot x^2) $$
    """

    def __init__(self, beta: float = 1.0) -> None:
        super(RBF, self).__init__()
        self.beta = nn.Parameter(torch.FloatTensor([beta]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.beta * input**2)




