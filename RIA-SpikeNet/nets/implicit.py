import torch
import torch.nn as nn


class ImplicitA(nn.Module):
    def __init__(self, out_channels):
        super(ImplicitA, self).__init__()
        self.channel = out_channels
        self.implicit = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        nn.init.normal_(self.implicit, std=.02)
       
        
    def forward(self):
        return self.implicit


class ImplicitC(nn.Module):
    def __init__(self, out_channels):
        super(ImplicitC, self).__init__()
        self.channel = out_channels
        self.implicit = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit


class ImplicitM(nn.Module):
    def __init__(self, out_channels):
        super(ImplicitM, self).__init__()
        self.channel = out_channels
        self.implicit = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)

    def forward(self):
        return self.implicit
    


class Implicit2DA(nn.Module):
    def __init__(self, atom, out_channels):
        super(Implicit2DA, self).__init__()
        self.channel = out_channels
        self.implicit = nn.Parameter(torch.zeros(1, atom, out_channels, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit


class Implicit2DC(nn.Module):
    def __init__(self, atom, out_channels):
        super(Implicit2DC, self).__init__()
        self.channel = out_channels
        atom=out_channels
        self.implicit = nn.Parameter(torch.zeros(1, atom, out_channels, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit


class Implicit2DM(nn.Module):
    def __init__(self, atom, out_channels):
        super(Implicit2DM, self).__init__()
        self.channel = out_channels
        atom=out_channels
        self.implicit = nn.Parameter(torch.ones(1, atom, out_channels, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)

    def forward(self):
        return self.implicit


class ScaleChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(ScaleChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return x.expand_as(a) * a


class ShiftChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, out_channels):
        super(ShiftChannel, self).__init__()
          # layer indices
        self.channel=out_channels
        self.implicit=ImplicitA(out_channels)

    def forward(self, outputs):
        x=self.implicit()
        a = outputs
        return x.expand_as(a) + a

class ShiftChannel2D(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, out_channels):
        super(ShiftChannel2D, self).__init__()
        self.channel=out_channels
        self.implicit=Implicit2DA(atom=out_channels,out_channels=out_channels)


    def forward(self, outputs):
        x = self.implicit().view(1,-1,1,1)
        x=x.view(1,-1,1,1)
        a=outputs
        return x.expand_as(a) + a



class ControlChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, out_channels):
        super(ControlChannel, self).__init__()
        self.channel=out_channels
        self.implicit=ImplicitM(out_channels)

    def forward(self, outputs):
        
        x=self.implicit()
        a = outputs
        return x.expand_as(a) * a


class ControlChannel2D(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, out_channels):
        super(ControlChannel2D, self).__init__()
        self.channel=out_channels
        self.implicit=Implicit2DM(atom=out_channels,out_channels=out_channels)

    def forward(self, outputs):
        x = self.implicit.view(1,-1,1,1)
        a=outputs
        return x.expand_as(a) * a


class AlternateChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, out_channels):
        super(AlternateChannel, self).__init__()
        self.channel=out_channels
        self.implicit=ImplicitC(out_channels)

    def forward(self, outputs):
        
        x=self.implicit()
        a = outputs
        return torch.cat([x.expand_as(a), a], dim=1)


class AlternateChannel2D(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, out_channels):
        super(AlternateChannel2D, self).__init__()
        self.channel=out_channels
        self.implicit=Implicit2DC(atom=out_channels,out_channels=out_channels)

    def forward(self, outputs):
        x = self.implicit.view(1,-1,1,1)
        a=outputs
        return torch.cat([x.expand_as(a), a], dim=1)


class SelectChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(SelectChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.sigmoid().expand_as(x) * x


class SelectChannel2D(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(SelectChannel2D, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]].view(1,-1,1,1)
        return a.sigmoid().expand_as(x) * x


class ScaleSpatial(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(ScaleSpatial, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return x * a