'''
Author: William Hinthorn
'''
import torch.nn as nn
import torch.nn.init as init


class L2Norm(nn.Module):
    ''' Take the channelwise L2 norm, learning a scaling
    parameter stored as a 1x1 channelwise 2d convolution
    '''
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        # self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.scale = nn.Conv2d(n_channels, n_channels, 1, groups=n_channels)
        self.reset_parameters()
        self.eps = 1e-12

    def reset_parameters(self):
        ''' Initialize parameters as gamma
        '''
        init.constant(self.scale.weight, self.gamma)

    def forward(self, x):
        x = nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
        # ^^ is faster on GPU vv is faster on cpu
        # x = torch.div(x, x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps)
        out = self.scale(x)
        return out
