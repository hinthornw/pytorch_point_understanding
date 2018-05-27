'''
Author: William Hinthorn
'''
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import mul, matmul
# pylint: disable=invalid-name, arguments-differ


class GatedLayer(nn.Module):
    ''' Implements a gated
    '''
    def __init__(self, indims, outdims, rpn_act='relu'):
        super(GatedLayer, self).__init__()
        self.gate_conv = nn.Conv2d(indims, outdims, 3, padding=1)
        self.signal_conv = nn.Conv2d(indims, outdims, 3, padding=1)
        if rpn_act == 'relu':
            signal_act = nn.ReLU(inplace=True)
        elif rpn_act == 'tanh':
            signal_act = nn.Tanh()
        else:
            raise ValueError("{} not a valid activation functino".format(
                rpn_act))
        n = 3. * 3. * outdims
        self.signal_act = signal_act
        self.gate_conv.weight.data.normal_(0.0, math.sqrt(2. / n))
        # Initially let all information through
        self.gate_conv.bias.data.fill_(1.)
        self.signal_conv.weight.data.normal_(0.0, math.sqrt(2. / n))
        self.signal_conv.bias.data.fill_(0.)

    def forward(self, x):
        """ y = f(W * x + b) dot g(W * x + b)
        """
        y_prime = self.signal_act(self.signal_conv(x))
        g = F.sigmoid(self.gate_conv(x))
        y = mul(y_prime, g)
        return y


class NonLocalBlock(nn.Module):
    ''' Implements a gated
    '''
    def __init__(self, indims):
        super(NonLocalBlock, self).__init__()

        bottle_channels = indims // 2
        self.g = nn.Conv2d(indims, bottle_channels, 1)
        self.phi = nn.Conv2d(indims, bottle_channels, 1)
        self.theta = nn.Conv2d(indims, bottle_channels, 1)
        self.out = nn.Conv2d(bottle_channels, indims, 1)
        # Random Start
        n = 3. * 3. * bottle_channels
        self.g.weight.data.normal_(0.0, math.sqrt(2. / n))
        self.phi.weight.data.normal_(0.0, math.sqrt(2. / n))
        self.theta.weight.data.normal_(0.0, math.sqrt(2. / n))
        self.out.weight.data.normal_(0.0, math.sqrt(2. / n))
        self.bottle_channels = bottle_channels

    def forward(self, x):
        """ y = f(W * x + b) dot g(W * x + b)
                f(xi,xj) = e^(theta(xi).T phi(xj)).
                NL(x) = 1/C(x) sum[f(xi, xj) g(xj)]
        """
        b, _, h, w = x.size()
        theta_out = self.theta(x).view(
            b, self.bottle_channels, -1).permute(0, 2, 1).contiguous()
        phi_out = self.phi(x).view(
            b, self.bottle_channels, -1)

        f = matmul(phi_out, theta_out)
        sm_out = F.softmax(f, -1)
        g_out = self.g(x).view(
            b, self.bottle_channels, -1).permute(0, 2, 1).contiguous()
        block_out = matmul(g_out, sm_out)
        block_out = block_out.permute(0, 2, 1).contiguous()
        block_out = block_out.view(b, self.bottle_channels, h, w)
        y = self.out(block_out)
        return y + x
