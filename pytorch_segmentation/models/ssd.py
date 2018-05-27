'''
Sigle Shot Multibox Detection Architecture
adapted from @amdegroot's implementation of
Wei Liu, et al. "SSD: Single Shot MultiBox Detector." ECCV2016.

His code in turnn was ported from @weiliu89's caffe implementation.
'''
# import os
# pylint: disable=import-error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from _ssd.layers import (
    PriorBox,
    L2Norm,
    Detect)
from _ssd.data import v2
from .vgg import VGGNet
from .drn import drn_d_107, drn_d_56

# pylint: disable=invalid-name,too-many-instance-attributes,too-many-arguments
# pylint: disable=arguments-differ

# configurations used for vgg
# v2 = {
#     'feature_maps': [38, 19, 10, 5, 3, 1],
#     'min_dim': 300,
#     'steps': [8, 16, 32, 64, 100, 300],
#     'min_sizes': [30, 60, 111, 162, 213, 264],
#     'max_sizes': [60, 111, 162, 213, 264, 315],
#     # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
#     #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
#     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'v2',
# }


class FeatExtr(nn.Module):
    '''Wrapper class for a network
        TODO (William):
    Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    '''
    def __init__(self, net, scale=10, num_classes=21, phase='train'):
        super(FeatExtr, self).__init__()
        # vgg has an outstride of 16 while drn has an outstride of 8
        if net == 'vgg':
            self.net = VGGNet(out_indices=[23])
            self.steps = [8, 16, 32, 64, 100, 300]
            self.min_sizes = [30, 60, 111, 162, 213, 264]
            self.max_sizes = [60, 111, 162, 213, 264, 315]

        elif net == 'drn_107':
            # drn models have an output stride of 8, so it is unnecessary
            # to extract features from early layers if we wish to
            # replecate the VGG behavior
            self.net = drn_d_107(pretrained=True,
                                 out_middle=True,
                                 out_indices=[2])
            self.steps = [4, 8, 16, 32, 64, 100, 300]
            self.min_sizes = [15, 30, 60, 111, 162, 213, 264]
            self.max_sizes = [30, 60, 111, 162, 213, 264, 315]

        elif net == 'drn_56':
            self.net = drn_d_56(pretrained=True,
                                out_middle=True,
                                out_indices=[2])
            self.steps = [4, 8, 16, 32, 64, 100, 300]
            self.min_sizes = [15, 30, 60, 111, 162, 213, 264]
            self.max_sizes = [30, 60, 111, 162, 213, 264, 315]
        else:
            raise NotImplementedError(
                "{} not (yet) a valid network type".format(net))

        self.net_name = net
        self.phase = phase
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.variance = [0.1, 0.2]

        net_ = self.net
        channels = net_.channels
        indices = net_.indices
        self.L2Norms = nn.ModuleList(
            [L2Norm(channels[i], scale) for i in indices])
        self.indices = indices

        self._add_extras()
        self._multibox()

        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        self.softmax = nn.Softmax(dim=-1)
        # num_classes, bkg_label, top_k, conf_thresh, nms_thresh
        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def _add_extras(self):
        ''' Add extra layers to refine features
        '''
        if self.net_name == 'vgg':
            cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
        elif self.net_name == 'drn_107':
            cfg = [256, 'S', 256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
        elif self.net_name == 'drn_38':
            cfg = [256, 'S', 256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
        else:
            raise RuntimeError("Invalid network type")

        # Extra layers added to VGG for feature scaling
        layers = []
        in_channels = self.feature_channels
        flag = False
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'S':
                    layers += [nn.Conv2d(
                        in_channels, cfg[k + 1],
                        kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(
                        in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        self.extra_layers = layers

    def _multibox(self):
        '''Wrapper to add the multibox predictions to a
        net.
        '''
        if self.net_name == 'vgg':
            # [8, 16, 32, 64, 100, 300]
            cfg = [4, 6, 6, 6, 4, 4]
        elif self.net_name == 'drn_107':
            cfg = [4 for _ in self.indices]
            cfg.extend([4, 4, 6, 6, 6, 4, 4])
        elif self.net_name == 'drn_38':
            cfg = [4 for _ in self.indices]
            cfg.extend([4, 4, 6, 6, 6, 4, 4])
        else:
            raise RuntimeError("Invalid network type")

        extra_layers = self.extra_layers
        num_classes = self.num_classes
        net = self.net
        loc_layers = []
        conf_layers = []
        net_source = [net.channels[i] for i in net.indices]
        for k, v in enumerate(net_source):
            loc_layers += [nn.Conv2d(v, cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v, cfg[k] * num_classes,
                                      kernel_size=3,
                                      padding=1)]

        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]

        # Location and confidence layers
        self.loc = loc_layers
        self.conf = conf_layers

    def forward(self, x):

        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        y, outs = self.net(x)
        outs = [outs[i] for i in self.indices]
        scaled_outs = [l(o) for l, o in zip(self.L2Norms, outs)]
        sources.append(y)
        sources.extend(scaled_outs)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extra_layers):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (tens, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(tens).permute(0, 2, 3, 1).contiguous())
            conf.append(c(tens).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.training is False:
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(
                    conf.view(conf.size(0), -1,
                              self.num_classes)),               # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output
