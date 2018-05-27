'''
author: William Hinthorn (whinthorn | at | gmail)

'''
import torch
import torch.nn as nn
from .resnet import resnet34
# from .upsampling import Upsampling
from .upsampling_simplified import UpsamplingBilinearlySpoof
from .drn import drn_d_107
from .vgg import VGGNet
from .layers import size_splits

# pylint: disable=invalid-name, arguments-differ


def _bake_function(f, **kwargs):
    import functools
    return functools.partial(f, **kwargs)


def _normal_initialization(layer):
    layer.bias.data.zero_()


class OPExtentNet(nn.Module):
    ''' Unified network that allows for various
    feature extraction modules to be swapped in and out.
    '''
    def __init__(self,
                 arch,
                 output_dims,
                 pretrained=True):

        print(arch)
        assert arch in ['drn', 'resnet', 'vgg', 'hourglass']
        super(OPExtentNet, self).__init__()

        # one dimension for each object and part (plus background)
        # TODO: in order to train on other datasets, will need to
        # push the bg class to the point file
        num_classes = 1 + sum([1 + output_dims[k] for k in output_dims])

        if arch == 'resnet':
            step_size = 8
            outplanes = 512
            # Number of channels at each stage of the decoder
            # upsampling_channels = [512, 128, 64, 32]
            net = resnet34(fully_conv=True,
                           pretrained=pretrained,
                           output_stride=step_size,
                           out_middle=True,
                           remove_avg_pool_layer=True)

        elif arch == 'vgg':
            step_size = 16
            outplanes = 1024
            # mergedat = {15: (512, 8), 8: (256, 4), 1: (128, 2)}
            net = VGGNet()
            raise NotImplementedError(
                "VGGNet architecture not yet debugged")

        elif arch == 'hourglass':
            # step_size = ???
            raise NotImplementedError(
                "Hourglass network architecture not yet implemented")

        elif arch == 'drn':
            step_size = 8
            outplanes = 512
            net = drn_d_107(pretrained=pretrained, out_middle=True)

        self.inplanes = outplanes
        self.net = net

        self.fc = nn.Conv2d(self.inplanes, num_classes, 1)

        # Randomly initialize the 1x1 Conv scoring layer
        _normal_initialization(self.fc)

        self.num_classes = num_classes
        self.output_dims = output_dims
        self.split_tensor = [1, len(self.output_dims)]
        self.split_tensor.extend([v for v in self.output_dims.values()])

        upsample = UpsamplingBilinearlySpoof(step_size)
        self.decode = upsample

    def forward(self, x):
        '''
        returns predictions for the object-part segmentation task and
        the semantic segmentation task
            Format:
            x = [background, obj_1, obj_2, ..., parts_1, parts_2, ...]
            out = [background, obj_or_part_1, obj_or_part_2, , ...]
        '''
        # insize = x.size()[-2:]
        x, _ = self.net(x)    # extract features
        # x = self.decode((x, y, insize))  # decode/upsample
        # objpart_logits = self.fc(x)        # classify
        x = self.fc(x)
        objpart_logits = self.decode([x])

        # Add object and part channels to predict a semantic segmentation
        splits = size_splits(objpart_logits, self.split_tensor, 1)
        bg, objects, parts = splits[0], splits[1], splits[2:]
        parts = [torch.sum(part, dim=1, keepdim=True) for part in parts]
        parts = torch.cat(parts, dim=1)
        out = objects + parts
        semantic_seg_logits = torch.cat([bg, out], dim=1)

        return objpart_logits, semantic_seg_logits
