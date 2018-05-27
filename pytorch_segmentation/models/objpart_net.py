'''
author: William Hinthorn (whinthorn | at | gmail)

'''
import torch
import torch.nn as nn
from .resnet import resnet34
# from .upsampling import Upsampling
# from .upsampling_simplified import Upsampling
from .upsampling_simplified import UpsamplingBilinearlySpoof
from .drn import drn_d_107, drn_d_54
from .vgg import VGGNet
from .layers import size_splits

# pylint: disable=invalid-name, arguments-differ,too-many-locals
# pylint: disable=too-many-instance-attributes,too-many-statements


def _bake_function(f, **kwargs):
    import functools
    return functools.partial(f, **kwargs)


def _normal_initialization(layer):
    layer.bias.data.zero_()


class OPSegNet(nn.Module):
    ''' Unified network that allows for various
    feature extraction modules to be swapped in and out.
    '''
    def __init__(self,
                 arch,
                 output_dims,
                 upsample='bilinear',
                 pretrained=True):

        print(arch)
        assert arch in ['drn', 'drn_54', 'resnet', 'vgg', 'hourglass']
        super(OPSegNet, self).__init__()

        # one dimension for each object and part (plus background)
        # to do: in order to train on other datasets, will need to
        # push the bg class to the point file
        mult = 1  # 5 if bbox
        self.mult = mult
        num_classes = 1 + sum(
            [1*mult + output_dims[k]*mult for k in output_dims])

        if arch == 'resnet':
            step_size = 8
            outplanes = 512
            # This comment block refers to the parameters for the
            # depracated upsampling unit.
            # Net will output y -> list of outputs from important blocks
            # feature_ind denotes which output to concatenate with upsampled
            # 0: conv1
            # 1: layer1
            # Index the feature vector from the base network
            # feature_ind = [0, 1]
            # Give the widths of each feature tensor (i.e dim 2 length)
            # Order is from smallest spatial resolution to largest
            # feature_widths = [64, 64]
            # add skip connections AFTER which upsamples
            # 0 here means BEFORE the first
            # merge_which_skips = set([1, 2])
            # mergedat = {0: (64, 2), 1: (64, 4)}

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
            # feature_ind = [3, 8, 15, 22]
            # feature_widths = [512, 256, 128, 64]
            # merge_which_skips = set([1, 2, 3, 4])
            # upsampling_channels = [1024, 256, 128, 64, 32]
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
            # feature_ind = [0, 1, 2]
            # feature_widths = [256, 32, 16]
            # merge_which_skips = set([1, 2, 3])
            # feature_ind = [1, 2]
            # feature_widths = [256, 32]
            # merge_which_skips = set([1, 2])
            # upsampling_channels = [512, 128, 64, 32]
            # mergedat = {2: (256, 4), 1: (32, 2)}
            net = drn_d_107(pretrained=pretrained, out_middle=True)

        elif arch == 'drn_54':
            step_size = 8
            outplanes = 512
            net = drn_d_54(pretrained=pretrained, out_middle=True)

        # self.inplanes = # num_classes # upsampling_channels[-1]
        # head_outchannels = outplanes // step_size
        # skip_outchannels = [mergedat[k][0] // mergedat[k][1]
        # for k in mergedat]
        # merge_inchannels = head_outchannels + sum(skip_outchannels)
        # self.inplanes = merge_inchannels
        self.inplanes = outplanes
        self.net = net

        self.fc = nn.Conv2d(self.inplanes, num_classes, 1)

        # Randomly initialize the 1x1 Conv scoring layer
        _normal_initialization(self.fc)

        self.num_classes = num_classes
        output_dims = {int(k): v for k, v in output_dims.items()}
        self.output_dims = output_dims
        # 1 for background, 1 for each object class
        # self.split_tensor = [1, len(self.output_dims)]
        split_tensor = [1]
        # maps each object prediction to its part(s) channels
        op_map = {}
        # Plus the number of part classes present for each cat
        i = 1
        for k in sorted(self.output_dims):
            split_tensor.append(mult)
            i += 1
        j = 1
        for k in sorted(self.output_dims):
            v = self.output_dims[k]
            if v > 0:
                split_tensor.append(v*mult)
                op_map[j] = i
                i += 1
            else:
                op_map[j] = -1
            j += 1

        self.op_map = op_map
        self.flat_map = {}
        for k, v in op_map.items():
            if v > 0:
                self.flat_map[k] = v
                self.flat_map[v] = k
        self.split_tensor = split_tensor

        # print("op map")
        # for k in sorted(op_map):
        #     print("{}:\t{}".format(k, op_map[k]))

        # print("flat map")
        # for k in sorted(self.flat_map):
        #     print("{}:\t{}".format(k, self.flat_map[k]))

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
        insize = x.size()[-2:]
        x, _ = self.net(x)    # extract features
        # x = self.decode((x, y, insize))  # decode/upsample
        # objpart_logits = self.fc(x)        # classify
        x = self.fc(x)
        objpart_logits = self.decode([x], insize)
        # Add object and part channels to predict a semantic segmentation
        splits = size_splits(objpart_logits, self.split_tensor, 1)
        # bg, objects, parts = splits[0], splits[1], splits[2:]

        bg, other_data = splits[0], splits[1:]
        op_data = [torch.sum(part, dim=1, keepdim=True)
                   for part in other_data]
        # the (-1) is since we separate out bg above
        out = []
        for o in sorted(self.op_map):
            to_add1 = op_data[o-1]
            p = self.op_map[o]
            if p > 0:
                to_add2 = op_data[p-1]
                out.append(to_add1 + to_add2)
            else:
                out.append(to_add1)

        # out = [op_data[o-1] + op_data[p-1] if p > 0 else
        #        op_data[o-1] for o, p in self.op_map.items()]
        out = torch.cat(out, dim=1)
        # parts = [torch.sum(part, dim=1, keepdim=True) for part in parts]
        # parts = torch.cat(parts, dim=1)
        # out = objects + parts
        semantic_seg_logits = torch.cat([bg, out], dim=1)

        return objpart_logits, semantic_seg_logits
