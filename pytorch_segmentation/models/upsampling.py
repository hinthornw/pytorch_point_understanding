''' Flexible upsampling and merging module that allows for the use
of transposed convolutions or bilinear upsampling. Allows for the
optional inclusion of skip layers as in the original FCN.
'''
import torch.nn as nn

# pylint: disable=too-many-arguments, too-many-locals, unidiomatic-typecheck,
# pylint: disable=invalid-name,arguments-differ


class Upsampling(nn.Module):
    ''' Defines a decoder.
    '''
    def __init__(
            self,
            stride,
            inchannels,
            channels,
            mode,
            feature_ind,
            merge_which_skips):
        '''
            args:
                :parameter ``stride``: the outstride of the network or total
                downsampling rate
                :parameter ``inchannels``: list() of ints or None's which
                state the width of the coarse features to be added to the pred
                :parameter ``channels``: list() or int of
        '''
        super(Upsampling, self).__init__()
        import math
        num_up = math.log(stride, 2)
        assert num_up % 1 == 0, 'stride must be a power of 2'
        assert len(inchannels) == len(merge_which_skips)
        assert type(channels) in [int, list]

        num_up = int(num_up)
        if isinstance(channels, int):
            channels = [channels for _ in range(num_up)]
        if mode == 'bilinear':
            # We don't currently support changes in width for the output
            # features using bilinear interpolation
            for i, elem in enumerate(channels[1:]):
                assert channels[i - 1] == elem

        # self.mode = mode
        # self.stride = stride
        merging = len(merge_which_skips) > 0
        layers = []
        merge_conv = []
        _inchannels = [el for el in inchannels]
        _inchannels.reverse()
        channels_in = channels[0]
        for i in range(num_up):
            # Each layer doubles spatial resolution
            if mode == 'transposed':
                layers.append(nn.ConvTranspose2d(
                    channels_in, channels[i + 1],
                    3, stride=2, padding=1,
                    output_padding=1))

            elif merging and mode == 'bilinear':
                layers.append(
                    nn.Upsample(
                        scale_factor=2,
                        mode='bilinear'))

            if i in merge_which_skips:
                merge_conv.append(
                    nn.Conv2d(
                        _inchannels.pop(),
                        channels[i],
                        1,
                        padding=0))
            else:
                merge_conv.append(None)
            channels_in = channels[i + 1]

        if num_up in merge_which_skips:
            merge_conv.append(
                nn.Conv2d(
                    _inchannels.pop(),
                    channels[num_up],
                    1,
                    padding=0))
        else:
            merge_conv.append(None)

        if not merging and mode == 'bilinear':
            layers = [
                nn.Upsample(
                    scale_factor=stride,
                    mode='bilinear')]

        self.layers = nn.ModuleList(layers)
        self.merge_conv = nn.ModuleList(merge_conv)
        self.feature_ind = [ind for ind in feature_ind]
        self.merge_features = [None for _ in range(num_up+1)]
        self.merge_which_skips = list(merge_which_skips)
        self.merge_which_skips.sort()
        self.feature_ind.sort(reverse=True)

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        assert len(x) == 3
        x, low, outsize = x
        for layer, feat in zip(self.merge_which_skips, self.feature_ind):
            self.merge_features[layer] = low[feat]

        ul_len = len(self.layers)
        # Merge after upsampling
        for i, layer in enumerate(self.layers):
            if self.merge_conv[i] is not None:
                # low_x = self.merge_conv[i](low[i])
                low_x = self.merge_conv[i](self.merge_features[i])
                x = x + low_x

            if i + 1 < ul_len and self.merge_conv[i + 1] is not None:
                sz = self.merge_features[i + 1].size()[-2:]
                x = layer(x, output_size=sz)
            elif i + 1 == ul_len:
                x = layer(x, output_size=outsize)
            else:
                x = layer(x)

        # Merge at original spatial res
        i = len(self.layers)
        if self.merge_features[i] is not None:
            low_x = self.merge_conv[i](self.merge_features[i])
            x = low_x + x

        return x
