'''
Sigle Shot Multibox Detection Architecture
adapted from @amdegroot's implementation of
Wei Liu, et al. "SSD: Single Shot MultiBox Detector." ECCV2016.

His code in turnn was ported from @weiliu89's caffe implementation.
'''
# import os
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from .activities import GatedLayer, NonLocalBlock
from .upsampling_simplified import BilinearUpsample
# from .layers import size_splits
from .vgg import VGGNet
from .drn import drn_d_107, drn_d_54, drn_d_38
# import pdb
# from ..extent_utils import tensor2flat, arr2im, tensor2im


# pylint: disable=invalid-name,too-many-instance-attributes,too-many-arguments
# pylint: disable=arguments-differ,too-many-locals,too-many-statements,
# pylint: disable=too-many-branches


class ExtentNet(nn.Module):
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
    def __init__(self, net, output_dims, style):
        super(ExtentNet, self).__init__()
        # vgg has an outstride of 16 while drn has an outstride of 8
        if net == 'vgg':
            outplanes = 1024  # check...
            outstride = 16
            self.net = VGGNet(out_indices=[])

        elif net == 'drn_107':
            # drn models have an output stride of 8, so it is unnecessary
            # to extract features from early layers if we wish to
            # replecate the VGG behavior
            outstride = 8
            outplanes = 512
            self.net = drn_d_107(pretrained=True,
                                 out_middle=True,
                                 out_indices=[])

        elif net == 'drn_54':
            outstride = 8
            outplanes = 512
            self.net = drn_d_54(pretrained=False,
                                out_middle=True,
                                out_indices=[])

        elif net == 'drn_38':
            outstride = 8
            outplanes = 512
            self.net = drn_d_38(pretrained=False,
                                out_middle=True,
                                out_indices=[])
        else:
            raise NotImplementedError(
                "{} not (yet) a valid network type".format(net))

        # FREEZE GRAD (trying this one out now 04/15/2018
        # print("Freezing base net layers")
        # for param in self.net.parameters():
        #     param.requires_grad = False
        output_dims = {int(k): v for k, v in output_dims.items()}
        self.net_name = net
        self.output_dims = output_dims
        self.inplanes = outplanes

        op2sem = [[0]]
        _pmapt = sorted(list(output_dims))
        _pmapt.pop(17)
        _pmapt.insert(10, 397)
        obj_ids = _pmapt
        j = 0
        N = len(output_dims) + 1  # w/ bg
        # for i, k in enumerate(sorted(output_dims), 1):
        for i, k in enumerate(obj_ids, 1):
            v = output_dims[k]
            inds = [i]
            if v > 0:
                inds.extend(list(range(N+j, N+j+v)))
                op2sem.append(inds)
                j += 1
            else:
                op2sem.append([i])
        self.op2sem = op2sem

        self.op_2compress = {}
        for vals in op2sem:
            if len(vals) > 1:
                self.op_2compress[vals[0]] = vals[1]
                self.op_2compress[vals[1]] = vals[0]

        # hacky way to decide on output dimensions
        breakup = [1, 4]
        split_tensor = [1]
        # maps each object prediction to its part(s) channels
        op_map = {}
        # Plus the number of part classes present for each cat
        i = 1
        for k in sorted(self.output_dims):
            split_tensor.extend(breakup)
            i += sum(breakup)

        j = 1
        for k in sorted(self.output_dims):
            v = self.output_dims[k]
            if v > 0:
                split_tensor.extend(v*breakup)
                op_map[j] = i
                i += 1
            else:
                op_map[j] = -1
            j += 1

        self.op_map = op_map

        self.split_tensor = torch.LongTensor(split_tensor)
        # - 1 for bg class
        cumsum = torch.cumsum(self.split_tensor, 0) - 1
        self.op_pred_ind = Variable(cumsum[
            self.split_tensor.eq(1).nonzero().view(-1)].cuda())
        bbox_inds = cumsum[
            self.split_tensor.eq(4).nonzero().view(-1)-1]+1
        bbox_inds = [range(v, v+4) for v in bbox_inds]
        bbox_inds = torch.LongTensor(bbox_inds).view(-1).cuda()
        self.obj_bbox_inds = Variable(bbox_inds[:4 * (N - 1)].contiguous())
        self.part_bbox_inds = Variable(bbox_inds[4 * (N - 1):].contiguous())

        # num_classes = sum(self.split_tensor)
        num_classes = sum([len(v) for v in self.op2sem])
        # Need to figure out dims to add a 1x1 conv for all branches
        fc1 = nn.Conv2d(self.inplanes, self.inplanes//2, 3, padding=1)
        fc2 = nn.Conv2d(self.inplanes//2, num_classes, 1, bias=True)
        self.fc = nn.Sequential(
            fc1,
            nn.BatchNorm2d(self.inplanes//2),
            nn.ReLU(inplace=True),
            fc2)
        objout = len(self.obj_bbox_inds)
        partout = len(self.part_bbox_inds)

        # 1-layer, ReLU activations - choose
        if style == '1relu':
            self.style = 'relu'
            context = nn.Conv2d(
                self.inplanes,
                self.inplanes//2, 3, padding=1)
            fc_bb_obj = nn.Conv2d(
                self.inplanes//2,
                objout, 1, bias=True)
            fc_bb_part = nn.Conv2d(
                self.inplanes//2,
                partout, 1, bias=True)
            self.rpn = nn.Sequential(
                context,
                nn.BatchNorm2d(self.inplanes//2),
                nn.ReLU(inplace=True))

            # normal initialize
            for m in [fc1, fc2, fc_bb_obj, fc_bb_part, context]:
                # context1, context2]:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif style == '2relu':
            # 2-layer, ReLU activations - choose
            self.style = 'relu'
            context1 = nn.Conv2d(
                self.inplanes,
                self.inplanes, 3, padding=1)
            context2 = nn.Conv2d(
                self.inplanes,
                self.inplanes, 3, padding=1)
            fc_bb_obj = nn.Conv2d(
                self.inplanes,
                objout, 1, bias=True)
            fc_bb_part = nn.Conv2d(
                self.inplanes,
                partout, 1, bias=True)
            self.rpn = nn.Sequential(
                context1,
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                context2,
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True))
            for m in [fc1, fc2, fc_bb_obj, fc_bb_part, context1, context2]:
                # context1, context2]:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # Gated TanH  - Choose
        elif 'gated' in style:
            self.style = 'gated_tanh'
            fc_bb_obj = nn.Conv2d(
                self.inplanes,
                objout, 1, bias=True)
            fc_bb_part = nn.Conv2d(
                self.inplanes,
                partout, 1, bias=True)
            for m in [fc1, fc2, fc_bb_obj, fc_bb_part]:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if '2' in style:
                # 2 layers - Choose
                context1 = GatedLayer(
                    self.inplanes,
                    self.inplanes, 'tanh')
                context2 = GatedLayer(
                    self.inplanes,
                    self.inplanes, 'tanh')
                self.rpn = nn.Sequential(context1, context2)
            elif '1' in style:
                # 1-layer - Choose
                context1 = GatedLayer(
                    self.inplanes,
                    self.inplanes, 'tanh')
                self.rpn = context1
            else:
                raise ValueError('{} not recognized'.format(style))
        elif style == 'nonlocal':
            self.style = 'nonlocal'
            fc_bb_obj = nn.Conv2d(
                self.inplanes,
                objout, 1, bias=True)
            fc_bb_part = nn.Conv2d(
                self.inplanes,
                partout, 1, bias=True)
            for m in [fc1, fc2, fc_bb_obj, fc_bb_part]:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            context1 = NonLocalBlock(
                self.inplanes)
            context2 = nn.Conv2d(
                self.inplanes,
                self.inplanes, 3, padding=1)
            self.rpn = nn.Sequential(context1, context2)

        elif 'attention' in style:
            self.style = 'attention'
            fc_bb_obj = nn.Conv2d(
                self.inplanes,
                objout, 1, bias=True)
            fc_bb_part = nn.Conv2d(
                self.inplanes,
                partout, 1, bias=True)
            expand = nn.Conv2d(self.inplanes, 1008, 1, bias=False)
            self.expand = expand
            # context1 = NonLocalBlock(
            #     self.inplanes)
            context1 = NonLocalBlock(
                1008)
            context2 = nn.Conv2d(
                1008,
                self.inplanes, 3, padding=1)
            self.rpn = nn.Sequential(context1, context2)
            for m in [fc1, fc2, fc_bb_obj, fc_bb_part, context2, expand]:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            # raise NotImplementedError(
            #     '{} style not yet implemented'.format(style))
        else:
            raise ValueError('{} not recognized'.format(style))

        # Keep for all
        self.fc_bb_obj = fc_bb_obj
        self.fc_bb_part = fc_bb_part
        self.decode = BilinearUpsample(scale_factor=outstride)
        self.smax = nn.Softmax2d()
        self.saveid = 0.0

    def attend(self, op_preds, x):
        ''' Use op attentions
        '''
        VECTOR_DIMS = 28
        reg_op_preds = self.smax(op_preds)
        b, c, h, w = reg_op_preds.shape
        reg_op_preds = Variable(
            reg_op_preds.data.unsqueeze(0)
            .expand(VECTOR_DIMS, b, c, h, w)
            .permute(1, 2, 0, 3, 4)
            .contiguous()
            .view(b, c*VECTOR_DIMS, h, w))
        expanded = self.expand(x)
        attended = reg_op_preds * expanded
        return attended

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3, H, W].

        Return:
                1: category_pred, Shape: [batch*num_classes*H*W]
                2: bounding boxes, Shape: [batch*num_classes*H*W, 4]

        Bounding boxes (in theory) predict [cx, cy, w, h] of the relevant
        part or object bbox
        """

        insize = x.size()
        # batches = insize[0]
        spatial_dims = insize[-2:]

        y, _ = self.net(x)

        category_pred = self.fc(y)
        # category_pred = self.smax(self.fc(y))

        semantic_pred = []
        for v in self.op2sem:
            cls_op_pred = category_pred[:, v]
            if cls_op_pred.size(1) > 1:
                cls_op_pred = cls_op_pred.mean(dim=1, keepdim=True)
                # cls_op_pred = cls_op_pred.sum(dim=1, keepdim=True)
            semantic_pred.append(cls_op_pred)
        semantic_pred = torch.cat(semantic_pred, dim=1)
        if self.style == 'attention':
            # Add object and part channels to predict a semantic segmentation
            attended = self.attend(category_pred, y)
            rpout = self.rpn(attended)
            obj_bboxes = self.fc_bb_obj(rpout)
            part_bboxes = self.fc_bb_part(rpout)

        else:
            # Add object and part channels to predict a semantic segmentation
            rpout = self.rpn(y)
            obj_bboxes = self.fc_bb_obj(rpout)
            part_bboxes = self.fc_bb_part(rpout)

        return self.decode(
            [semantic_pred,
             category_pred,
             obj_bboxes,
             part_bboxes], spatial_dims)

        # def saveim(bar_, nm, sort=False):
        #     from ..extent_utils import tensor2im
        #     # print("tensor2im")
        #     tensor2im(
        #         bar_,
        #         './predictions/activities/'+nm+'_'+str(self.saveid)+'_.png',
        #         sort)

        # pdb.set_trace()

        # def _saveall_activities(model):
        #     print("Saving")
        #     # saveim(rpout, model+'_rpout', False)
        #     saveim(obj_bboxes, model+'_obj_bboxes', False)
        #     saveim(part_bboxes, model+'_part_bboxes', False)
        #     saveim(self.smax(category_pred), model+'_category_pred', False)
        #     saveim(self.smax(semantic_pred), model+'_sem_pred', False)
        #     self.saveid += 1

        # def waste():
        #     from ..extent_utils import tensor2flat, arr2im
        #     pwrpn = tensor2flat(self.rpn[0].weight)
        #     pwrpnim = arr2im(pwrpn)
        #     pwrpnim.save('./predictions/activities/rpnim.png')

        # (semantic_pred,
        #  category_pred,
        #  obj_bboxes,
        #  part_bboxes) = self.decode(
        #     [semantic_pred,
        #      category_pred,
        #      obj_bboxes,
        #      part_bboxes], spatial_dims)
        # _saveall_activities("nonlocal")

        # return (semantic_pred,
        #         category_pred,
        #         obj_bboxes,
        #         part_bboxes)

        # fc_bb_obj1 = nn.Conv2d(
        #     len(self.obj_bbox_inds),
        #     len(self.obj_bbox_inds), 3)
        # fc_bb_obj2 = nn.Conv2d(
        #     len(self.obj_bbox_inds),
        #     len(self.obj_bbox_inds), 3)
        # # self.fc_bb_obj = fc_bb_obj
        # self.fc_bb_obj = nn.Sequential(
        #     fc_bb_obj1,
        #     nn.Tanh(),
        #     fc_bb_obj2)
        # fc_bb_part1 = nn.Conv2d(
        #     len(self.part_bbox_inds),
        #     len(self.part_bbox_inds), 3)
        # fc_bb_part2 = nn.Conv2d(
        #     len(self.part_bbox_inds),
        #     len(self.part_bbox_inds), 3)
        # # self.fc_bb_part = fc_bb_part
        # self.fc_bb_part = nn.Sequential(
        #     fc_bb_part1,
        #     nn.Tanh(),
        #     fc_bb_part2)
        # self.decode = BilinearUpsample(scale_factor=outstride)
        # return semantic_pred, category_pred, obj_bboxes, part_bboxes

        # logits = self.fc(y)
        # # objpart_logits = self.decode(logits, spatial_dims)
        # objpart_logits = logits
        # # Add object and part channels to predict a semantic segmentation

        # category_pred = objpart_logits.index_select(1, self.op_pred_ind)
        # obj_bboxes = objpart_logits.index_select(1, self.obj_bbox_inds)
        # part_bboxes = objpart_logits.index_select(1, self.part_bbox_inds)
        # obj_bboxes = self.fc_bb_obj(obj_bboxes)
        # part_bboxes = self.fc_bb_part(part_bboxes)

        # semantic_pred = []
        # for v in self.op2sem:
        #     cls_op_pred = category_pred[:, v]
        #     if cls_op_pred.size(1) > 1:
        #         cls_op_pred = cls_op_pred.mean(dim=1, keepdim=True)
        #     semantic_pred.append(cls_op_pred)
        # semantic_pred = torch.cat(semantic_pred, dim=1)

        # semantic_pred = self.decode(semantic_pred, spatial_dims)
        # category_pred = self.decode(category_pred, spatial_dims)
        # obj_bboxes = self.decode(obj_bboxes, spatial_dims)
        # part_bboxes = self.decode(part_bboxes, spatial_dims)

        # return semantic_pred, category_pred, obj_bboxes, part_bboxes
