'''
author: William Hinthorn
with many functions adapted from @warmspringwinds
'''
# import sys
import os
import shutil
from collections import defaultdict as defdic
import json
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
# from PIL import Image, ImageFont, ImageDraw
from PIL import Image, ImageDraw

# Must be after to use locally modified torchvision libs
import torchvision
import torchvision.transforms

# from .models import resnet_dilated
# from .models import partsnet
from .models import extent_net
# from .datasets.pascal_voc import PascalVOCSegmentation
from .datasets.pascal_voc_extent import PascalVOCExtent
from .transforms import (ComposeJoint,
                         # FixedScaleJoint,
                         RandomScaleJoint,
                         RandomHorizontalFlipJoint,
                         RandomCropJoint)

# pylint: disable=too-many-arguments,invalid-name,no-member,too-many-lines
# pylint: disable=len-as-condition,unnecessary-lambda
# pylint: disable=too-many-locals, too-many-branches, arguments-differ
# flake8: noqa=W291


def poly_lr_scheduler(optimizer, init_lr, iteration, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iteration is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
        Credit @trypag
        https://discuss.pytorch.org/t/solved-learning-rate-decay/6825/5
    """
    if iteration % lr_decay_iter or iteration > max_iter:
        return optimizer

    lr = init_lr * (1 - iteration / float(max_iter))**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_training_loaders(dataset_dir, network_dims, batch_size=8,
                         num_workers=4, mask_type="consensus",
                         which='binary'):
    '''Returns loaders for the training set.
        args:
            :param ``network_dims``: ``dict`` which will
            store the output label splits
            :param ``dataset_dir``: str indicating the directory
            inwhich the Pascal VOC dataset is stored
            ... etc.
            :param ``mask_type``:
            :param ``which``: one of 'binary,' 'merged', or 'sparse'
            'binary': for each class: object or part
            'merged': for each class: object or one of k "super-parts"
            'sparse': for each calss: object or one of N parts
    '''

    assert isinstance(network_dims, dict)
    insize = 512  # 224
    train_transform = ComposeJoint(
        [
            # RandomScaleJoint(
            #     0.6, 1.05,
            #     (Image.BILINEAR, Image.NEAREST,
            #      Image.NEAREST, Image.NEAREST)),
            RandomHorizontalFlipJoint(),
            # # TO DO: make the -1 ignore index work for
            # bbox tensor..
            RandomCropJoint(crop_size=(insize, insize), pad_values=[
                0, 255, 255, -1]),
            [torchvision.transforms.ToTensor(), None, None, None],
            [torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
             None, None, None],
            # convert labels to tensors
            [None, torchvision.transforms.Lambda(
                lambda x: torch.from_numpy(np.asarray(x)).long()),
             torchvision.transforms.Lambda(
                 lambda x: torch.from_numpy(np.asarray(x)).long()),
             # Point Labels
             torchvision.transforms.Lambda(
                 lambda x: torch.from_numpy(np.asarray(x)).long())]
        ])

    trainset = PascalVOCExtent(dataset_dir,
                               network_dims=network_dims,
                               download=False,
                               joint_transform=train_transform,
                               mask_type=mask_type,
                               which=which)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True)

    valid_transform = ComposeJoint(
        [
            # FixedScaleJoint(
            #     insize,
            #     (Image.BILINEAR, Image.NEAREST,
            #      Image.NEAREST, Image.NEAREST)),
            [torchvision.transforms.ToTensor(), None, None, None],
            [torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
             None, None, None],
            # convert labels to tensors
            [None, torchvision.transforms.Lambda(
                lambda x: torch.from_numpy(np.asarray(x)).long()),
             torchvision.transforms.Lambda(
                 lambda x: torch.from_numpy(np.asarray(x)).long()),
             # Point Labels
             torchvision.transforms.Lambda(
                 lambda x: torch.from_numpy(np.asarray(x)).long())]
        ])

    valset = PascalVOCExtent(dataset_dir,
                             network_dims={},
                             train=False,
                             download=False,
                             joint_transform=valid_transform,
                             mask_type=mask_type,
                             which=which,
                             insize=insize)

    valset_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                                shuffle=False, num_workers=2)

    return (trainloader, trainset), (valset_loader, valset)


def flatten_logits(logits, number_of_classes):
    """Flattens the logits batch except for the logits dimension"""
    logits_permuted = logits.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    logits_flatten = logits_permuted_cont.view(-1, number_of_classes)
    return logits_flatten


def get_valid_logits(logits, index, number_of_classes):
    ''' processes predictions based on the valid indices (selected
    from annotations)
    '''
    if not index.size():
        if isinstance(logits, Variable):
            return Variable(torch.Tensor([]))
        import pdb
        pdb.set_trace()
        return torch.Tensor([])
    logits_flatten = flatten_logits(
        logits, number_of_classes=number_of_classes)
    return torch.index_select(logits_flatten, 0, index)


def flatten_annotations(annotations):
    '''Literally just remove dimensions of tensor.
    '''
    return annotations.view(-1)


def get_valid_annotations_index(flat_annos, mask_out_values=255):
    ''' Returns a tensor of indices of all nonzero values
    in a flat tensor.
    '''
    if isinstance(mask_out_values, int):
        mask_out_values = [mask_out_values]
    # nonz = torch.nonzero((flat_annos != mask_out_values))
    nonz = torch.ones_like(flat_annos).type(torch.ByteTensor)
    for val in mask_out_values:
        nonz = nonz & flat_annos.ne(val)
        # nonz[flat_annos.eq(val)] = 0
    nonz = torch.nonzero(nonz)
    if not nonz.size():
        return torch.LongTensor([])
    return torch.squeeze(nonz, 1)


def get_valid_annos(anno, mask_out_values):
    ''' selects labels not masked out
        returns a flattened tensor of annotations and the indices which are
        valid
    '''
    anno_flatten = flatten_annotations(anno)
    index = get_valid_annotations_index(
        anno_flatten, mask_out_values=mask_out_values)
    if index.numel() == 0:
        return index.clone(), index
    anno_flatten_valid = torch.index_select(anno_flatten, 0, index)
    return anno_flatten_valid, index


def numpyify_logits_and_annotations(logits, anno, flatten=True):
    ''' process logits and annotations for input into
    confusion matrix function.

    args::
        ``logits``: network predictoins
        ``anno``: ground truth annotations

    returns::
        flattened predictions, flattened annotations
    '''
    # First we do argmax on gpu and then transfer it to cpu
    if isinstance(logits, Variable):
        _logits = logits.data
    else:
        import pdb
        pdb.set_trace()
        _logits = logits
    if _logits.numel() == 0:
        return np.array([]), np.array([])
    # why keepdims then flatten???
    _, prediction = _logits.max(1, keepdim=True)
    prediction = prediction.squeeze(1)
    prediction_np = prediction.cpu().numpy()
    if isinstance(anno, Variable):
        anno_np = anno.data.cpu().numpy()
    else:
        anno_np = anno.cpu().numpy()
    if flatten:
        return prediction_np.flatten(), anno_np.flatten()
    return prediction_np, anno_np


def outputs_tonp_gt_1dim(logits, anno, op_map):
    ''' process logits and annotations for input into
    confusion matrix function.

    args::
        ``logits``: network predictoins
        ``anno``: ground truth annotations

    returns::
        flattened predictions, flattened annotations
    '''
    # outputs_tonp_gt_1dim(logits anno op_map)
    # check validity of stuff

    def to_pair(label, op_map):
        '''Use gt labels to select the correct pair of
           indices'''
        other_label = op_map[label]
        # new_label, pair = to_pair(label, 20)
        if label < other_label:
            return [label, other_label]
        return [other_label, label]

    _logits = logits.data.cpu().numpy()
    if isinstance(anno, Variable):
        anno_np = anno.data.cpu().numpy()
    else:
        anno_np = anno.numpy()
    predictions = np.zeros_like(anno_np)
    # for index, anno_ind in np.ndenumerate(anno_np):
    for index in zip(*np.where(np.logical_and(anno_np > 0, anno_np != 255))):
        anno_ind = anno_np[index]
        channel_indices = to_pair(anno_ind, op_map)  # num_to_aggregate)
        aided_prediction = np.argmax(
            [_logits[index, ci] for ci in channel_indices])
        predictions[index] = aided_prediction

    for i, lab in enumerate(anno_np):
        anno_ind = to_pair(lab, op_map)
        binl = anno_ind.index(lab)
        anno_np[i] = binl

    return predictions, anno_np


def outputs_tonp_gt(logits, anno, op_map, flatten=True):
    ''' process logits and annotations for input into
    confusion matrix function.

    args::
        ``logits``: network predictoins
        ``anno``: ground truth annotations

    returns::
        flattened predictions, flattened annotations
    '''
    def to_pair(label, op_map):
        '''Use gt labels to select the correct pair of
           indices'''
        other_label = op_map[label]
        # new_label, pair = to_pair(label, 20)
        if label < other_label:
            return [label, other_label]
        return [other_label, label]

    _logits = logits.data.cpu()
    if isinstance(anno, Variable):
        anno_np = anno.data.cpu().numpy()
    else:
        anno_np = anno.numpy()
    predictions = np.zeros_like(anno_np)
    # for index, anno_ind in np.ndenumerate(anno_np):
    for index in zip(*np.where(np.logical_and(anno_np > 0, anno_np != 255))):
        anno_ind = anno_np[index]
        batch_ind = index[0]
        i = index[1]
        j = index[2]
        channel_indices = to_pair(anno_ind, op_map)  # num_to_aggregate)
        aided_prediction = channel_indices[np.argmax(
            [_logits[batch_ind, ci, i, j] for ci in channel_indices])]

        predictions[batch_ind, i, j] = aided_prediction
    if flatten:
        return predictions.flatten(), anno_np.flatten()
    return predictions, anno_np

def compress_objpart_logits(logits, anno, op_map):
    ''' Reduce N x 41 tensor ``logits`` to an N x 2 tensor ``compressed``,
        where ``compressed``[0] => "object" and ``compressed``[1] => part
        (generic).
    args::
        ``logits``: network predictions => 2D tensor of shape (N, 41)
        ``anno``: ground truth annotations => 1D tensor of length N

    returns::
        compressed tensor of shape (N, 2)
    '''
    anno = anno.data.cpu()
    def to_pair(label, op_map):
        '''Use gt labels to isloate op loss
        '''
        other_label = op_map[label]
        # new_label, pair = to_pair(label, 20)
        if label < other_label:
            pair = [label, other_label]
            new_label = 0
        else:
            pair = [other_label, label]
            new_label = 1
        return new_label, pair

    indices = []
    new_anno = []
    for _, label in enumerate(anno):
        new_label, pair = to_pair(label, op_map)
        indices.append(pair)
        new_anno.append(new_label)
    len_ = len(indices)
    new_anno = Variable(torch.LongTensor(new_anno).cuda())
    indices = Variable(torch.LongTensor(indices).cuda())
    if len_ == 0:
        compressed_logits = Variable(torch.Tensor([]).cuda())
    else:
        compressed_logits = torch.gather(logits, 1, indices)
    return compressed_logits, new_anno


def get_miou(iou):
    '''Simply take mean of tensor...'''
    return np.mean(iou)


def get_mious(ious):
    ''' get mious for every iou in iterable
    '''
    resps = []
    for iou in ious:
        resps.append(get_miou(iou))
    return tuple(resps)


def get_ious(conf_mats):
    ''' Wrapper class to bulk get ious
    '''
    mats = []
    for cm in conf_mats:
        mats.append(get_iou(cm))
    return tuple(mats)


def get_iou(conf_mat):
    '''

    Used for computing the intersection over union metric
    using a confusion matrix. Pads unseen labels (union)
    with epsilon to avoid nan.
    Returns a vector of length |labels| with
    the IoU for each class in its appropriate
    place.

    '''
    intersection = np.diag(conf_mat)
    gt_set = conf_mat.sum(axis=1)
    predicted_set = conf_mat.sum(axis=0)
    union = gt_set + predicted_set - intersection
    # Ensure no divide by 1 errors
    eps = 1  # 1e-5
    union[union == 0] = eps
    iou = intersection / union.astype(np.float32)
    return iou


def get_precision_recall(conf_mat):
    ''' Returns the class-wise precision and recall given a
        confusion matrix.
        Note that this defaults to 0 to avoids divide by zero errors.
    '''
    intersection = np.diag(conf_mat)
    gt_set = conf_mat.sum(axis=1)
    predicted_set = conf_mat.sum(axis=0)
    precision = intersection / \
        np.array([np.max([pred, 1.0]) for pred in predicted_set]).astype(
            np.float32)
    recall = intersection / \
        np.array([np.max([gt, 1.0]) for gt in gt_set]).astype(np.float32)
    return precision, recall


def get_dense_confusion(log_fv, anno_fv, labels, op_map=None):
    ''' Returns a confusion matrix for the given semantic
    segmentation-esque outputs. If op_map is provided,
    seeks to index the output tensor first and turn it into a
    binary classification task.
    '''
    if op_map is None:
        log_np, anno_np = numpyify_logits_and_annotations(log_fv, anno_fv)
    else:
        # log_np, anno_np = outputs_tonp_gt(log_fv, anno_fv, op_map)
        log_np, anno_np = outputs_tonp_gt_1dim(log_fv, anno_fv, op_map)
        # log_fv_ = index_bbox_tensor(log_fv, anno_fv, 1)
        # if isinstance(log_fv_, Variable):
        #     log_fv_ = log_fv_.data
        # log_np = log_fv_.cpu().numpy()
        # anno_fv_ = anno_fv
        # if isinstance(anno_fv, Variable):
        #     anno_fv_ = anno_fv.data
        # anno_np = anno_fv_.cpu().numpy()
    if anno_np.size > 0:
        return confusion_matrix(
            y_true=anno_np,
            y_pred=log_np,
            labels=labels)
    return np.zeros((len(labels), len(labels)), np.int64)


def var2np(elem):
    '''Convert a variable to a numpy array'''
    if isinstance(elem, Variable):
        return elem.data.cpu().numpy()
    elif isinstance(elem, torch.Tensor):
        return elem.cpu().numpy()
    return np.array(elem)


def xywh2xyxy(bb):
    ''' Convert from (xc, yc, w, h) parametrization
        to (x1, y1, x2, y2) parametrization.
        returns a tuple
    '''
    xc, yc, w, h = tuple(bb)
    x1 = (xc - w / 2.0)
    x2 = (xc + w / 2.0)
    y1 = (yc - h / 2.0)
    y2 = (yc + h / 2.0)
    return (x1, y1, x2, y2)


def get_bbox_iou(bb1, bb2, separate=False):
    '''Adapted from
       pyimagesearch.com/2016/11/07/
       intersection-over-union-iou-for-object-detection/
    '''
    x1a, y1a, x2a, y2a = xywh2xyxy(bb1)
    x1b, y1b, x2b, y2b = xywh2xyxy(bb2)
    x1int = max(x1a, x1b)
    y1int = max(y1a, y1b)
    # x2int = max(x2a, x2b)
    # y2int = max(y2b, y2b)
    x2int = min(x2a, x2b)
    y2int = min(y2a, y2b)
    if x2int < x1int or y2int < y1int:
        interArea = 0.0
    else:
        interArea = float((x2int - x1int + 1) * (y2int - y1int + 1))
    aArea = (x2a - x1a + 1) * (y2a - y1a + 1)
    bArea = (x2b - x1b + 1) * (y2b - y1b + 1)
    union = float(aArea + bArea - interArea)
    if separate:
        return interArea, union
    return interArea / union

def _create_xy_list():
    return {'x': [], 'y': [], 'w': [], 'h': []}

def _create_iou_dict():
    return {'inters': 0.0, 'union': 0.0, 'iou': []}

rolling_pred = {}
rolling_anno = {}
rolling_pred_post = {}
rolling_anno_post = {}
rolling_iou = {}
# model_name = '2layerrelu'
# model_name = '1layergated'
# model_name = 'nonlocal'
model_name = '1layerfullrelu'
print("Saving bbox ious to : {}".format(model_name))
roll = False


def get_bbox_ious(bb_preds, bb_annos, mask, id2mean, thresh=0.5):
    ''' bb_preds = N(?) wise bb prediction tensor
    '''
    # import pdb
    # pdb.set_trace()
    bb_preds = var2np(bb_preds)
    bb_annos = var2np(bb_annos)
    # if bb_annos.size == 0:
    #     return {}
    mask = var2np(mask)
    bb_preds = bb_preds.reshape(-1, 4)
    bb_annos = bb_annos.reshape(-1, 4)
    # Must convert BOTH back to normal extent..
    uniques = [u for u in np.unique(mask) if u not in [255]]  # [0, 255]]
    inters = defdic(float)
    unions = defdic(float)
    ious = defdic(list)
    ok_pred = defdic(float)
    total_pred = defdic(float)

    def map_back(arr, meanw, meanh):
        '''Convert back to xy space'''
        arr[:, 2] = np.exp(arr[:, 2])
        arr[:, 3] = np.exp(arr[:, 3])
        arr[:, 0] *= meanw
        arr[:, 1] *= meanh
        arr[:, 2] *= meanw
        arr[:, 3] *= meanh

    if bb_preds.size == 0:
        return inters, unions, ok_pred, total_pred
    for u in uniques:
        inds = mask == u
        cwise_preds = bb_preds[inds]
        cwise_annos = bb_annos[inds]
        uind = str(int(u))
        if roll and uind not in rolling_anno:
            rolling_anno[uind] = _create_xy_list()
            rolling_anno_post[uind] = _create_xy_list()
            rolling_pred[uind] = _create_xy_list()
            rolling_pred_post[uind] = _create_xy_list()
            rolling_iou[uind] = _create_iou_dict()

            for bb_pred, bb_anno in zip(cwise_preds, cwise_annos):
                for xywh, ix in zip(['x', 'y', 'w', 'h'], [0, 1, 2, 3]):
                    rolling_anno[uind][xywh].append(float(bb_anno[ix]))
                    rolling_pred[uind][xywh].append(float(bb_pred[ix]))
        means = id2mean[u]
        meanh = means['h']
        meanw = means['w']
        map_back(cwise_preds, meanw, meanh)
        map_back(cwise_annos, meanw, meanh)
        for bb_pred, bb_anno in zip(cwise_preds, cwise_annos):
            inters_, unis = get_bbox_iou(bb_pred, bb_anno, True)
            if roll:
                for xywh, ix in zip(['x', 'y', 'w', 'h'], [0, 1, 2, 3]):
                    rolling_anno_post[uind][xywh].append(float(bb_anno[ix]))
                    rolling_pred_post[uind][xywh].append(float(bb_pred[ix]))
                rolling_iou[uind]['inters'] += inters_
                rolling_iou[uind]['union'] += unis
                rolling_iou[uind]['iou'].append(inters_ / unis)
            # tqdm.tqdm.write("Prediction/anno:\n\t{}\n\t{}".format(bb_pred, bb_anno))
            # tqdm.tqdm.write("IOU:\t{}".format(inters_ / unis))
            # tqdm.tqdm.write(
            #     "Rolling: {} ({})".format(
            #         rolling_iou['inters'] / rolling_iou['union'],
            #         np.mean(rolling_iou['iou'])))

            if inters_ > unis:
                import pdb
                pdb.set_trace()
            inters[u] += inters_
            unions[u] += unis
            iou = inters_ / unis
            ious[u].append(iou)
            if iou > thresh:
                ok_pred[u] += 1
            ok_pred[u] += 0
            total_pred[u] += 1
    if roll:
        with open('valdata/rolling_pred_{}.json'.format(model_name), 'w+') as f:
            json.dump(rolling_pred, f)
        with open('valdata/rolling_anno_{}.json'.format(model_name), 'w+') as f:
            json.dump(rolling_anno, f)
        with open('valdata/rolling_pred_post_{}.json'.format(model_name), 'w+') as f:
            json.dump(rolling_pred_post, f)
        with open('valdata/rolling_anno_post_{}.json'.format(model_name), 'w+') as f:
            json.dump(rolling_anno_post, f)
        with open('valdata/rolling_iou_{}.json'.format(model_name), 'w+') as f:
            json.dump(rolling_iou, f)
    return inters, unions, ious, ok_pred, total_pred


def get_bbox_scores(bbox_preds, bbox_annos, mask):
    ''' Score bbox predictions indexed (previously) by
    ``mask''. Returns dictionary of average (spatial+batch)
    loss per bbox. The loss is divided into by x, y, w, h
    values.
    '''

    bbox_preds = var2np(bbox_preds)
    bbox_annos = var2np(bbox_annos)
    if bbox_annos.size == 0:
        return {}
    mask = var2np(mask)
    bbox_preds = bbox_preds.reshape(-1, 4)
    bbox_annos = bbox_annos.reshape(-1, 4)
    uniques = np.unique(mask)
    dic_ = {}

    for u in uniques:
        inds = mask == u
        N = inds.sum()
        cwise_preds = bbox_preds[inds]
        cwise_annos = bbox_annos[inds]
        l1_norms = [np.linalg.norm(
            cwise_preds[:, i] - cwise_annos[:, i], 1) for i in range(4)]
        l1_norms = np.array(l1_norms) / N
        dic_[u] = l1_norms
    return dic_


def validate_batch(logits, annos, masks, matrices, op_map):
    ''' Computes the running IoU for the semantic and object-part tasks.
        args::
            :param (objpart_logits, objpart_anno): prediction, ground_truth
                    tensors for the object-part inference task
            :param (semantic_logits, semantic_anno): ditto for the semantic
            segmentation task
            :param overal_semantic_confusion_matrix: None or tensor of length
                                    |segmentation classes|. Total confusion
                                    matrix for semantic segmentation task
                                    for this epoch.
            :param overal_part_confusion_matrix: None or tensor of length
                                    |segmentation classes|. Total confusion
                                    matrix for object-part inference task
                                    for this epoch.

    '''
    # vvv TOO EFFING HACKY
    sem_l = list(range(21))
    op_l = list(range(36))  # check if is 1-indexed...
    bin_l = [0, 1]

    sem_log_fv, op_log_fv, bb_obj_fv, bb_part_fv = logits
    sem_anno_fv, op_anno_fv, bb_obj_anno, bb_part_anno = annos
    sem_mask, part_mask = masks
    ov_sem_cm, ov_op_cm, ov_op_gt_cm = matrices

    # sem_log_np, sem_anno_np = numpyify_logits_and_annotations(
    #     sem_log_fv, sem_anno_fv)
    # import pdb
    # pdb.set_trace()
    curr_sem_cm = get_dense_confusion(sem_log_fv, sem_anno_fv, sem_l)
    curr_op_cm = get_dense_confusion(op_log_fv, op_anno_fv, op_l)
    curr_op_gt_cm = get_dense_confusion(op_log_fv, op_anno_fv, bin_l, op_map)
    ov_cm = [ov_sem_cm, ov_op_cm, ov_op_gt_cm]
    currs = [curr_sem_cm, curr_op_cm, curr_op_gt_cm]
    for i, (ov, curr) in enumerate(zip(ov_cm, currs)):
        if ov is None:
            ov_cm[i] = curr
        else:
            ov_cm[i] += curr
    #  Handle bbox scoring
    curr_bb_obj_dic = get_bbox_scores(bb_obj_fv, bb_obj_anno, sem_mask)
    curr_bb_part_dic = get_bbox_scores(bb_part_fv, bb_part_anno, part_mask)
    # Handle bbox iou data
    # import pdb
    # pdb.set_trace()
    # curr_bb_obj_ious = get_bbox_ious(bb_obj_fv, bb_obj_anno, sem_mask, id2mean, 0.5)
    # curr_bb_part_ious = get_bbox_ious(bb_part_fv, bb_part_anno, part_mask, id2mean, 0.5)

    curr_cm = (curr_sem_cm, curr_op_gt_cm, curr_op_gt_cm)
    bb_dics = (curr_bb_obj_dic, curr_bb_part_dic)
    # bb_iou_dics = (curr_bb_obj_ious, curr_bb_part_ious)

    return curr_cm, ov_cm, bb_dics  # , bb_iou_dics


def write_ious(writer, index, ioudests, ioulbs, ious):
    ''' Write out vector-valued elements
    '''
    for ioudest, ioulb, iou in zip(ioudests, ioulbs, ious):
        writer.add_scalars(ioudest,
                           {ioulb + ' ' + str(i): v for i,
                            v in enumerate(iou)},
                           index)


def write_scalars(writer, prefs, scalars, index):
    ''' Bulk write scalars
    '''
    for pref, scalar in zip(prefs, scalars):
        writer.add_scalar(pref, scalar, index)


def write_dics(writer, index, dests, prefs, dics):
    ''' Wrapper to write multiple dics of the same style
    '''
    for dest, pref, dic in zip(dests, prefs, dics):
        write_dic(writer, dest, pref, dic, index)


def write_dic(writer, dest, pref, dic, index):
    '''tboard write out a flat dictionary'''
    writer.add_scalars(
        dest,
        {pref + ' ' + str(k): v for k,
         v in dic.items()}, index)


def save_checkpoint(state, is_best, folder='models',
                    filename='checkpoint.pth.tar'):
    ''' Saves a model
        args::
            :param ``staet``: dictionary containing training data.
            :param ``is_best``: boolean determining if this represents
                            the best-trained model of this session
            :param ``folder``: relative path to folder in which to save
            checkpoint
            :param ``filename``: name of the checkpoint file

        additionally copies to "[architecture]" + "_model_best.pth.tar"
        if is_best.
    '''
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(
                folder, filename), os.path.join(
                    folder, state['arch'] + '_model_best.pth.tar'))


def load_checkpoint(load_path, fcn):
    ''' Loads network parameters (and optimizer params) from a checkpoint file.
        args::
            :param ``load_path``: string path to checkpoint file.
            :param ``fcn``: torch.nn network
            :param ``optimizer``: duh
        returns the starting epoch and best scores
    '''
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)

        try:
            start_epoch = checkpoint['epoch']
        except KeyError:
            start_epoch = 0

        scores = {
            'best_op_bin_score': 0.0,
            'best_op_mult_score': 0.0,
            'best_ov_bb_obj_prec': 0.0,
            'best_ov_bb_part_prec': 0.0,
            'best_ov_bb_obj_miou': 0.0,
            'best_ov_bb_part_miou': 0.0,
            'best_sem_miou': 0.0,
            'best_op_miou': 0.0,
            'best_op_gt_miou': 0.0}

        for k, v in scores.items():
            if k in checkpoint:
                scores[k] = checkpoint[k]
        # try:
        #     best_semantic_val_score = checkpoint['best_semantic_mIoU']
        #     best_objpart_val_score = checkpoint['best_objpart_mIoU']
        # except KeyError:
        #     best_semantic_val_score = 0.0
        #     best_objpart_val_score = 0.0
        # try:
        #     best_objpart_accuracy = checkpoint['best_objpart_accuracy']
        # except KeyError:
        #     best_objpart_accuracy = 0.0

        state_dict = {}
        model_sd = fcn.state_dict()

        if 'state_dict' in checkpoint:
            it = checkpoint['state_dict'].items()
        else:
            it = checkpoint.items()

        import re
        for k, v in it:
            # For drn38 original basefile
            k2 = re.sub('base.([0-9]+).([0-9]+).', r'net.layer\1.\2.', k)
            if k2 in model_sd:
                if v.size() != model_sd[k2].size():
                    print(
                        "{} not equal to {}".format(
                            model_sd[k2].size(),
                            v.size()))
                    continue
                state_dict[k2] = v
                continue
            k_ = k.split(".")
            if k_[0] == 'resnet34_8s':
                k_[0] = 'net'
            elif 'layer' in k_[0]:
                k_.insert(0, 'net')
            k_ = ".".join(k_)
            if k_ not in model_sd:
                print("Layer {} from checkpoint not found in model".format(k_))
                continue
            elif model_sd[k_].size() != v.size():
                print(
                    "{} not equal to {}".format(
                        model_sd[k_].size(),
                        v.size()))
                continue
            state_dict[k_] = v
            # optim_state_dict[k_] = checkpoint['optimizer'][k]
        # fcn.load_state_dict(checkpoint['state_dict'])
        fcn.load_state_dict(state_dict, strict=False)
        _epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_path, _epoch))
    else:
        raise RuntimeError("{} does not exist.".format(load_path))

    return (start_epoch, scores)


def safe_index_select(a, dim, b):
    ''' Handles empty index tensors
    '''
    if b.numel() == 0 or a.numel() == 0:
        c = torch.Tensor([])
        if isinstance(a, Variable):
            return Variable(c)
        return c.type(a.type())
    return torch.index_select(a, dim, b)


def index_bbox_tensor(preds, which, BBOX_L=4):
    ''' ExtentNet outputs dense predictions, with a bbox predicted
    for each object and part at every spatial location. We only
    use the regression loss on predictions for the appropriate
    channel.
    Parameters:
        preds: Predictions. shape: Nx(4C) where C is the number
        of classes.
        which: Semantic classes indicating the obj/part number
            shape: N

    Returns a tensor of length 4N (could be reshaped to be
    Nx4, with each i representing a unique spatial location
    in the original tensor.

    '''
    # BBOX_L = 4  # Each BBox is a len 4 vector
    if preds.numel() == 0:
        return preds
    N, four_x_c = preds.shape
    C = four_x_c // BBOX_L
    offset = Variable(torch.arange(0, C*N, C)
                      .type(torch.LongTensor).cuda())
    inds = which + offset  # - 1
    preds = preds.view(-1, BBOX_L)
    # print(preds.shape, inds.data.cpu().min(), inds.data.cpu().max())
    if inds.data.cpu().max() >= preds.shape[0]:
        import pdb
        pdb.set_trace()
    return preds.index_select(0, inds).view(-1)


def get_bb_mean_dic():
    ''' hacky way of fetching a map from id to mean values
    '''
    root = os.path.dirname('.')
    fold = os.path.normpath(os.path.abspath(os.path.join(
        root, '..', '..', 'datasets', 'pascal_object_parts')))
    fname = os.path.join(fold, 'bb_op_means.json')
    with open(fname, 'r') as f:
        mean_dic = json.load(f)
    mean_dic = {(int(k)-1)//4: v for k, v in mean_dic.items()}
    return mean_dic


def v2bb_pred_2xywh(bb_tensor, mask_, h, w, id2mean):
    ''' Undoes the conversion of width/height channels
    '''
    mask = mask_.data.cpu().numpy()
    uniques = [int(el) - 1 for el in np.unique(
        mask) if int(el) not in [
            0, 255, -1]]
    b, c, h, w = bb_tensor.shape
    _bb_tensor = np.array(bb_tensor.view(b, -1, 4, h, w))
    # import pdb
    # pdb.set_trace()
    # bbnz = np.logical_and(_bb_tensor[:, :, 2, :, :] 0, _bb_tensor[:, :, 3, :, :] > 0)
    for u in uniques:
        means = id2mean[u]
        meanh = means['h']
        meanw = means['w']
        # where = (mask == u + 1)
        # where = np.logical_and(mask == u + 1, bbnz)
        where = mask == u + 1
        # without log
        # _bb_tensor[:, u, 0, :, :][where] *= meanw
        # _bb_tensor[:, u, 1, :, :][where] *= meanh
        # _bb_tensor[:, u, 2, :, :][where] *= meanw
        # _bb_tensor[:, u, 3, :, :][where] *= meanh
        # with log
        _bb_tensor[:, u, 2, :, :][where] = np.exp(_bb_tensor[:, u, 2, :, :][where])
        _bb_tensor[:, u, 3, :, :][where] = np.exp(_bb_tensor[:, u, 3, :, :][where])
        _bb_tensor[:, u, 0, :, :][where] *= meanw
        _bb_tensor[:, u, 1, :, :][where] *= meanh
        _bb_tensor[:, u, 2, :, :][where] *= meanw
        _bb_tensor[:, u, 3, :, :][where] *= meanh
    return _bb_tensor.reshape(b, c, h, w)


def v2bb_whxy_2delta(bb_tensor, mask, id2mean):
    ''' converts the width/height channels of a tensor
        to deltas from the mean over the dataset.
    '''
    # re: pdb - look at all values here to see if dic is relevant
    uniques = [int(el) - 1 for el in np.unique(mask) if int(el) not in [
        0, 255, -1]]
    bbnz = np.logical_and(bb_tensor[:, 2, :, :] > 0, bb_tensor[:, 3, :, :] > 0)
    for u in uniques:
        means = id2mean[u]
        meanh = means['h']
        meanw = means['w']
        where = np.logical_and(mask == u + 1, bbnz)
        # without log
        # bb_tensor[:, 0, :, :][where] /= meanw
        # bb_tensor[:, 1, :, :][where] /= meanh
        # bb_tensor[:, 2, :, :][where] /= meanw
        # bb_tensor[:, 3, :, :][where] /= meanh
        # with log
        bb_tensor[:, 0, :, :][where] /= meanw
        bb_tensor[:, 1, :, :][where] /= meanh
        bb_tensor[:, 2, :, :][where] /= meanw
        bb_tensor[:, 3, :, :][where] /= meanh
        # bb_tensor[:, 0, :, :][where] /= meanw
        # bb_tensor[:, 1, :, :][where] /= meanh
        bb_tensor[:, 2, :, :][where] = bb_tensor[:, 2, :, :][where].log()
        bb_tensor[:, 3, :, :][where] = bb_tensor[:, 3, :, :][where].log()


def bb_pred_2xywh(bb_tensor, mask_, h, w, id2mean):
    ''' Undoes the conversion of width/height channels
    '''
    mask = mask_.data.cpu().numpy()
    uniques = [int(el) - 1 for el in np.unique(
        mask) if int(el) not in [
            0, 255, -1]]
    b, c, h, w = bb_tensor.shape
    _bb_tensor = np.array(bb_tensor.view(b, -1, 4, h, w))
    _bb_tensor[:, :, 0, :, :] *= w
    _bb_tensor[:, :, 1, :, :] *= h
    _bb_tensor[:, :, 2, :, :] *= w
    _bb_tensor[:, :, 3, :, :] *= h
    # import pdb
    # pdb.set_trace()
    for u in uniques:
        means = id2mean[u]
        meanh = means['h']
        meanw = means['w']
        where = (mask == u + 1)
        _bb_tensor[:, u, 2, :, :][where] += meanw
        _bb_tensor[:, u, 3, :, :][where] += meanh
    return _bb_tensor.reshape(b, c, h, w)


def bb_wh_2delta(bb_tensor, mask, id2mean):
    ''' converts the width/height channels of a tensor
        to deltas from the mean over the dataset.
    '''
    # re: pdb - look at all values here to see if dic is relevant
    uniques = [int(el) - 1 for el in np.unique(mask) if int(el) not in [
        0, 255, -1]]
    for u in uniques:
        means = id2mean[u]
        meanh = means['h']
        meanw = means['w']
        where = mask == u + 1
        bb_tensor[:, 2, :, :][where] -= meanw
        bb_tensor[:, 3, :, :][where] -= meanh


class SupervisedSmoothL1Loss(nn.Module):
    '''
        Selects the appropriate class to supervise based on
        an image's semantic segmentation
    '''
    def __init__(self, size_average=True, reduce=True):
        super(SupervisedSmoothL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss(size_average=size_average, reduce=reduce)

    def forward(self, output, target):
        '''
        args:
            :param ``output``: Network output - size: N x 4*C
            where C is the number of object (part) classes
            :param ``target``: labels - size N
            :param ``which`1: states the semantic class associated
            with each  spatial dimension. size: N
            Values range from 1 to |C| where |C| for semantic
            segmentation in Pascal is 20 (bb not included).
        '''
        # BBOX_L = 4
        # N, four_x_c = output.shape
        # C = four_x_c // BBOX_L
        # offset = Variable(torch.arange(0, C*N, C)
        #                   .type(torch.LongTensor).cuda())
        # inds = which + offset
        # output_selected = output.view(-1, BBOX_L) \
        #     .index_select(0, inds).view(-1)
        bbox_loss = self.loss(output, target)
        return bbox_loss

class NegLogSoftmax(nn.Module):
    ''' Computes the log of a softmaxed logits and then computes NLLLoss
    '''
    def __init__(self):
        super(NegLogSoftmax, self).__init__()
        self.loss = nn.NLLLoss(size_average=False, ignore_index=255)

    def forward(self, logits, gt):
        return self.loss(logits.log(), gt)


def get_network_and_optimizer(
        arch,
        network_dims,
        load_from_local=False,
        model_path=None,
        train_params=None,
        init_lr=0.0001,
        style='1relu'):
    ''' Gets the network and corresponding optimizer.
        args::
            # of semantic segmentaton classes (final output)
            :param ``number_of_classes``:
            :param ``to_aggregate``: # of classes which have corresponding
            object and part channels.
            :param ``load_from_local``: boolean variable to determine whether
            to load parameters from a local checkpoint file
            :param ``model_path``: required if ``load_from_local`` is ``True``,
            String path to checkpoint file

        returns a net, optimizer, (criteria), and best scores

        TODO: make this flexible. This is really sloppy.

    '''

    fcn = extent_net.ExtentNet(arch, network_dims, style)
    params_to_optimize = [param for param in fcn.parameters(
        ) if param.requires_grad]

    optimizer = optim.Adam(params_to_optimize, lr=init_lr, weight_decay=0.0001)
    # optimizer = optim.Adam(fcn.parameters(), lr=init_lr, weight_decay=0.0001)
    if load_from_local:
        (start_epoch, scores) = load_checkpoint(
            model_path, fcn)
    else:
        start_epoch = 0
        scores = {}
    fcn.cuda()
    fcn.train()

    # semantic_criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    # bbox_criterion = nn.SmoothL1Loss(size_average=True, reduce=True)
    bbox_criterion = SupervisedSmoothL1Loss(size_average=False, reduce=True)
    # objpart_criterion = NegLogSoftmax().cuda()
    # semantic_criterion = NegLogSoftmax().cuda()
    objpart_criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    semantic_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=255).cuda()

    _to_update = {
        'bbox_criterion': bbox_criterion,
        'semantic_criterion': semantic_criterion,
        'objpart_criterion': objpart_criterion,
        'start_epoch': start_epoch
    }
    train_params.update(_to_update)
    train_params.update(scores)

    # optimizer = optim.Adam(fcn.parameters(), lr=0.0001, weight_decay=0.0001)
    return fcn, optimizer


def get_cmap():
    ''' Return a colormap stored on disk
    '''
    fname = os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)),
        'colortable.json')
    with open(fname, 'r') as f:
        cmap = json.load(f)
    return cmap

def paint_point(prediction, im):
    ''' semantic segmentation
    '''
    im = im - im.min()
    im = im / np.max(im)
    im = im * 255
    im = im.astype(np.uint8)
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    eyes, jays = np.where(np.logical_and(prediction > 0, prediction != 255))
    for i, j in zip(eyes, jays):
        cls = prediction[i, j]
        draw.arc(
            ((j - 3, i - 3),
             (j + 3, i + 3)), 0, 360, fill='red')
        loc_ = (j + 3, i + 3)
        loc2_ = tuple([l+d for l, d in zip(loc_, (20, 10))])
        draw.rectangle((loc_, loc2_), fill="black")
        draw.text(loc_, "[{}]".format(cls), fill=(255, 255, 255, 128))
    return im

def paint_image(prediction, im, cmap, alpha=0.6):
    ''' semantic segmentation
    '''
    # import pdb
    # pdb.set_trace()
    tqdm.tqdm.write("Painting Image")
    cmask = np.zeros_like(im, dtype=np.float32)
    classes = [u for u in np.unique(prediction) if u != 0]
    inds = {}
    for cls in classes:
        if cls <= 0:
            continue
        ind = prediction == cls
        topleft = tuple([min(l) for l in np.where(ind)])
        inds[cls] = topleft
        cmask[ind, :] = cmap[cls]
    mx = cmask.max()
    if mx > 0:
        cmask = cmask.astype(np.float32) / mx
    ind = prediction > 0
    im[ind] = im[ind] * \
        (1.0 - alpha) + cmask[ind] * (float(alpha))
    im = im - im.min()
    im = im / np.max(im)
    im = im * 255
    im = im.astype(np.uint8)
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    # draw.text((10, 10), "{}".format(classes), fill=(255, 255, 255, 128))
    for cls, loc in inds.items():
        loc_ = loc[::-1]
        loc2_ = tuple([l+d for l, d in zip(loc_, (20, 10))])
        draw.rectangle((loc_, loc2_), fill="black")
        draw.text(loc_, "[{}]".format(cls), fill=(255, 255, 255, 128))
    # font=ImageFont.truetype(
    return im


def validate_and_output_images(net, loader, bb_mean_dic, alpha=0.6):
    # pylint: disable=too-many-statements
    ''' Computes mIoU for``net`` over the a set.
        args:: :param ``net``: network (in this case resnet34_8s_dilated
            :param ``loader``: dataloader (in this case, validation set loader)

        returns the mIoU (ignoring classes where no parts were annotated) for
        the semantic segmentation task and object-part inference task.
      0: "background",
      1: "aeroplane",
      2: "bicycle",
      3: "bird",
      4: "boat",
      5: "bottle",
      6: "bus",
      7: "car",
      8: "cat",
      9: "chair",
      10: "cow",
      11: "diningtable",
      12: "dog",
      13: "horse",
      14: "motorbike",
      15: "person",
      16: "pottedplant",
      17: "sheep",
      18: "sofa",
      19: "train",
      20: "tvmonitor"
      21: "aeroplane_part",
      22: "bicycle_part",
      23: "bird_part",
      24: "bottle_part",
      25: "bus_part",
      26: "car_part",
      27: "cat_part",
      28: "cow_part",
      29: "dog_part",
      30: "horse_part",
      31: "motorbike_part",
      32: "person_part",
      33: "pottedplant_part",
      34: "sheep_part",
      35: "train_part",

    '''
    # Get text reference
    fname = os.path.join(os.path.dirname(__file__), 'bbdict.json')
    with open(fname) as f:
        bbdict = json.load(f)
        bbdict = {int(k): v for k, v in bbdict.items()}
    valdir = os.path.join(
        os.path.dirname(__file__),
        'valscores')
    valf = os.path.join(
        valdir,
        'epoch_19_bb_scores.json')
        # 'epoch_-1_bb_scores.json')
    with open(valf) as f:
        valscores = json.load(f)
    valscores = {int(k): v for k, v in valscores.items()}
    scores_sorted = sorted(valscores.items(), key=lambda x: x[1])
    N = 20
    inds_to_score = {}
    inds_to_score['worst'] = {k: v for k, v in scores_sorted[:N]}
    inds_to_score['best'] = {k: v for k, v in scores_sorted[-N:]}
    def draw_bb(bb, i, j, bid, im):
        ''' Draw a single bounding box
        '''
        dx, dy, w, h = bb
        # xc = np.floor(j + dx)
        # yc = np.floor(i + dy)
        # TODO: Redo all the tensor annos so this isnt backward
        xc = np.floor(j - dx)
        yc = np.floor(i - dy)
        x0 = np.floor(xc - w//2 + 0.5)
        x1 = np.floor(xc + w//2 + 0.5)
        y0 = np.floor(yc - h//2 + 0.5)
        y1 = np.floor(yc + h//2 + 0.5)
        draw = ImageDraw.Draw(im)
        draw.rectangle(((x0, y0), (x1, y1)), fill=None, outline="yellow")
        draw.ellipse([(j-3, i-3), (j+3, i+3)])
        draw.text((x0+10, y0+10), "{}".format(bid))  # font=ImageFont.truetype(
        # "font_path123"))

    def draw_bbox(prediction, op_anno, im, h, w, bb_mean_dic):
        ''' Draw a bbox on im
        '''

        # map predictions back to pixel space
        # tscaled = bb_pred_2xywh(prediction, op_anno, h, w, bb_mean_dic)
        tscaled = v2bb_pred_2xywh(prediction, op_anno, h, w, bb_mean_dic)
        tscaled = tscaled.reshape(1, -1, 4, h, w)
        op_anno_np = op_anno.data.cpu().numpy()
        uniques = [u for u in np.unique(op_anno_np) if u not in [-1, 255]]
        for u in uniques:
            ind = np.where(op_anno_np == u)[1:]
            # where = (mask == u + 1)
            bboxes = tscaled[0, u-1][:, ind[0], ind[1]]
            bboxes = bboxes.transpose(1, 0)
            name = bbdict[u]
            for k, (bb, i, j) in enumerate(zip(bboxes, ind[0], ind[1])):
                bid = '{}.{}'.format(name, k)
                draw_bb(bb, i, j, bid, im)
                # print(bb, i, j)


    net.eval()
    # hardcoded in for the object-part infernce
    # no_parts = [0, 4, 9, 11, 18, 20, 24, 29, 31, 38, 40]
    # objpart_labels, semantic_labels = labels
    cmap = get_cmap()
    # valset_loader
    # i = 0
    for i, data in enumerate(tqdm.tqdm(loader)):
        img, sem_target, bbox_anno, objpart_anno, _ = data
        if i in inds_to_score['best']:
            prefix = 'best'
            index_score = inds_to_score['best'][i]
        elif i in inds_to_score['worst']:
            prefix = "worst"
            index_score = inds_to_score['worst'][i]
        else:
            continue
        # import pdb
        # pdb.set_trace()
        # print(np.unique(sem_target), np.unique(objpart_anno))
        _semseg_im = np.array(img).squeeze(0).transpose(1, 2, 0)
        _bb_im = np.copy(_semseg_im).astype(np.float64)
        _bb_im -= _bb_im.min()
        _bb_im /= _bb_im.max()
        _bb_im *= 255
        _bb_im = _bb_im.astype(np.uint8)
        _bb_im = Image.fromarray(_bb_im)
        img = Variable(img.cuda())
        sem_target = Variable(sem_target.cuda())
        bbox_anno = Variable(bbox_anno.cuda())
        objpart_anno = Variable(objpart_anno.cuda())
        semantic_logits, op_logits, bb_obj_preds, bb_part_preds = net(img)
        # pdb.set_trace()

        h, w, _ = _semseg_im.shape
        desth = h*2
        destw = w*2
        x_offset = 0
        y_offset = 0
        dest_im = Image.new('RGB', (destw, desth))

        _semseg_im = _semseg_im.astype(np.float32)
        _semseg_im -= _semseg_im.min()
        _semseg_im /= _semseg_im.max()

        # Copy to allow for tiling of inferences
        _gt_im = np.copy(_semseg_im)
        _op_im = np.copy(_semseg_im)


        # Paint semseg for im
        prediction, sem_annos = numpyify_logits_and_annotations(
            semantic_logits, sem_target, flatten=False)
        prediction = prediction.squeeze(0)
        _semseg_im = paint_image(prediction, _semseg_im, cmap, alpha)
        dest_im.paste(_semseg_im, (x_offset, y_offset))
        x_offset += w
        sem_annos = sem_annos.squeeze(0)
        _gt_im = paint_image(sem_annos, _gt_im, cmap, alpha)
        dest_im.paste(_gt_im, (x_offset, y_offset))
        x_offset = 0
        y_offset += h
        # Look for bbox_anno data.
        bb_preds = torch.cat(
            [bb_obj_preds, bb_part_preds],
            dim=1)
        draw_bbox(
            bb_preds.data.cpu(),
            objpart_anno,
            _bb_im, h, w, bb_mean_dic)
        dest_im.paste(_bb_im, (x_offset, y_offset))
        x_offset += w
        # Output op data
        op_prediction, sem_annos = numpyify_logits_and_annotations(
            op_logits, objpart_anno, flatten=False)
        op_prediction = op_prediction.squeeze(0)
        _op_im = paint_image(op_prediction, _op_im, cmap, alpha)
        dest_im.paste(_op_im, (x_offset, y_offset))

        dest_im.save(
            "predictions/extent/{}_validation_tiled_{}_{}.png".format(
                prefix, i, index_score))
        # i += 1
        _semseg_im.close()
        dest_im.close()
        _gt_im.close()


def tensor2flat(t_, sort=True):
    '''Creates a flat, zero-padded array of a activity
    (or parameter) tensor
    '''
    from itertools import product
    t_ = t_.data.cpu()
    h, w = t_.shape[-2:]
    t = t_.view(-1, h, w)
    assert t.dim() == 3
    c, h, w = t.shape
    sqr = np.sqrt(c)
    new_w, new_h = int(sqr)*w, int(sqr)*h
    if sqr % 1 > 0:
        if w > h:
            new_h += h
        else:
            new_w += w
    iters = range(0, new_h, h), range(0, new_w, w)
    outarr = np.zeros((new_h, new_w))
    if sort:
        iterobj = sorted(t, key=lambda x: np.linalg.norm(x))
    else:
        iterobj = t
    for (i, j), el in zip(product(*iters), iterobj):
        el_ = el - el.min()
        el_ = el_ / (el_.max()+1e-7)
        el_ *= 255
        outarr[i:i+h, j:j+w] = el_
    return outarr


def arr2im(arr):
    ''' Scale a numpy array to 0-255 uint
    '''
    arr = (arr - arr.min())
    arr = arr / arr.max()
    arr *= 255
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    return im


def tensor2im(t_, name, sort=True):
    ''' Wrapper
    '''
    t = tensor2flat(t_, sort)
    im = arr2im(t)
    im.save(name)
