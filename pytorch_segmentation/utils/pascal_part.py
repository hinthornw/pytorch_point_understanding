'''
pascal_part.py
author: William Hinthorn
'''
import os
import json
# import torch
import numpy as np
# from PIL import Image

# pylint: disable=fixme,invalid-name,missing-docstring


def get_pascal_object_part_points(points_root, fname):
    '''
    Loads the pointwise annotations for the pascal
    object-part inference task.

    Arguments:
      points_root: the root path for the pascal parts
        dataset folder

    Returns data, a dictionary of the form:
    { image id : {
                  "xCoordinate_yCoordinate": [response1, response2, ...]
                }
    }

    '''
    # fname = "pascal_gt.json"
    # TODO: Check these files...
    with open(os.path.join(points_root, fname), 'r') as f:
        # txt = f.readline().strip()
        parts_per_class = json.loads(f.readline().strip())
        data = json.loads(f.readline().strip())
    return data, parts_per_class


def get_valid_circle_indices(arr, center, r):
    ''' Returns the indices of all points of circle (center[1], center[0], r)
    within bounds of array arr.
    '''
    h, w = arr.shape
    i, j = center
    i = min(max(i, 0), h - 1)
    j = min(max(j, 0), w - 1)
    if r > 0:
        istart, istop = max(i - r, 0), min(i + r + 1, h)
        jstart, jstop = max(j - r, 0), min(j + r + 1, w)
        eyes = [y for y in range(istart, istop) for _ in range(jstart, jstop)]
        jays = [x for _ in range(istart, istop) for x in range(jstart, jstop)]
    else:
        eyes = [i]
        jays = [j]
    return eyes, jays


def get_point_mask(point_annotations, mask_type, size, smooth=False):

    # Ignore all non-placed points
    point_mask = np.zeros(size[:2], np.int32) - 1
    # weights = np.zeros(size, np.float32)
    if not point_annotations:
        return point_mask  # , weights

    # mode: Each annotation is the mode
    # of all responses
    if mask_type == 0:
        for point, answers in point_annotations.items():
            coords = point.split("_")
            i, j = int(coords[1]), int(coords[0])
            _answers = np.array([ans for ans in answers if ans >= 0])
            if _answers.size == 0:
                continue
            ans_counts = np.bincount(np.array(_answers))
            modes = np.argwhere(ans_counts == np.amax(ans_counts)).flatten()
            # Choose most common non-ambiguous choice
            # Currently randomly breaks ties.
            # TODO: Develop better logic
            # modes.sort()
            if len(modes) != 1:
                continue
            # np.random.shuffle(modes)
            if smooth:
                inds = get_valid_circle_indices(point_mask, (i, j), 3)
                point_mask[inds] = modes[0]
            else:
                point_mask[i, j] = modes[0]

            if point_mask[i, j] == 0:
                raise RuntimeError(
                    " pointmask 0 here... pascal_part.py line 74")
            # weights[i,j] = 1

    # consensus: only select those points
    # for which the (valid) responses are unanimous.
    # Ignores negative responses.
    elif mask_type == 1:
        for point, answers in point_annotations.items():
            coords = point.split("_")
            i, j = int(coords[1]), int(coords[0])
            _answers = np.array([ans for ans in answers if ans >= 0])
            # Not all responses agree. OR none
            if len(set(_answers)) != 1:
                continue
            # ans_counts = np.argmax(np.bincount(_answers))
            ans_counts = np.bincount(_answers)
            modes = np.argwhere(ans_counts == np.amax(ans_counts)).flatten()

            # Choose most common non-ambiguous choice
            # Currently preferences object over part
            # TODO: Develop better logic
            modes = [m for m in modes if m >= 0]
            if len(modes) != 1:
                continue
            if smooth:
                inds = get_valid_circle_indices(point_mask, (i, j), 3)
                point_mask[inds] = modes[0]
            else:
                point_mask[i, j] = modes[0]

            # weights[i,j] = 1
            if point_mask[i, j] == 0:
                raise RuntimeError(
                    "pointmask 0 here... pascal_part.py line 93")

    # weighted: The ground truth annotations
    # are weighted by their ambiguity.
    elif mask_type == 2:
        raise NotImplementedError(
            "mask_type 'weighted' ({}) not implemented".format(mask_type))

    else:
        raise NotImplementedError(
            "mask_type {} not implemented".format(mask_type))

    return point_mask  # , weights


def get_pascal_object_part_bboxes(bboxes_root):  # , imnames):
    '''
    Loads the pointwise annotations for the pascal
    object-part inference task.

    Arguments:
      bboxes_root: the root path for the pascal parts
        dataset folder
      imnames:  imnames[0] lists abs paths for training set
      imnames[1] lists abs path for val set

    '''
    bboxes_names = {}
    # im_abspaths = imnames[0] + imnames[1]
    # for im, _ in im_abspaths:
    # fname = os.path.splitext(os.path.split(im)[-1])[0]
    for fname in os.listdir(bboxes_root):
        imname, ext = os.path.splitext(fname)
        imname = os.path.split(imname)[-1]
        if ext == '.npz':
            bbfname = os.path.join(bboxes_root, imname + ext)
            bboxes_names[imname] = bbfname
    return bboxes_names


def load_bbox(fname, size):
    '''Load the npz file
    '''
    if os.path.isfile(fname):
        return np.load(fname)['arr']
    return np.zeros((10,)+size) - 1


def map_indices(tensor, pmap, ignore=255):
    rm = set([0, -1, ignore])
    vals = set(np.unique(tensor)) - rm
    for v in vals:
        tensor[tensor == v] = pmap[v]


def get_bbox(fname, op_map, size, op_map2=None):
    op_mapuse = {}
    tpose = False
    if op_map2 is not None:
        tpose = True
        op_mapuse = op_map
        # op_mapuse['obj'] = {}
        # for k, v in op_map2['obj'].items():
        #     op_mapuse['obj'][k] = op_map['obj'][v]
        # op_mapuse['parts'] = {}
        # for k, v in op_map2['parts'].items():
        #     for k2, v2 in v.items():
        #         op_mapuse['parts'][k2] = op_map['parts'][v2]
    else:
        op_mapuse = op_map
    bbox = load_bbox(fname, size)
    if tpose:
        bbox = bbox.transpose(2, 0, 1)
    obj_map = op_mapuse['obj']
    map_indices(bbox[0], obj_map)
    # map_indices(bbox[10], obj_map)
    parts_map = op_mapuse['parts']
    # print("parts_map:\t{}".format(parts_map))
    # print("partsun:\t{}".format(np.unique(bbox[5])))
    map_indices(bbox[5], parts_map)
    return bbox


def get_pmap(fname, which):
    return json.load(fname)[which+'_bbox']


def intify_dict(dic):
    '''converts number-stringed keys to proper integers'''
    if not isinstance(dic, dict):
        return dic
    dic_ = {}
    for k in dic:
        try:
            k_ = int(k)
        except ValueError:
            k_ = k
        dic_[k_] = intify_dict(dic[k])
    return dic_


def prune_ims(ltrain_val, dic):
    ''' Only choose images for which there are bbox anno
    '''
    pruned = []
    for stage in ltrain_val:
        stg_l = []
        for im, anno in stage:
            fname = os.path.split(im)[-1]
            fname = os.path.splitext(fname)[0]
            if fname in dic:
                stg_l.append((im, anno))
        pruned.append(stg_l)
    return pruned
