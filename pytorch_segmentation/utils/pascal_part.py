import os
import json
# import torch
import numpy as np
# from PIL import Image

# pylint: disable=fixme

# def get_pascal_parts_dictionary():
#     reference = {}
#     with open('id_reference.json', 'r') as f:
#       reference = json.load(f)
#     return reference
#
#
# def separate_key(key):
#   vals = key.rsplit("_", 5)
#   return {
#       'imid': vals[0],
#       'obj_id': vals[1],
#       'inst_id': vals[2],
#       'part_id': vals[3],
#       'xCoord': vals[4],
#   }


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
    with open(os.path.join(points_root, fname), 'r') as f:
        txt = f.readline().strip()
        parts_per_class = json.loads(txt)
        data = json.load(f)
    return data, parts_per_class


def get_point_mask(point_annotations, mask_type, size):

    # Ignore all non-placed points
    point_mask = np.zeros(size, np.int32) - 1
    # weights = np.zeros(size, np.float32)
    if len(point_annotations) == 0:
        return point_mask  # , weights

    # mode: Each annotation is the mode
    # of all responses
    if mask_type == 0:
        for point, answers in point_annotations.items():
            coords = point.split("_")
            i, j = int(coords[1]), int(coords[0])
            _answers = np.array([ans for ans in answers if ans >= 0])
            if len(_answers) == 0:
                continue
            ans_counts = np.bincount(np.array(_answers))
            modes = np.argwhere(ans_counts == np.amax(ans_counts)).flatten()
            # Choose most common non-ambiguous choice
            # Currently randomly breaks ties.
            # TODO: Develop better logic
            # modes.sort()
            np.random.shuffle(modes)
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
            ans_counts = np.argmax(np.bincount(_answers))
            modes = np.argwhere(ans_counts == np.amax(ans_counts)).flatten()

            # Choose most common non-ambiguous choice
            # Currently preferences object over part
            # TODO: Develop better logic
            modes = [m for m in modes if m >= 0]
            if len(modes) != 1:
                continue
            point_mask[i, j] = modes[0]
            # weights[i,j] = 1
            if point_mask[i, j] == 0:
                raise RuntimeError(
                    "pointmask 0 here... pascal_part.py line 74")

    # weighted: The ground truth annotations
    # are weighted by their ambiguity.
    elif mask_type == 2:
        raise NotImplementedError(
            "mask_type 'weighted' ({}) not implemented".format(mask_type))

    else:
        raise NotImplementedError(
            "mask_type {} not implemented".format(mask_type))

    return point_mask  # , weights
