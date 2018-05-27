import os
import sys
import tarfile
import torch
import torch.utils.data as data

import numpy as np

from six.moves import urllib
from PIL import Image


from ..utils.pascal_voc import get_augmented_pascal_image_annotation_filename_pairs
from ..utils.pascal_voc import convert_pascal_berkeley_augmented_mat_annotations_to_png
from ..utils.pascal_part import get_pascal_object_part_points, get_point_mask

_MASKTYPE = {'mode': 0, 'consensus': 1, 'weighted': 2}

# flake8: noqa=E501
# pylint: disable=too-many-instance-attributes, too-few-public-methods, fixme


class PascalVOCSegmentation(data.Dataset):

    # Class names if all parts are merged
    CLASS_NAMES = [
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted-plant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor',
        'aeroplane_part',
        'bicycle_part',
        'bird_part',
        'boat_part',
        'bottle_part',
        'bus_part',
        'car_part',
        'cat_part',
        'chair_part',
        'cow_part',
        'diningtable_part',
        'dog_part',
        'horse_part',
        'motorbike_part',
        'person_part',
        'potted-plant',
        'sheep_part',
        'sofa_part',
        'train_part',
        'tvmonitor_part']

    # Urls of original pascal and additional segmentations masks
    PASCAL_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    BERKELEY_URL = 'http://www.eecs.berkeley.edu/Research/Projects/' \
        'CS/vision/grouping/semantic_contours/benchmark.tgz'

    PASCAL_TAR_FILENAME = "VOCtrainval_11-May-2012.tar"
    BERKELEY_TAR_FILENAME = "benchmark.tgz"

    PASCAL_ROOT_FOLDER_NAME = "VOCdevkit"
    BERKELEY_ROOT_FOLDER_NAME = "benchmark_RELEASE"

    PASCAL_POINT_FOLDER_NAME = "pascal_object_parts"
    PASCAL_POINT_FILE_NAME = "pascal_gt.json"

    def __init__(self,
                 root,
                 network_dims,
                 train=True,
                 joint_transform=None,
                 download=False,
                 split_mode=2,
                 merge_mode="objpart",
                 mask_type="consensus",
                 which='binary'):
        ''' Returns label_attrs: a dictionary of the number of parts associated
        with each semantic part in the dataset. This is used for defining the
        output dimensions of the network.
        '''
        assert isinstance(network_dims, dict), 'Unable to pass on network' \
            'directions iwth network_dims dict'
        assert mask_type in _MASKTYPE, "mask_type {} error. Must be one of {}".format(
            mask_type, list(_MASKTYPE))

        assert which in ['binary', 'merged', 'sparse']
        self.PASCAL_POINT_FILE_NAME = which + "_" + self.PASCAL_POINT_FILE_NAME
        print("Collecting points from : {}".format(self.PASCAL_POINT_FILE_NAME))

        self.mask_type = _MASKTYPE[mask_type]
        self.root = root
        self.pascal_tar_full_download_filename = os.path.join(
            self.root, self.PASCAL_TAR_FILENAME)
        self.berkeley_tar_full_download_filename = os.path.join(
            self.root, self.BERKELEY_TAR_FILENAME)

        self.pascal_full_root_folder_path = os.path.join(
            self.root, self.PASCAL_ROOT_FOLDER_NAME)
        self.berkeley_full_root_folder_path = os.path.join(
            self.root, self.BERKELEY_ROOT_FOLDER_NAME)

        self.joint_transform = joint_transform

        if download:
            self._download_dataset()
            self._extract_dataset()
            self._prepare_dataset()

        pasc_anno_fnames_train_val = get_augmented_pascal_image_annotation_filename_pairs(
            self.pascal_full_root_folder_path, self.berkeley_full_root_folder_path, mode=split_mode)

        self.point_annotations, label_attrs = get_pascal_object_part_points(os.path.join(
            self.root, self.PASCAL_POINT_FOLDER_NAME), self.PASCAL_POINT_FILE_NAME)

        if train:
            self.img_anno_pairs = pasc_anno_fnames_train_val[0]
        else:
            self.img_anno_pairs = pasc_anno_fnames_train_val[1]
            # print("Saving validation images")
            # with open('val_ims.txt', 'w') as f:
            #     for l in self.img_anno_pairs:
            #         f.write("{}\n".format(l))

        network_dims.update(label_attrs)

    def __len__(self):

        return len(self.img_anno_pairs)

    def __getitem__(self, index):

        img_path, annotation_path = self.img_anno_pairs[index]
        # print(img_path, annotation_path)

        _img = Image.open(img_path).convert('RGB')

        # TODO: maybe can be done in a better way
        _semantic_target = Image.open(annotation_path)

        imid = os.path.splitext(os.path.split(img_path)[-1])[0]

        point_annotations = self.point_annotations[imid] if imid in self.point_annotations else {
        }

        _point_target = get_point_mask(
            point_annotations,
            self.mask_type,
            np.array(_semantic_target).shape)

        _point_target = Image.fromarray(_point_target)

        if self.joint_transform is not None:
            _img, _semantic_target, _point_target = self.joint_transform(
                [_img, _semantic_target, _point_target])

        return _img, _semantic_target, _point_target  # , _weights

    def _download_dataset(self):

        # Add a progress bar for the download
        def _progress(count, block_size, total_size):

            progress_string = "\r>> {:.2%}".format(
                float(count * block_size) / float(total_size))
            sys.stdout.write(progress_string)
            sys.stdout.flush()

        # Create the root folder with all the intermediate
        # folders if it doesn't exist yet.
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # TODO: factor this out into separate function because we repeat
        # same operation two times
        if os.path.isfile(self.pascal_tar_full_download_filename):

            print('\n PASCAL VOC segmentation dataset file already exists')
        else:

            print(
                "\n Downloading PASCAL VOC segmentation dataset to {}".format(
                    self.pascal_tar_full_download_filename))
            urllib.request.urlretrieve(
                self.PASCAL_URL,
                self.pascal_tar_full_download_filename,
                _progress)

        if os.path.isfile(self.berkeley_tar_full_download_filename):

            print('\n Berkeley segmentation dataset file already exists')
        else:

            print(
                "\n Downloading Berkeley segmentation additional dataset to {}".format(
                    self.berkeley_tar_full_download_filename))
            urllib.request.urlretrieve(
                self.BERKELEY_URL,
                self.berkeley_tar_full_download_filename,
                _progress)

    def _extract_tar_to_the_root_folder(self, tar_full_filename):
        # TODO: change to with: statement instead
        with tarfile.open(tar_full_filename) as tf:
            tf.extractall(path=self.root)
        # tar_obj = tarfile.open(tar_full_filename)
        #
        # tar_obj.extractall(path=self.root)
        #
        # tar_obj.close()

    def _extract_dataset(self):

        print(
            "\n Extracting PASCAL VOC segmentation dataset to {}".format(
                self.pascal_full_root_folder_path))
        self._extract_tar_to_the_root_folder(
            self.pascal_tar_full_download_filename)

        print(
            "\n Extracting Berkeley segmentation dataset to {}".format(
                self.berkeley_full_root_folder_path))
        self._extract_tar_to_the_root_folder(
            self.berkeley_tar_full_download_filename)

    def _prepare_dataset(self):

        print("\n Converting .mat files in the Berkeley dataset to pngs")

        convert_pascal_berkeley_augmented_mat_annotations_to_png(
            self.berkeley_full_root_folder_path)
