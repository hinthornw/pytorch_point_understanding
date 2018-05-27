'''Training various models on on Pascal Object-Part task
    author: William Hinthorn
    Loosely based on Daniil Pakhomov's (warmspringwinds)
    training code for semantic segmentation.

'''
# flake8: noqa = E402
# pylint: disable = fixme,wrong-import-position,unused-import,import-error,too-many-statements,too-many-locals,
# pylint: disable = invalid-name, len-as-condition, too-many-branches

import sys
import os
import argparse
import datetime
import tqdm
import torch
import torch.autograd
from torch.autograd import Variable
from graphviz import Digraph
from torchviz import make_dot

import tensorboardX

# from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix

# Allow for access to the modified torchvision library and
# various datasets
PATH = os.path.dirname(os.path.realpath(__file__))
PATHARR = PATH.split(os.sep)
print(PATHARR)
HOME_DIR = os.path.join(
    '/', *PATHARR[:PATHARR.index('obj_part_segmentation') + 1])
# vision_dir = os.path.join(HOME_DIR, 'vision')
DATASET_DIR = os.path.join(HOME_DIR, 'datasets')
sys.path.insert(0, HOME_DIR)
# sys.path.insert(0, vision_dir)
# import obj_part_segmentation
import pytorch_segmentation
from pytorch_segmentation.evaluation import (
    poly_lr_scheduler,
    get_training_loaders,
    get_valid_logits,
    get_valid_annos,
    numpyify_logits_and_annotations,
    outputs_tonp_gt,
    compress_objpart_logits,
    get_iou,
    validate_batch,
    get_precision_recall,
    save_checkpoint,
    validate_and_output_images,
    get_network_and_optimizer)


def str2bool(v):
    ''' Helper for command line args.
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    '''
    main(): Primary function to manage the training.
    '''

    print(args)
    # *************************************
    architecture = args.arch
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    load_model_name = args.load_model_name
    num_workers = args.num_workers
    which_vis_type = args.paint_images
    print(which_vis_type)
    assert which_vis_type in [None, 'None', 'objpart', 'semantic', 'separated']
    merge_level = args.part_grouping
    assert merge_level in ['binary', 'sparse', 'merged']
    mask_type = args.mask_type
    assert mask_type in ['mode', 'consensus']
    device = args.device
    _start_epoch = args.origin_epoch
    _load_folder = args.load_folder

    # **************************************
    time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    # Define training parameters
    # edit below
    # *******************************************************
    experiment_name = experiment_name + time_now  # "drn_objpart_" + time_now
    # merge_level = 'binary'  # 'merged', 'sparse'
    load_model_path = os.path.join(
        HOME_DIR, 'pytorch_segmentation', 'training', _load_folder)
    if load_model_name is not None:
        load_model_path = os.path.join(load_model_path, load_model_name)

    # End define training parameters
    # **********************************************************

    print("Setting visible GPUS to machine {}".format(device))

    # Use second GPU -pytorch-segmentation-detection- change if you want to
    # use a first one
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    print("=> Getting training and validation loaders")
    print(DATASET_DIR)
    network_dims = {}
    (_, _), (valset_loader, _) = get_training_loaders(
        DATASET_DIR, network_dims, batch_size,
        num_workers, mask_type, merge_level)

    number_of_classes = 1  # 1 for background...
    print(network_dims)
    for k in network_dims:
        number_of_classes += 1 + network_dims[k]
    print("Training for {} objpart categories".format(
        number_of_classes))
    print("=> Creating network and optimizer")
    train_params = {}
    net, _ = get_network_and_optimizer(
        architecture,
        network_dims,
        load_model_name is not None, load_model_path, train_params)

    print("=> Validating network")
    validate(net, valset_loader)



def validate(net, loader):
    ''' Computes mIoU for``net`` over the a set.
        args::
            :param ``net``: network (in this case resnet34_8s_dilated
            :param ``loader``: dataloader (in this case, validation set loader)

        returns the mIoU (ignoring classes where no parts were annotated) for
        the semantic segmentation task and object-part inference task.
    '''

    net.eval()

    dotfile = None
    for _, (image, _, _) in enumerate(tqdm.tqdm(loader)):
        image = Variable(image.cuda())
        _, semantic_logits = net(image)
        dotfile = make_dot(semantic_logits.mean(), params=dict(net.named_parameters()))
        break

    print(dotfile.source)
    dotfile.format = 'svg'
    dotfile.render('network_graph.gv', view=True)



if __name__ == "__main__":
    # TODO: move parameters from hard_coded in fn to command_line variables

    parser = argparse.ArgumentParser(
        description='Hyperparameters for training.')
    parser.add_argument('-f', '--validate-batch-frequency',
                        metavar='F', type=int,
                        default=50,
                        help='Check training outputs every F batches.')
    parser.add_argument('-s', '--save-model-name', required=True,
                        type=str, help='prefix for model checkpoing')
    parser.add_argument('-l', '--load-model-name', default=None,
                        type=str, help="prefix for model checkpoint"
                        "to load")
    parser.add_argument('-p', '--paint-images', type=str, default=None,
                        help="Type of masked images to output before"
                        "training (for debugging). One of:"
                        "['objpart', 'semantic', 'separated', 'None']")
    parser.add_argument('-v', '--validate-first', type=str2bool, default=False,
                        help="Whether or not to validate the loaded model"
                        "before training")
    parser.add_argument('-d', '--device', type=str, default='0',
                        help="Which CUDA capable device on which to train")
    parser.add_argument('-w', '--num-workers', type=int, default=4,
                        help="Number of workers for the dataloader.")
    parser.add_argument('-e', '--epochs-to-train', type=int, default=10,
                        help="Number of epochs to train this time around.")
    parser.add_argument('-o', '--origin-epoch', type=int, required=False,
                        help="Epoch from which to originate. (default is "
                        "from checkpoint)")
    parser.add_argument('-m', '--mask-type', type=str, default="mode",
                        help="How to select the valid points for supervision."
                        " One of 'mode' or 'consensus'. ")
    parser.add_argument('-b', '--batch-size', type=int, default=12,
                        help="Number of inputs to process in each batch")
    parser.add_argument('-a', '--arch', '--architecture', type=str,
                        default='resnet', help='Which model to use')
    parser.add_argument('-n', '--experiment-name', type=str,
                        default='op', help='Experiment name for checkpointing')
    parser.add_argument(
        '-g',
        '--part-grouping',
        type=str,
        default='binary',
        help="Whether to predict for all parts, in part groups,"
        "or just for a binary obj/part task (for each class)."
        "Legal values are 'binary', 'merged', and 'sparse'")
    parser.add_argument('--load-folder', type=str, default='experiments')
    argvals = parser.parse_args()
    main(argvals)
