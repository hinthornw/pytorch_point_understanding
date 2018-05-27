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
DATASET_DIR = os.path.join(HOME_DIR, 'datasets')
sys.path.insert(0, HOME_DIR)

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
    validate_first = args.validate_first
    save_model_name = args.save_model_name
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
    epochs_to_train = args.epochs_to_train
    validate_batch_frequency = args.validate_batch_frequency
    _start_epoch = args.origin_epoch
    _load_folder = args.load_folder

    # **************************************
    time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    # Define training parameters
    # edit below
    # *******************************************************
    experiment_name = experiment_name + time_now  # "drn_objpart_" + time_now
    # architecture = ['resnet', 'drn', 'vgg', 'hourglass'][0]
    init_lr = 0.0016  # 2.5e-4
    # batch_size = 12  # 8
    # num_workers = 4
    # number_of_classes = 41
    number_of_semantic_classes = 21  # Pascal VOC
    semantic_labels = range(number_of_semantic_classes)
    #  validate_first = False
    #  output_predicted_images = False
    # which_vis_type = ['None' ,'objpart', 'semantic', 'separated'][0]
    if which_vis_type is None or which_vis_type == 'None':
        output_predicted_images = False
    # merge_level = 'binary'  # 'merged', 'sparse'
    # save_model_name = "drn"  # 'resnet_34_8s'
    save_model_folder = os.path.join(
        HOME_DIR,
        'pytorch_segmentation',
        'training',
        'experiments')
    load_model_path = os.path.join(
        HOME_DIR, 'pytorch_segmentation', 'training', _load_folder)
    if load_model_name is not None:
        # load_model_name = 'resnet_34_8s_model_best.pth.tar'
        # load_model_path = os.path.join(
        #     HOME_DIR, 'pytorch_segmentation', 'training', 'models', load_model_name)
        # load_model_name = 'drn_model_best.pth.tar'
        load_model_path = os.path.join(load_model_path, load_model_name)

    # iter_size = 20
    # epochs_to_train = 40
    # mask_type = "mode"  # "consensus"
    # device = '2'  # could be '0', '1', '2', or '3' on visualai#
    # validate_batch_frequency = 20  # compute mIoU every k batches
    # End define training parameters
    # **********************************************************

    print("Setting visible GPUS to machine {}".format(device))

    # Use second GPU -pytorch-segmentation-detection- change if you want to
    # use a first one
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    print("=> Getting training and validation loaders")
    print(DATASET_DIR)
    network_dims = {}
    (trainloader, trainset), (valset_loader, _) = get_training_loaders(
        DATASET_DIR, network_dims, batch_size,
        num_workers, mask_type, merge_level)

    number_of_classes = 1  # 1 for background...
    print(network_dims)
    for k in network_dims:
        number_of_classes += 1 + network_dims[k]
    objpart_labels = range(number_of_classes)
    print("Training for {} objpart categories".format(
        number_of_classes))
    print("=> Creating network and optimizer")
    train_params = {}
    net, optimizer = get_network_and_optimizer(
        architecture,
        network_dims,
        load_model_name is not None, load_model_path, train_params, init_lr)
    try:
        train_params['best_op_val_score'] = train_params['best_objpart_val_score']
        train_params['best_sem_val_score'] = train_params['best_semantic_val_score']
        # train_params['best_objpart_accuracy'] = 0.5
    except KeyError:
        print(list(train_params))
    writer = tensorboardX.SummaryWriter(os.path.join('runs', experiment_name))
    if _start_epoch is not None:
        train_params['start_epoch'] = _start_epoch
    train_params.update({
        'net': net,
        'optimizer': optimizer,
        'epochs_to_train': epochs_to_train,
        'trainloader': trainloader,
        'trainset': trainset,
        'valset_loader': valset_loader,
        'init_lr': init_lr,
        'writer': writer,
        'validate_batch_frequency': validate_batch_frequency,
        'number_of_classes': number_of_classes,
        'number_of_semantic_classes': number_of_semantic_classes,
        'save_model_name': save_model_name,
        'save_model_folder': save_model_folder
    })

    op_map = net.flat_map

    if output_predicted_images:
        print("=> Outputting predicted images to folder 'predictions'")
        validate_and_output_images(
            net, valset_loader, op_map, which=which_vis_type, alpha=0.7)
        while True:
            resp = input("=> Done. Do you wish to continue training? (y/n):\t")
            if resp[0] == 'n':
                return
            elif resp[0] == 'y':
                break
            else:
                print("{} not understood".format(resp))

    if validate_first:
        print("=> Validating network")
        sc1, sc2, sc3 = validate(
            net, valset_loader, (objpart_labels, semantic_labels))
        print("{}\t{}\t{}".format(sc1, sc2, sc3))

    print("=> Entering training function")
    train(train_params)
    print('=> Finished Training')
    writer.close()


def validate(net, loader, labels):
    ''' Computes mIoU for``net`` over the a set.
        args::
            :param ``net``: network (in this case resnet34_8s_dilated
            :param ``loader``: dataloader (in this case, validation set loader)

        returns the mIoU (ignoring classes where no parts were annotated) for
        the semantic segmentation task and object-part inference task.
    '''

    net.eval()
    # hardcoded in for the object-part infernce
    # TODO: change to be flexible/architecture-dependent
    no_parts = [0, 4, 9, 11, 18, 20, 24, 29, 31, 38, 40]
    objpart_labels, semantic_labels = labels

    overall_sem_conf_mat = None
    overall_part_conf_mat = None
    gt_total = 0.0
    gt_right = 0.0
    showevery = 50
    # valset_loader
    for i, (image, semantic_anno, objpart_anno) in enumerate(
            tqdm.tqdm(loader)):
        image = Variable(image.cuda())
        objpart_logits, semantic_logits = net(image)

        # First we do argmax on gpu and then transfer it to cpu
        sem_pred_np, sem_anno_np = numpyify_logits_and_annotations(
            semantic_logits, semantic_anno)
        # objpart_pred_np, objpart_anno_np = numpyify_logits_and_annotations(
        #     objpart_logits, objpart_anno)
        objpart_pred_np, objpart_anno_np = outputs_tonp_gt(
            objpart_logits, objpart_anno, net.flat_map)
        opprednonz = np.array([el for el in objpart_pred_np if el > 0])
        opannononz = np.array([el for el in objpart_anno_np if el > 0])
        gt_right += (opprednonz == opannononz).sum()
        gt_total += len(opannononz)

        current_semantic_confusion_matrix = confusion_matrix(
            y_true=sem_anno_np, y_pred=sem_pred_np, labels=semantic_labels)

        if overall_sem_conf_mat is None:
            overall_sem_conf_mat = current_semantic_confusion_matrix
        else:
            overall_sem_conf_mat += current_semantic_confusion_matrix

        if (objpart_anno_np > 0).sum() == 0:
            continue
        current_objpart_conf_mat = confusion_matrix(
            y_true=objpart_anno_np, y_pred=objpart_pred_np,
            labels=objpart_labels)
        if overall_part_conf_mat is None:
            overall_part_conf_mat = current_objpart_conf_mat
        else:
            overall_part_conf_mat += current_objpart_conf_mat
        if i % showevery == 1:
            tqdm.tqdm.write(
                "Object-part accuracy ({}):\t{:%}".format(i, gt_right / gt_total))

    # Semantic segmentation task
    semantic_IoU = get_iou(
        overall_sem_conf_mat)

    semantic_mIoU = np.mean(
        semantic_IoU)

    objpart_prec, objpart_rec = get_precision_recall(
        overall_part_conf_mat)
    tqdm.tqdm.write(
        "precision/recall:\t\n{}\n\n{}\n".format(
            [
                objpart_prec[i] for i, _ in enumerate(objpart_prec) if i not in no_parts], [
                    objpart_rec[i] for i, _ in enumerate(objpart_rec) if i not in no_parts]))

    # Part segmentation task
    objpart_IoU = get_iou(overall_part_conf_mat)
    objpart_mIoU = np.mean([objpart_IoU[i] for i,
                            _ in enumerate(objpart_IoU) if i not in no_parts])
    overall_objpart_accuracy = gt_right / gt_total
    net.train()
    return objpart_mIoU, semantic_mIoU, overall_objpart_accuracy


def train(train_params):
    ''' Main function for training the net.
    '''
    net = train_params['net']
    optimizer = train_params['optimizer']
    start_epoch = train_params['start_epoch']
    epochs_to_train = train_params['epochs_to_train']
    trainloader = train_params['trainloader']
    # trainset = train_params['trainset']
    # batch_size = train_params['batch_size']
    valset_loader = train_params['valset_loader']
    init_lr = train_params['init_lr']
    writer = train_params['writer']
    validate_batch_frequency = train_params['validate_batch_frequency']
    best_op_val_score = train_params['best_op_val_score']
    best_op_gt_valscore = train_params['best_objpart_accuracy']
    best_op_gt_valscore = 0.0
    best_sem_val_score = train_params['best_sem_val_score']
    semantic_criterion = train_params['semantic_criterion']
    objpart_criterion = train_params['objpart_criterion']
    number_of_classes = train_params['number_of_classes']
    number_of_semantic_classes = train_params['number_of_semantic_classes']
    save_model_folder = train_params['save_model_folder']
    save_model_name = train_params['save_model_name']
    # could try to learn these as parameters...
    # currently not implemented
    objpart_weight = Variable(torch.Tensor([1])).cuda()
    semantic_weight = Variable(torch.Tensor([1])).cuda()
    _one_weight = Variable(torch.Tensor([1])).cuda()

    objpart_labels = range(number_of_classes)
    semantic_labels = range(number_of_semantic_classes)

    spatial_average = False  # True
    batch_average = True

    # loop over the dataset multiple times
    print(
        "Training from epoch {} for {} epochs".format(
            start_epoch,
            epochs_to_train))
    sz = None
    number_training_batches = len(trainloader)
    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs_to_train)):
        # semantic_running_loss = 0.0
        # objpart_running_loss = 0.0
        poly_lr_scheduler(optimizer, init_lr, epoch,
                          lr_decay_iter=1, max_iter=100, power=0.9)
        overall_sem_conf_mat = None
        overall_part_conf_mat = None
        correct_points = 0.0
        num_points = 0.0
        tqdm.tqdm.write("=> Starting epoch {}".format(epoch))
        tqdm.tqdm.write("=> Current time:\t{}".format(datetime.datetime.now()))
        for i, data in tqdm.tqdm(
                enumerate(
                    trainloader, 0), total=number_training_batches):
            # img, semantic_anno, objpart_anno, objpart_weights=data
            img, semantic_anno, objpart_anno = data
            batch_size = img.size(0)

            if sz is None:
                sz = np.prod(img.size())

            # We need to flatten annotations and logits to apply index of valid
            # annotations. All of this is because pytorch doesn't have
            # tf.gather_nd()
            semantic_anno_flatten_valid, semantic_index = get_valid_annos(
                semantic_anno, 255)
            op_anno_flt_vld, objpart_index = get_valid_annos(
                objpart_anno, -1)

            # wrap them in Variable
            # the index can be acquired on the gpu
            img, semantic_anno_flatten_valid, semantic_index = Variable(
                img.cuda()), Variable(
                    semantic_anno_flatten_valid.cuda()), Variable(
                        semantic_index.cuda())
            op_anno_flt_vld, objpart_index = Variable(
                op_anno_flt_vld.cuda()), Variable(
                    objpart_index.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()
            # adjust_learning_rate(optimizer, loss_current_iteration)

            # forward + backward + optimize
            objpart_logits, semantic_logits = net(img)
            # import pdb; pdb.set_trace()

            op_log_flt_vld = get_valid_logits(
                objpart_logits, objpart_index, number_of_classes)
            semantic_logits_flatten_valid = get_valid_logits(
                semantic_logits, semantic_index, number_of_semantic_classes)
            # Compress objpart logits reduces the problem to a binary inference
            #
            # op_log_flt_vld, op_anno_flt_vld = compress_objpart_logits(
            #     op_log_flt_vld, op_anno_flt_vld)

            # Score the overall accuracy
            # prepend _ to avoid interference with the training
            _op_log_flt_vld, _op_anno_flt_vld = compress_objpart_logits(
                op_log_flt_vld, op_anno_flt_vld, net.flat_map)

            if len(_op_log_flt_vld) > 0:
                _, logits_toscore = torch.max(_op_log_flt_vld, dim=1)
                _ind_ = torch.gt(_op_anno_flt_vld, 0)
                op_pred_gz = logits_toscore[_ind_]
                _op_anno_gz = _op_anno_flt_vld[_ind_]
                correct_points += float((op_pred_gz == _op_anno_gz).sum())
                num_points += len(_op_anno_gz)

                # Balance the weights
                this_num_points = float(len(_ind_))
                op_scale = Variable(torch.Tensor([this_num_points])).cuda() # _one_weight
                # if num_points != 0:
                #     multiplier = Variable(torch.Tensor(
                #         [sz / this_num_points])).cuda()

                # Compute cross-entropy loss for the object-part inference task
                objpart_loss = objpart_criterion(
                    op_log_flt_vld, op_anno_flt_vld)
            else:
                objpart_loss = Variable(torch.Tensor([0])).cuda()
                op_scale = objpart_weight


            sem_scale = Variable(torch.FloatTensor(
                [(semantic_anno_flatten_valid.data > 0).sum()])).cuda()
            semantic_loss = semantic_criterion(
                semantic_logits_flatten_valid, semantic_anno_flatten_valid)

            # import pdb; pdb.set_trace()
            if spatial_average:
                semantic_loss /= sem_scale
                objpart_loss /= op_scale

            # tqdm.tqdm.write("oploss, semloss:\t({}, {}) - {} - {:%}".format(
            # objpart_loss.data[0], semantic_loss.data[0], len(_ind_),
            # correct_points/num_points))

            # TODO: Consider clipping??
            # Consider modulating the weighting of the losses?
            semantic_batch_weight = semantic_weight
            objpart_batch_weight = objpart_weight
            loss = semantic_loss * semantic_batch_weight + \
            objpart_loss * objpart_batch_weight

            if batch_average:
                loss = loss / batch_size

            writer.add_scalar(
                'losses/semantic_loss',
                semantic_loss.data[0] /
                semantic_logits_flatten_valid.size(0),
                number_training_batches * epoch + i)

            if len(op_log_flt_vld) > 0:
                writer.add_scalar(
                    'losses/objpart_loss',
                    objpart_loss.data[0] /
                    op_log_flt_vld.size(0),
                    number_training_batches * epoch + i)

            loss.backward()
            optimizer.step()
            if i % validate_batch_frequency == 1:
                valout = validate_batch(
                    (objpart_logits, objpart_anno),
                    (semantic_logits, semantic_anno),
                    overall_part_conf_mat, overall_sem_conf_mat,
                    (objpart_labels, semantic_labels), net.flat_map,
                    writer,
                    number_training_batches * epoch + i)
                ((objpart_mPrec, objpart_mRec),
                 semantic_mIoU,
                 overall_part_conf_mat,
                 overall_sem_conf_mat) = valout
                # tqdm.tqdm.write("OP Acc ({}):\t{:%}".format(i, correct_points/num_points))
                if num_points > 0:
                    writer.add_scalar('data/obj_part_accuracy',
                                      correct_points / num_points,
                                      number_training_batches * epoch + i)

                writer.add_scalar('data/semantic_mIoU',
                                  semantic_mIoU,
                                  number_training_batches * epoch + i)

                if objpart_mPrec is not None:

                    writer.add_scalar('data/obj_part_mPrec',
                                      objpart_mPrec,
                                      number_training_batches * epoch + i)

                    writer.add_scalar('data/obj_part_mRec',
                                      objpart_mRec,
                                      number_training_batches * epoch + i)

                correct_points = 0.0
                num_points = 0.0

        # Validate and save if best model
        (curr_op_valscore,
         curr_sem_valscore,
         curr_op_gt_valscore) = validate(
             net, valset_loader, (objpart_labels, semantic_labels))
        writer.add_scalar('validation/semantic_validation_score',
                          curr_sem_valscore, epoch)
        writer.add_scalar('validation/objpart_validation_score',
                          curr_op_valscore, epoch)
        writer.add_scalar('validation/overall_objpart_validation_score',
                          curr_op_valscore, epoch)
        is_best = False
        if curr_op_gt_valscore > best_op_gt_valscore:
            best_op_gt_valscore = curr_op_gt_valscore
            is_best = True
        if curr_op_valscore > best_op_val_score:
            best_op_val_score = curr_op_valscore
            # is_best = True
        if curr_sem_valscore > best_sem_val_score:
            best_sem_val_score = curr_sem_valscore
            # is_best = True

        # label as best IFF beats best obj-part inference score
        # Allow for equality (TODO check)
        tqdm.tqdm.write(
            "This epochs scores:\n\tSemantic:\t{}\n\tmOP:\t{}\n\tOP:\t{}".format(
                curr_sem_valscore, curr_op_valscore, curr_op_gt_valscore))
        tqdm.tqdm.write("Current Best Semantic Validation Score:\t{}".format(
            best_sem_val_score))
        tqdm.tqdm.write("Current Best objpart Validation Score:\t{}".format(
            best_op_val_score))
        tqdm.tqdm.write("Best  Overall objpart Validation Score:\t{}".format(
            best_op_gt_valscore))
        writer.export_scalars_to_json("./all_scalars.json")

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': save_model_name,
            'state_dict': net.state_dict(),
            'best_semantic_mIoU': best_sem_val_score,
            'best_objpart_mIoU': best_op_val_score,
            'best_objpart_accuracy': best_op_gt_valscore,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder=save_model_folder,
                        filename='{}_epoch_{}.pth.tar'.format(save_model_name, epoch))


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
