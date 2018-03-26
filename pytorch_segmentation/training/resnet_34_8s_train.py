# Training Resnet 34 8s on Pascal Object-Part task
import sys
import os
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
path = os.path.dirname(os.path.realpath(__file__))
patharr = path.split(os.sep)
home_dir = os.path.join(
    '/', *patharr[:patharr.index('obj_part_segmentation') + 1])
# vision_dir = os.path.join(home_dir, 'vision')
dataset_dir = os.path.join(home_dir, 'datasets')
sys.path.insert(0, home_dir)
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
                save_checkpoint,
                validate_and_output_images,
                get_network_and_optimizer)



# pylint: disable=fixme

__config__ = {
    'data_provider': 'data.pascal_part_seg.dp',
    'network': 'models.partsnet.PartsNet',
    'inference': {
        'nstack': 4,
        'inp_dim': 256,
        'oup_dim': 68,
        'num_parts': 17,
        'increase': 128,
        'keys': ['imgs']
    },

    'train': {
        'batchsize': 32,
        # 'batchsize': 16,
        'input_res': 256,  # 512, #reduced for part segmentation
        'output_res': 64,  # 128, #reduced for part segmentation
        'train_iters': 1000,
        'valid_iters': 10,
        'learning_rate': 2e-4,
        'num_loss': 4,

        'loss': [
            ['push_loss', 1e-4],  # 1e-3], #reduced for part segmentation
            ['pull_loss', 1e-4],  # 1e-3], #reduced for part segmentation
            ['detection_loss', 1],  # Mean Squared Error loss
        ],

        'max_num_people': 30,  # TODO: change this
        'num_workers': 2,
        'use_data_loader': True,
    },
}


def main(args):
    '''
    main(): Primary function to manage the training.
    '''
    # partsnet_params = {
    #                     'nstack': 4,
    #                     'inp_dim': 256,
    #                     'oup_dim': 68,
    #                     'num_parts': 17,
    #                     'increase': 128,
    #                     'keys': ['imgs']
    #                     }

    # Define training parameters
    # edit below
    # *******************************************************
    init_lr = 0.0001  # 2.5e-4
    batch_size = 14  # 8
    num_workers = 4
    number_of_classes = 41
    number_of_semantic_classes = 21
    validate_first = True
    output_predicted_images = False
    which_vis_type = 'semantic'  # 'objpart'
    merge_level = 'binary'  # 'merged', 'sparse'
    model_name = ""
    model_path = ""
    load_from_local = True
    if load_from_local:
        model_name = 'resnet_34_8s_model_best.pth.tar'
        model_path = os.path.join(
            home_dir, 'pytorch_segmentation', 'training', 'models', model_name)

    # iter_size = 20
    epochs_to_train = 40
    mask_type = "mode"  # "consensus"
    device = '2'  # could be '0', '1', '2', or '3' on visualai#
    validate_batch_frequency = 20  # compute mIoU every k batches
    # End define training parameters
    # **********************************************************

    objpart_labels = range(number_of_classes)
    semantic_labels = range(number_of_semantic_classes)
    print("Setting visible GPUS to machine {}".format(device))

    # Use second GPU -pytorch-segmentation-detection- change if you want to
    # use a first one
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    print("=> Getting training and validation loaders")
    print(dataset_dir)
    network_dims = {}
    (trainloader, trainset), (valset_loader, valset) = get_training_loaders(
        dataset_dir, network_dims, batch_size,
        num_workers, mask_type, merge_level)
    print("=> Creating network and optimizer")
    train_params = {}
    net, optimizer = get_network_and_optimizer(
        number_of_classes,
        load_from_local, model_path, train_params)
    # net, optimizer = get_network_and_optimizer(number_of_classes,
    # to_aggregate, load_from_local, model_path, train_params)
    writer = tensorboardX.SummaryWriter()
    train_params['start_epoch'] = 0
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
        'objpart_labels': objpart_labels,
        'semantic_labels': semantic_labels,
        # 'validate_first': validate_first
        # 'batch_size': batch_size,
    })

    if output_predicted_images:
        print("=> Outputting predicted images to folder 'predictions'")
        validate_and_output_images(
            net, valset_loader, which=which_vis_type, alpha=0.7)
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
        sc1, sc2 = validate(
            net, valset_loader, (objpart_labels, semantic_labels))
        print("{}\t{}".format(sc1, sc2))

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

    # valset_loader
    for image, semantic_anno, objpart_anno in tqdm.tqdm(loader):
        image = Variable(image.cuda())
        objpart_logits, semantic_logits = net(image)

        # First we do argmax on gpu and then transfer it to cpu
        sem_pred_np, sem_anno_np = numpyify_logits_and_annotations(
            semantic_logits, semantic_anno)
        # objpart_pred_np, objpart_anno_np = numpyify_logits_and_annotations(
        #     objpart_logits, objpart_anno)
        objpart_pred_np, objpart_anno_np = outputs_tonp_gt(
            objpart_logits, objpart_anno)

        current_semantic_confusion_matrix = confusion_matrix(
            y_true=sem_anno_np, y_pred=sem_pred_np, labels=semantic_labels)

        if overall_sem_conf_mat is None:
            overall_sem_conf_mat = current_semantic_confusion_matrix
        else:
            overall_sem_conf_mat += current_semantic_confusion_matrix

        if ((objpart_anno_np > 0).sum() == 0):
            continue
        current_objpart_conf_mat = confusion_matrix(
            y_true=objpart_anno_np, y_pred=objpart_pred_np,
            labels=objpart_labels)
        if overall_part_conf_mat is None:
            overall_part_conf_mat = current_objpart_conf_mat
        else:
            overall_part_conf_mat += current_objpart_conf_mat

    # Semantic segmentation task
    semantic_IoU = get_iou(
        overall_sem_conf_mat)

    semantic_mIoU = np.mean(
        semantic_IoU)

    # Part segmentation task
    objpart_IoU = get_iou(overall_part_conf_mat)
    objpart_mIoU = np.mean([objpart_IoU[i] for i,
                            _ in enumerate(objpart_IoU) if i not in no_parts])
    net.train()
    return objpart_mIoU, semantic_mIoU


def train(train_params):
    net = train_params['net']
    optimizer = train_params['optimizer']
    start_epoch = train_params['start_epoch']
    epochs_to_train = train_params['epochs_to_train']
    trainloader = train_params['trainloader']
    # trainset = train_params['trainset']
    # batch_size=train_params['batch_size']
    valset_loader = train_params['valset_loader']
    init_lr = train_params['init_lr']
    writer = train_params['writer']
    validate_batch_frequency = train_params['validate_batch_frequency']
    best_op_val_score = train_params['best_op_val_score']
    best_sem_val_score = train_params['best_sem_val_score']
    semantic_criterion = train_params['semantic_criterion']
    objpart_criterion = train_params['objpart_criterion']
    number_of_classes = train_params['number_of_classes']
    number_of_semantic_classes = train_params['number_of_semantic_classes']
    objpart_labels = train_params['objpart_labels']
    semantic_labels = train_params['semantic_labels']

    # could try to learn these as parameters...
    # currently not implemented
    objpart_weight = Variable(torch.Tensor([1])).cuda()
    semantic_weight = Variable(torch.Tensor([1])).cuda()
    # loss_current_iteration = 0

    # loop over the dataset multiple times
    print(
        "Training from epoch {} for {} epochs".format(
            start_epoch,
            epochs_to_train))
    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs_to_train)):
        # semantic_running_loss = 0.0
        # objpart_running_loss = 0.0
        poly_lr_scheduler(optimizer, init_lr, epoch,
                          lr_decay_iter=1, max_iter=100, power=0.9)
        overall_sem_conf_mat = None
        overall_part_conf_mat = None
        tqdm.tqdm.write("=> Starting epoch {}".format(epoch))
        for i, data in tqdm.tqdm(
                enumerate(
                    trainloader, 0), total=len(trainloader)):
            # img, semantic_anno, objpart_anno, objpart_weights=data
            img, semantic_anno, objpart_anno = data

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

            op_log_flt_vld = get_valid_logits(
                objpart_logits, objpart_index, number_of_classes)
            semantic_logits_flatten_valid = get_valid_logits(
                semantic_logits, semantic_index, number_of_semantic_classes)

            op_log_flt_vld, op_anno_flt_vld = compress_objpart_logits(
                op_log_flt_vld, op_anno_flt_vld)

            # Compute cross-entropy loss for the object-part inference task
            objpart_loss = objpart_criterion(
                op_log_flt_vld, op_anno_flt_vld)
            semantic_loss = semantic_criterion(
                semantic_logits_flatten_valid, semantic_anno_flatten_valid)

            # TODO: Consider clipping??
            # Consider modulating the weighting of the losses?g
            semantic_batch_weight = semantic_weight
            objpart_batch_weight = objpart_weight
            loss = semantic_loss * semantic_batch_weight + \
                objpart_loss * objpart_batch_weight
            writer.add_scalar(
                'losses/semantic_loss',
                semantic_loss.data[0] /
                semantic_logits_flatten_valid.size(0),
                i)
            writer.add_scalar(
                'losses/objpart_loss',
                objpart_loss.data[0] /
                op_log_flt_vld.size(0),
                i)

            loss.backward()
            optimizer.step()
            if i % validate_batch_frequency == 1:
                valout = validate_batch(
                    (objpart_logits, objpart_anno),
                    (semantic_logits, semantic_anno),
                    overall_part_conf_mat, overall_sem_conf_mat,
                    (objpart_labels, semantic_labels), writer, i)
                ((objpart_mPrec, objpart_mRec),
                 semantic_mIoU,
                 overall_part_conf_mat,
                 overall_sem_conf_mat) = valout

        # Validate and save if best model
        curr_op_valscore, curr_sem_valscore = validate(
            net, valset_loader, (objpart_labels, semantic_labels))
        writer.add_scalar('validation/semantic_validation_score',
                          curr_sem_valscore, epoch)
        writer.add_scalar('validation/objpart_validation_score',
                          curr_op_valscore, epoch)
        is_best = False
        if curr_op_valscore > best_op_val_score:
            best_op_val_score = curr_op_valscore
            is_best = True
        if curr_sem_valscore > best_sem_val_score:
            best_sem_val_score = curr_sem_valscore
            # is_best = True

        # label as best IFF beats best obj-part inference score
        # Allow for equality (TODO check)
        tqdm.tqdm.write("Current Best Semantic Validation Score:\t{}".format(
            best_sem_val_score))
        tqdm.tqdm.write("Current Best objpart Validation Score:\t{}".format(
            best_op_val_score))
        writer.export_scalars_to_json("./all_scalars.json")

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'resnet_34_8s',
            'state_dict': net.state_dict(),
            'best_semantic_mIoU': best_sem_val_score,
            'best_objpart_mIoU': best_op_val_score,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder='models',
                        filename='resnet_34_8s_epoch_{}.pth.tar'.format(epoch))


if __name__ == "__main__":
    # TODO: move parameters from hard_coded in fn to command_line variables
    args = ""
    main(args)
