import yaml
import vqa.lib.utils as vqa_utils
import vqa.datasets as datasets
import vqa.models as models_vqa
import datasets.utils.paths_utils as paths_utils
from torchsummary import summary
import argparse
import glob
from torch.autograd import Function
import torch
from collections import OrderedDict
import pdb
import cv2
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import models, transforms, utils
from PIL import Image
import requests
import io
import os
import sys


parser = argparse.ArgumentParser(
    description='Train/Evaluate models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
##################################################
# yaml options file contains all default choices #
parser.add_argument('--path_opt', default='options/breast/default.yaml', type=str,
                    help='path to a yaml options file')
################################################
# change cli options to modify default choices #
# logs options
parser.add_argument('--dir_logs', type=str, help='dir logs')
# data options
parser.add_argument('--vqa_trainsplit', type=str,
                    choices=['train', 'trainval'], default="train")
# model options
parser.add_argument('--arch', choices=models_vqa.model_names,
                    help='vqa model architecture: ' +
                    ' | '.join(models_vqa.model_names))
parser.add_argument('--st_type',
                    help='skipthoughts type')
parser.add_argument('--st_dropout', type=float)
parser.add_argument('--st_fixed_emb', default=None, type=vqa_utils.str2bool,
                    help='backprop on embedding')
# optim options
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='initial learning rate')
parser.add_argument('-b', '--batch_size', type=int,
                    help='mini-batch size')
parser.add_argument('--epochs', type=int,
                    help='number of total epochs to run')
# options not in yaml file
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint')
parser.add_argument('--save_model', default=True, type=vqa_utils.str2bool,
                    help='able or disable save model and optim state')
parser.add_argument('--save_all_from', type=int,
                    help='''delete the preceding checkpoint until an epoch,'''
                         ''' then keep all (useful to save disk space)')''')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation and test set')
parser.add_argument('-j', '--workers', default=2, type=int,
                    help='number of data loading workers')
parser.add_argument('--print_freq', '-p', default=2, type=int,
                    help='print frequency')
################################################
parser.add_argument('-ho', '--help_opt', dest='help_opt', action='store_true',
                    help='show selected options before running')


def load_image_model():
    def rename_key(state_dict):
        old_keys_list = state_dict.keys()
        for old_key in old_keys_list:
            # print(old_key)
            new_key = old_key.replace('module.', '')
            # print(new_key)
            state_dict = update_ordereddict(state_dict, old_key, new_key)
        return state_dict

    def update_ordereddict(state_dict, old_key, new_key):
        new_state_dict = OrderedDict(
            [(new_key, v) if k == old_key else (k, v) for k, v in state_dict.items()])
        return new_state_dict

    filename = 'C:/Users/minhm/Documents/GitHub/vqa_idrid/data/image_models/best_resnet152_crossentropyloss_breast.pth.tar'
    model = models.resnet152()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # PyTorch v0.4.0
    checkpoint = torch.load(filename, map_location=device)
    state_dict = checkpoint['state_dict']
    state_dict = rename_key(state_dict)
    model.load_state_dict(state_dict)

    return model


def get_gadcam_image(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def process_image(path, net, finalconv_name):
    net.eval()

    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_name = paths_utils.get_filename_without_extension(path)
    img_pil = Image.open(path)
    in_path = "temp/{}_in.jpg".format(img_name)
    img_pil.save(in_path)

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    # download the imagenet category list
    classes = {
        0: "Benign",
        1: "InSitu",
        2: "Invasive",
        3: "Normal"
    }

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(0, 4):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = get_gadcam_image(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
    img = cv2.imread(in_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(
        CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    out_path = "temp/{}_out.jpg".format(img_name)
    cv2.imwrite(out_path, result)


def process_vqa_sample(path, net, finalconv_name="linear_classif"):
    net.eval()

    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_name = paths_utils.get_filename_without_extension(path)
    img_pil = Image.open(path)
    in_path = "temp/{}_in.jpg".format(img_name)
    img_pil.save(in_path)

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    # download the imagenet category list
    classes = {
        0: "Benign",
        1: "InSitu",
        2: "Invasive",
        3: "Normal"
    }

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(0, 4):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = get_gadcam_image(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
    img = cv2.imread(in_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(
        CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    out_path = "temp/{}_out.jpg".format(img_name)
    cv2.imwrite(out_path, result)


def get_model_vqa(vqa_model="minhmul_noatt_train"):
    path = "options/breast/{}.yaml".format(vqa_model)
    args = parser.parse_args()
    options = {
        'vqa': {
            'trainsplit': args.vqa_trainsplit
        },
        'logs': {
            'dir_logs': args.dir_logs
        },
        'model': {
            'arch': args.arch,
            'seq2vec': {
                'type': args.st_type,
                'dropout': args.st_dropout,
                'fixed_emb': args.st_fixed_emb
            }
        },
        'optim': {
            'lr': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }
    }
    with open(path, 'r') as handle:
        options_yaml = yaml.load(handle)
    options = vqa_utils.update_values(options, options_yaml)
    if 'vgenome' not in options:
        options['vgenome'] = None

    trainset = datasets.factory_VQA(options['vqa']['trainsplit'],
                                    options['vqa'],
                                    options['coco'],
                                    options['vgenome'])

    model = models_vqa.factory(options['model'],
                               trainset.vocab_words(), trainset.vocab_answers(),
                               cuda=False, data_parallel=False)

    # load checkpoint
    path_ckpt_model = "logs/breast/{}/best_model.pth.tar".format(vqa_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isfile(path_ckpt_model):
        model_state = torch.load(path_ckpt_model, map_location=device)
        model.load_state_dict(model_state)
    return model


def process_vqa(vqa_model="minhmul_noatt_train"):
    model = get_model_vqa(vqa_model=vqa_model)
    print(model)
    print(model._modules)
    params = list(model.parameters())

    classif_w_params = np.squeeze(params[10].data.numpy())
    classif_b_params = np.squeeze(params[11].data.numpy())
    return classif_w_params, classif_b_params


def test_vqa(vqa_model="minhmul_noatt_train"):
    path = "options/breast/{}.yaml".format(vqa_model)
    args = parser.parse_args()
    options = {
        'vqa': {
            'trainsplit': args.vqa_trainsplit
        },
        'logs': {
            'dir_logs': args.dir_logs
        },
        'model': {
            'arch': args.arch,
            'seq2vec': {
                'type': args.st_type,
                'dropout': args.st_dropout,
                'fixed_emb': args.st_fixed_emb
            }
        },
        'optim': {
            'lr': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }
    }
    with open(path, 'r') as handle:
        options_yaml = yaml.load(handle)
    options = vqa_utils.update_values(options, options_yaml)
    if 'vgenome' not in options:
        options['vgenome'] = None

    trainset = datasets.factory_VQA(options['vqa']['trainsplit'],
                                    options['vqa'],
                                    options['coco'],
                                    options['vgenome'])

    train_loader = trainset.data_loader(batch_size=1,
                                        num_workers=0,
                                        shuffle=True)


    # for i in range(10):
    #     for batch in train_loader:
    dataloader_iterator = iter(train_loader)
    for i in range(5):
        try:
            data = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_loader)
            data, target = next(dataloader_iterator)

    # from torch.autograd import Variable
    # for i, sample in enumerate(train_loader):
    #     batch_size = sample['visual'].size(0)

    #     input_visual   = Variable(sample['visual'])
    #     input_question = Variable(sample['question'])
    #     target_answer  = Variable(sample['answer'].cuda(async=True))


    # dataloader_iterator = iter(train_loader)
    # for i in range(5):
    #     try:
    #         data, target = next(dataloader_iterator)
    #     except StopIteration:
    #         dataloader_iterator = iter(train_loader)
    #         data, target = next(dataloader_iterator)
        

    return train_loader


def main():
    img_dirs = glob.glob(os.path.join(
        "C:/Users/minhm/Documents/GitHub/vqa_idrid/temp/test/", "*.jpg"))
    for path in img_dirs:
        net = load_image_model()
        finalconv_name = "layer4"
        process_image(path, net, finalconv_name)


if __name__ == '__main__':
    # main()
    # process_vqa("minhmul_att_train")
    # process_vqa("minhmul_noatt_train")
    process_vqa("minhmul_noatt_train_2048")
    # test_vqa("minhmul_noatt_train")
