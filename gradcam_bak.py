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
import json


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


def process_visual(path_img, path_dir, vqa_model="minhmul_noatt_train_2048"):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    visual_PIL = Image.open(path_img)
    visual_tensor = transform(visual_PIL)
    visual_data = torch.FloatTensor(1, 3,
                                    visual_tensor.size(1),
                                    visual_tensor.size(2))
    visual_data[0][0] = visual_tensor[0]
    visual_data[0][1] = visual_tensor[1]
    visual_data[0][2] = visual_tensor[2]
    print('visual', visual_data.size(), visual_data.mean())

    visual_data = visual_data.cuda()
    visual_input = Variable(visual_data, volatile=True)

    cnn = load_image_model(path_dir)

    visual_features = cnn(visual_input)
    if 'noatt' in vqa_model:
        nb_regions = visual_features.size(2) * visual_features.size(3)
        visual_features = visual_features.sum(
            3).sum(2).div(nb_regions).view(-1, 2048)
    return visual_features


def process_question(question_str):
    question_tokens = tokenize_mcb(question_str)
    question_data = torch.LongTensor(1, len(question_tokens))
    for i, word in enumerate(question_tokens):
        if word in trainset.word_to_wid:
            question_data[0][i] = trainset.word_to_wid[word]
        else:
            question_data[0][i] = trainset.word_to_wid['UNK']
    if args.cuda:
        question_data = question_data.cuda(async=True)
    question_input = Variable(question_data, volatile=True)
    print('question', question_str, question_tokens, question_data)

    return question_input


def process_answer(answer_var, trainset, model):
    answer_sm = torch.nn.functional.softmax(Variable(answer_var.data[0].cpu()))
    max_, aid = answer_sm.topk(5, 0, True, True)
    ans = []
    val = []
    for i in range(5):
        ans.append(trainset.aid_to_ans[aid.data[i]])
        val.append(max_.data[i])
    answer = {'ans': ans, 'val': val}
    return answer, answer_sm


def load_image_model(path_dir):
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

    filename = path_dir + 'data/image_models/best_resnet152_crossentropyloss_breast.pth.tar'

    model = models.resnet152()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
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


def load_dict_torch_031(model, path_ckpt):
    import torch._utils
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(
                storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    model_dict = torch.load(path_ckpt)
    model_dict_clone = model_dict.copy()  # We can't mutate while iterating
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    model.load_state_dict(model_dict, False)
    return model


def get_model_vqa(vqa_model="minhmul_noatt_train_2048"):
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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isfile(path_ckpt_model):
        # model_state = torch.load(path_ckpt_model, map_location=device)
        # style_model.load_state_dict(torch.load(args.model))

        # model_state = torch.load(path_ckpt_model)
        # model.load_state_dict(model_state)

        model = load_dict_torch_031(model, path_ckpt_model)

    return model


def get_data(vqa_model="minhmul_noatt_train_2048"):
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

    data = []

    dataloader_iterator = iter(train_loader)
    for i in range(5):
        try:
            sample = next(dataloader_iterator)
            data.append(sample)
        except:
            print("something wrong")

    return data, trainset


def process_vqa(vqa_model="minhmul_noatt_train_2048", finalconv_name="linear_classif"):
    model = get_model_vqa(vqa_model=vqa_model)
    net = model
    print(model)
    print(model._modules)
    params = list(model.parameters())

    if "noatt" in vqa_model:
        classif_w_params = np.squeeze(params[10].data.numpy())
        classif_b_params = np.squeeze(params[11].data.numpy())

    data, trainset = get_data(vqa_model)

    net.eval()
    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    sample = data[0]

    input_visual = Variable(sample['visual'])
    input_question = Variable(sample['question'])
    target_answer = Variable(sample['answer'])

    answer, answer_sm = process_answer(
        net(input_visual, input_question), trainset, net)

    return classif_w_params, classif_b_params


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


def example_process_image():
    laptop_path = "C:/Users/minhm/Documents/GitHub/vqa_idrid/"
    desktop_path = "/home/minhvu/github/vqa_idrid/"

    is_laptop = False
    if is_laptop:
        path_dir = laptop_path
        ext = "*.jpg"
    else:
        path_dir = desktop_path
        ext = "*.png"

    path = path_dir + "temp/test/"
    img_dirs = glob.glob(os.path.join(path, ext))

    for path in img_dirs:
        net = load_image_model(path_dir)
        finalconv_name = "layer4"
        process_image(path, net, finalconv_name)


if __name__ == '__main__':
    example_process_image()

    # process_vqa("minhmul_att_train")
    # process_vqa("minhmul_noatt_train")

    # process_vqa("minhmul_noatt_train_2048")
