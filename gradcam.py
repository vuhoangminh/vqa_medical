import torch._utils
import yaml
import vqa.lib.utils as utils
import vqa.datasets as datasets
import vqa.models as models_vqa
import datasets.utils.paths_utils as paths_utils
import argparse
import glob
from torch.autograd import Function
import torch
from collections import OrderedDict
import cv2
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import requests
import io
import os
import sys
import json
import vqa.models.convnets as convnets
import vqa.models.convnets_idrid as convnets_idrid
import vqa.models.convnets_breast as convnets_breast
import vqa.models.convnets_tools as convnets_tools
from vqa.datasets.vqa_processed import tokenize_mcb
from pprint import pprint
from train import load_checkpoint


parser = argparse.ArgumentParser(
    description='Demo server',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--vqa_model', type=str,
                    default='minhmul_noatt_train_2048')
parser.add_argument('--dir_logs', type=str,
                    default='logs/breast/minhmul_noatt_train_2048',
                    help='dir logs')
parser.add_argument('--path_opt', type=str,
                    # default='logs/vqa2/blocmutan_noatt_fbresnet152torchported_save_all/blocmutan_noatt.yaml',
                    default='logs/breast/minhmul_noatt_train_2048/minhmul_noatt_train_2048.yaml',
                    help='path to a yaml options file')
parser.add_argument('--resume', type=str,
                    default='best',
                    help='path to latest checkpoint')
parser.add_argument('--cuda', type=bool,
                    const=True,
                    nargs='?',
                    default=True,
                    help='path to latest checkpoint')
parser.add_argument('--vqa_trainsplit', type=str,
                    choices=['train', 'trainval'], default="train")
parser.add_argument('--st_type',
                    help='skipthoughts type')
parser.add_argument('--st_dropout', type=float)
parser.add_argument('--st_fixed_emb', default=None, type=utils.str2bool,
                    help='backprop on embedding')
# model options
parser.add_argument('--arch', choices=models_vqa.model_names,
                    help='vqa model architecture: ' +
                    ' | '.join(models_vqa.model_names))


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


def process_visual(path_img, cnn, vqa_model="minhmul_noatt_train_2048"):
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
    # print('visual', visual_data.size(), visual_data.mean())

    visual_data = visual_data.cuda(async=True)
    visual_input = Variable(visual_data)

    visual_features = cnn(visual_input)
    if 'noatt' in vqa_model:
        nb_regions = visual_features.size(2) * visual_features.size(3)
        visual_features = visual_features.sum(
            3).sum(2).div(nb_regions).view(-1, 2048)
    return visual_features


def process_question(question_str, trainset):
    question_tokens = tokenize_mcb(question_str)
    question_data = torch.LongTensor(1, len(question_tokens))
    for i, word in enumerate(question_tokens):
        if word in trainset.word_to_wid:
            question_data[0][i] = trainset.word_to_wid[word]
        else:
            question_data[0][i] = trainset.word_to_wid['UNK']
    if args.cuda:
        question_data = question_data.cuda(async=True)
    question_input = Variable(question_data)
    # print('question', question_str, question_tokens, question_data)

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


def load_dict_torch_031(model, path_ckpt):
    model_dict = torch.load(path_ckpt)
    model_dict_clone = model_dict.copy()  # We can't mutate while iterating
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    model.load_state_dict(model_dict, False)
    return model


def load_vqa_model(vqa_model="minhmul_noatt_train_2048"):
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
        }
    }
    with open(path, 'r') as handle:
        options_yaml = yaml.load(handle)
    options = utils.update_values(options, options_yaml)
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
    if os.path.isfile(path_ckpt_model):
        model = load_dict_torch_031(model, path_ckpt_model)
    return model


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

    filename = '/home/minhvu/github/vqa_idrid/data/image_models/best_resnet152_crossentropyloss_breast.pth.tar'
    model = models.resnet152()
    model = load_dict_torch_031(model, filename)
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


def get_gradcam_from_image_model(path_img, cnn, finalconv_name="layer4"):

    cnn.eval()

    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    cnn._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(cnn.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_name = paths_utils.get_filename_without_extension(path_img)
    img_pil = Image.open(path_img)
    in_path = "temp/{}_in.jpg".format(img_name)
    img_pil.save(in_path)

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    img_variable = img_variable.cuda(async=True)
    logit = cnn(img_variable)

    # download the imagenet category list
    classes = {
        0: "Benign",
        1: "InSitu",
        2: "Invasive",
        3: "Normal"
    }

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()

    # output the prediction
    # for i in range(0, 4):
    #     print("{:.3f} -> {}".format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = get_gadcam_image(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
    img = cv2.imread(in_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(
        CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    out_path = "temp/{}_cnn.jpg".format(img_name)
    cv2.imwrite(out_path, result)

    return result, out_path, features_blobs


def get_gadcam_vqa(feature_conv, weight_softmax, weight_softmax_b, class_idx):
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


def get_gradcam_from_vqa_model(visual_features,
                               question_features,
                               features_blobs_visual,
                               ans,
                               path_img,
                               cnn,
                               model,
                               question_str,
                               vqa_model="minhmul_noatt_train_2048",
                               finalconv_name="linear_classif",
                               is_show_image=False):

    model.eval()

    # hook the feature extractor
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    if "noatt" in vqa_model:
        classif_w_params = np.squeeze(params[10].data.cpu().numpy())
        classif_b_params = np.squeeze(params[11].data.cpu().numpy())
    else:
        classif_w_params = np.squeeze(params[26].data.cpu().numpy())
        temp_classif_w_params = np.zeros((classif_w_params.shape[0], 2048))
        temp_classif_w_params = (classif_w_params[:, 0:2048] + classif_w_params[:, 2048:2048*2] +
                                 classif_w_params[:, 2048*2:2048*3] + classif_w_params[:, 2048*3:2048*4]/4)
        classif_w_params = temp_classif_w_params
        classif_b_params = np.squeeze(params[27].data.cpu().numpy())

    logit = model(visual_features, question_features)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()

    CAMs = get_gadcam_vqa(features_blobs_visual[0],
                          classif_w_params, classif_b_params, [idx[0]])

    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s' % ans["ans"][idx[0]])

    img_name = paths_utils.get_filename_without_extension(path_img)

    img = cv2.imread(path_img)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(
        CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5

    question_str = question_str.replace(' ', '-')
    if "noatt" in vqa_model:
        out_path = "temp/{}_noatt_question_{}.jpg".format(
            img_name, question_str)
    else:
        out_path = "temp/{}_att_question_{}.jpg".format(img_name, question_str)
    cv2.imwrite(out_path, result)

    im_out = Image.open(out_path)

    if is_show_image:
        im_out.show()

    return logit


def initialize(args, dataset="breast"):
    options = {
        'logs': {
            'dir_logs': args.dir_logs
        }
    }
    if args.path_opt is not None:
        with open(args.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        options = utils.update_values(options, options_yaml)

    print("\n>> load trainset...")
    trainset = datasets.factory_VQA(options['vqa']['trainsplit'],
                                    options['vqa'],
                                    options['coco'],
                                    options['vgenome'])

    print("\n>> load cnn model...")
    if dataset == "idrid":
        cnn = convnets_idrid.factory(
            {'arch': "resnet152_idrid"}, cuda=True, data_parallel=False)
    elif dataset == "tools":
        cnn = convnets_tools.factory(
            {'arch': "resnet152_tools"}, cuda=True, data_parallel=False)
    elif dataset == "breast":
        cnn = convnets_breast.factory(
            {'arch': "resnet152_breast"}, cuda=True, data_parallel=False)
    cnn = cnn.cuda()

    print("\n>> load vqa model...")
    model = load_vqa_model(args.vqa_model)
    model = model.cuda()

    return cnn, model, trainset


def process_one_example(args, cnn, model, trainset, path_img, question_str, dataset="breast", is_show_image=False):
    print("\n>> extract visual features...")
    visual_features = process_visual(path_img, cnn, args.vqa_model)

    print("\n>> extract question features...")
    question_features = process_question(question_str, trainset)

    print("\n>> get answers...")
    answer, answer_sm = process_answer(
        model(visual_features, question_features), trainset, model)

    print("\n>> get gradcam of cnn...")
    result, out_path, features_blobs_visual = get_gradcam_from_image_model(
        path_img, cnn.net)

    print(question_str)
    print(answer)
    im_in = Image.open(path_img)
    im_out = Image.open(out_path)

    if is_show_image:
        im_in.show()
        im_out.show()

    return visual_features, question_features, answer, answer_sm, features_blobs_visual


def update_args(args, vqa_model="minhmul_noatt_train_2048", dataset="breast"):
    args.vqa_model = vqa_model
    args.dir_logs = "logs/{}/{}".format(dataset, vqa_model)
    args.path_opt = "logs/{}/{}/{}.yaml".format(dataset, vqa_model, vqa_model)
    return args


def main(dataset="breast"):
    global args
    args = parser.parse_args()

    laptop_path = "C:/Users/minhm/Documents/GitHub/vqa_idrid/"
    desktop_path = "/home/minhvu/github/vqa_idrid/"

    is_laptop = False
    if is_laptop:
        path_dir = laptop_path
        ext = "*.jpg"
    else:
        path_dir = desktop_path
        ext = "*"

    LIST_QUESTION_BREAST = [
        "how many classes are there",
        "is normal larger than benign",
        "is normal in 0_0_32_32 location",
        "how many pixels of normal class in the image"
        "is benign larger than normal",
    ]

    LIST_QUESTION_TOOLS = [
        # "how many tools are there",
        # "is there any grasper in the image",
        # "is grasper in 0_0_32_32 location",
        # "which tool has pointed tip on the left of the image",
        "is there any bipolar in the image",
        "is there any hook in the image",
        "is there any scissors in the image",
        "is there any clipper in the image",
        "is there any irrigator in the image",
        "is there any specimenbag in the image",
    ]

    if dataset == "breast":
        path = path_dir + "temp/test_breast/"
        list_question = LIST_QUESTION_BREAST
    elif dataset == "tools":
        path = path_dir + "temp/test_tools/"
        list_question = LIST_QUESTION_TOOLS
    
    img_dirs = glob.glob(os.path.join(path, ext))
    
    args = update_args(
        args, vqa_model="minhmul_noatt_train_2048", dataset=dataset)

    cnn, model, trainset = initialize(args, dataset=dataset)

    for question_str in list_question:
        for path_img in img_dirs:
            visual_features, question_features, ans, answer_sm, features_blobs_visual = process_one_example(args,
                                                                                                            cnn,
                                                                                                            model,
                                                                                                            trainset,
                                                                                                            path_img,
                                                                                                            question_str,
                                                                                                            dataset=dataset)

            get_gradcam_from_vqa_model(visual_features,
                                        question_features,
                                        features_blobs_visual,
                                        ans,
                                        path_img,
                                        cnn,
                                        model,
                                        question_str,
                                        vqa_model="minhmul_noatt_train_2048",
                                        finalconv_name="linear_classif")



    args = update_args(
        args, vqa_model="minhmul_att_train_2048", dataset=dataset)
        
    cnn, model, trainset = initialize(args, dataset=dataset)

    for question_str in list_question:
        for path_img in img_dirs:
            visual_features, question_features, ans, answer_sm, features_blobs_visual = process_one_example(args,
                                                                                                            cnn,
                                                                                                            model,
                                                                                                            trainset,
                                                                                                            path_img,
                                                                                                            question_str,
                                                                                                            dataset=dataset)

            get_gradcam_from_vqa_model(visual_features,
                                        question_features,
                                        features_blobs_visual,
                                        ans,
                                        path_img,
                                        cnn,
                                        model,
                                        question_str,
                                        vqa_model="minhmul_att_train_2048",
                                        finalconv_name="linear_classif")


if __name__ == '__main__':
    dataset = "breast"
    dataset = "tools"
    main(dataset)
