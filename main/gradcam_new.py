import torch._utils
import yaml
import vqa.lib.utils as utils
import vqa.datasets as datasets
import vqa.models as models_vqa
import datasets.utils.paths_utils as paths_utils
import argparse
import glob
import torch
import cv2
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import vqa.models.convnets_idrid as convnets_idrid
import vqa.models.convnets_breast as convnets_breast
import vqa.models.convnets_tools as convnets_tools
from vqa.datasets.vqa_processed import tokenize_mcb


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


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def _fusion_att(self, x_v, x_q):
        x_att = torch.pow(x_q, 2)
        x_att = torch.mul(x_v, x_att)
        return x_att

    def _fusion_classif(self, x_v, x_q):
        x_mm = torch.pow(x_q, 2)
        x_mm = torch.mul(x_v, x_mm)
        return x_mm

    def __call__(self, x, y):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if name == "seq2vec":
                x_q_vec = module(y)
            if name == "conv_v_att":
                x_v = module(x)
            if name == "linear_q_att":
                x_q = module(x_q_vec)
                x_q = x_q.view(1,
                               1,
                               2048)
                x_q = x_q.expand(1,
                                 7 * 7,
                                 2048)
            if name in self.target_layers:
                x_v.register_hook(self.save_gradient)
                outputs += [x_v]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(
            self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x, y):
        target_activations, output = self.feature_extractor(x, y)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, visual_features, question_features):
        return self.model(visual_features, question_features)

    def __call__(self, visual_features, question_features, index=None):
        if self.cuda:
            features, output = self.extractor(
                visual_features.cuda(), question_features.cuda())
        else:
            features, output = self.extractor(
                visual_features, question_features)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        logit = self.model(visual_features, question_features)

        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        output = h_x
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()

        one_hot = np.zeros((1, probs.size), dtype=np.float32)
        one_hot[0][idx[0]] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output)
        self.model.conv_v_att.zero_grad()
        self.model.linear_classif.zero_grad()
        one_hot.backward(retain_graph=True)

        # one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        # one_hot[0][index] = 1
        # one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        # if self.cuda:
        #     one_hot = torch.sum(one_hot.cuda() * output)
        # else:
        #     one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        # one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


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


def process_question(args, question_str, trainset):
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


def process_answer(answer_var, trainset, model, dataset):
    answer_sm = torch.nn.functional.softmax(Variable(answer_var.data[0].cpu()))

    if dataset == "idrid":
        topk = 3
    else:
        topk = 5

    max_, aid = answer_sm.topk(topk, 0, True, True)

    ans = []
    val = []
    for i in range(topk):
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


def load_vqa_model(args, dataset, vqa_model="minhmul_noatt_train_2048"):
    path = "options/{}/{}.yaml".format(dataset, vqa_model)
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
    path_ckpt_model = "logs/{}/{}/best_model.pth.tar".format(
        dataset, vqa_model)
    if os.path.isfile(path_ckpt_model):
        model = load_dict_torch_031(model, path_ckpt_model)
    return model


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    result = heatmap * 0.5 + img * 0.5
    # cam = cam / np.max(cam)
    return result


def get_gadcam_image(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, size_upsample)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
    return cam


def get_gradcam_from_image_model(path_img, cnn, dataset, finalconv_name="layer4"):

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

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    img_variable = img_variable.cuda(async=True)
    logit = cnn(img_variable)

    paths_utils.make_dir("temp/gradcam/{}/".format(dataset))
    in_path = "temp/gradcam/{}/{}_in.jpg".format(dataset, img_name)

    # img_pil.thumbnail((256, 256), Image.ANTIALIAS)
    img_pil = img_pil.resize((256, 256), resample=PIL.Image.NEAREST)
    img_pil.save(in_path)

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

    # generate class activation mapping for the top1 prediction
    cam = get_gadcam_image(features_blobs[0], weight_softmax, [idx[0]])

    img_name = paths_utils.get_filename_without_extension(path_img)

    img = cv2.imread(in_path)

    result = show_cam_on_image(img, cam)

    out_path = "temp/gradcam/{}/{}_cnn.jpg".format(dataset,
                                                   img_name)

    cv2.imwrite(out_path, result)

    return result, out_path, features_blobs


def get_gadcam_vqa(feature_conv, weight_softmax, weight_softmax_b, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, size_upsample)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
    return cam


def get_gradcam_from_vqa_model(visual_features,
                               question_features,
                               features_blobs_visual,
                               ans,
                               path_img,
                               cnn,
                               model,
                               question_str,
                               dataset,
                               vqa_model="minhmul_noatt_train_2048",
                               finalconv_name="linear_classif",
                               is_show_image=False):

    # grad_cam = GradCam(model=model,
    #                    target_layer_names=["conv_v_att"], use_cuda=True)

    # mask = grad_cam(visual_features, question_features)

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
    output = h_x
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()

    one_hot = np.zeros((1, probs.size), dtype=np.float32)
    one_hot[0][idx[0]] = 1
    one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
    one_hot = torch.sum(one_hot.cuda() * output)
    
    # model.conv_v_att.zero_grad()
    # model.linear_classif.zero_grad()

    model.conv_att.register_backward_hook(print)    
    model.conv_v_att.register_backward_hook(print)
    one_hot.backward(retain_graph=True)


    cam = get_gadcam_vqa(features_blobs_visual[0],
                         classif_w_params, classif_b_params, [idx[0]])

    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s' % ans["ans"][idx[0]])

    img_name = paths_utils.get_filename_without_extension(path_img)

    in_path = "temp/gradcam/{}/{}_in.jpg".format(dataset, img_name)

    img = cv2.imread(in_path)

    result = show_cam_on_image(img, cam)

    question_str = question_str.replace(' ', '-')

    paths_utils.make_dir("temp/gradcam/{}/".format(dataset))
    if "noatt" in vqa_model:
        out_path = "temp/gradcam/{}/{}_noatt_question_{}.jpg".format(dataset,
                                                                     img_name,
                                                                     question_str)
    else:
        out_path = "temp/gradcam/{}/{}_att_question_{}.jpg".format(dataset,
                                                                   img_name,
                                                                   question_str)

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
    model = load_vqa_model(args, dataset, args.vqa_model)
    model = model.cuda()

    return cnn, model, trainset


def process_one_example(args, cnn, model, trainset, path_img, question_str, dataset="breast", is_show_image=False):
    print("\n>> extract visual features...")
    visual_features = process_visual(path_img, cnn, args.vqa_model)

    print("\n>> extract question features...")
    question_features = process_question(args, question_str, trainset)

    print("\n>> get answers...")
    answer, answer_sm = process_answer(
        model(visual_features, question_features), trainset, model, dataset)

    print("\n>> get gradcam of cnn...")
    result, out_path, features_blobs_visual = get_gradcam_from_image_model(
        path_img, cnn.net, dataset)

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
    # global args
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
        "is there any benign in the image",
        "is there any in situ carcinoma in the image",
        "is there any invasive carcinoma in the image",
        "what is the major class",
        "what is the minor class",
        "is there benign in the region 64_64_16_16?",
        "is there invasive carcinoma in the region 80_80_16_16?",
    ]

    LIST_QUESTION_TOOLS = [
        "how many tools are there",
        "is there any grasper in the image",
        "is there any bipolar in the image",
        "is there any hook in the image",
        "is there any scissors in the image",
        "is there any clipper in the image",
        "is there any irrigator in the image",
        "is there any specimenbag in the image",
    ]

    LIST_QUESTION_IDRID = [
        "is there haemorrhages in the fundus",
        "is there microaneurysms in the fundus",
        "is there soft exudates in the fundus",
        "is there hard exudates in the fundus",
        "is hard exudates larger than soft exudates",
        "is haemorrhages smaller than microaneurysms",
        "is there haemorrhages in the region 64_64_16_16?",
        "is there microaneurysms in the region 80_80_16_16?",
    ]

    if dataset == "breast":
        path = path_dir + "temp/test_breast/"
        list_question = LIST_QUESTION_BREAST
    elif dataset == "tools":
        path = path_dir + "temp/test_tools/"
        list_question = LIST_QUESTION_TOOLS
    elif dataset == "idrid":
        path = path_dir + "temp/test_idrid/"
        list_question = LIST_QUESTION_IDRID

    img_dirs = glob.glob(os.path.join(path, ext))

    # args = update_args(
    #     args, vqa_model="minhmul_noatt_train_2048", dataset=dataset)

    # cnn, model, trainset = initialize(args, dataset=dataset)

    # for question_str in list_question:
    #     for path_img in img_dirs:
    #         visual_features, question_features, ans, answer_sm, features_blobs_visual = process_one_example(args,
    #                                                                                                         cnn,
    #                                                                                                         model,
    #                                                                                                         trainset,
    #                                                                                                         path_img,
    #                                                                                                         question_str,
    #                                                                                                         dataset=dataset)

    #         get_gradcam_from_vqa_model(visual_features,
    #                                    question_features,
    #                                    features_blobs_visual,
    #                                    ans,
    #                                    path_img,
    #                                    cnn,
    #                                    model,
    #                                    question_str,
    #                                    dataset,
    #                                    vqa_model="minhmul_noatt_train_2048",
    #                                    finalconv_name="linear_classif")

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
                                       dataset,
                                       vqa_model="minhmul_att_train_2048",
                                       finalconv_name="linear_classif")


if __name__ == '__main__':
    # dataset = "breast"
    # main(dataset)
    dataset = "tools"
    main(dataset)
    # dataset = "idrid"
    # main(dataset)
