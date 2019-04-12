from collections import OrderedDict
import pretrainedmodels as torch7_models
import copy
import torch
import torch.nn as nn
import torchvision.models as pytorch_models
import sys
sys.path.append('vqa/external/pretrained-models.pytorch')

pytorch_resnet_names = sorted(name for name in pytorch_models.__dict__
                              if name.islower()
                              and name.startswith("resnet")
                              and callable(pytorch_models.__dict__[name]))

torch7_resnet_names = sorted(name for name in torch7_models.__dict__
                             if name.islower()
                             and callable(torch7_models.__dict__[name]))

model_names = pytorch_resnet_names + torch7_resnet_names + \
    ['resnet18_med', 'resnet152_med']


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


def factory(opt, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    class WrapperModule(nn.Module):
        def __init__(self, net, forward_fn):
            super(WrapperModule, self).__init__()
            self.net = net
            self.forward_fn = forward_fn

        def forward(self, x):
            return self.forward_fn(self.net, x)

        def __getattr__(self, attr):
            try:
                return super(WrapperModule, self).__getattr__(attr)
            except AttributeError:
                return getattr(self.net, attr)

    def forward_resnet(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x = self.layer4(x3)

        if 'pooling' in opt and opt['pooling']:
            x = self.avgpool(x)
            div = x.size(3) + x.size(2)
            x = x.sum(3)
            x = x.sum(2)
            x = x.view(x.size(0), -1)
            x = x.div(div)

        return x, x3

    def forward_resnext(self, x):
        x = self.features(x)

        if 'pooling' in opt and opt['pooling']:
            x = self.avgpool(x)
            div = x.size(3) + x.size(2)
            x = x.sum(3)
            x = x.sum(2)
            x = x.view(x.size(0), -1)
            x = x.div(div)

        return x

    if opt['arch'] in pytorch_resnet_names:
        model = pytorch_models.__dict__[opt['arch']](pretrained=True)

        #  ugly hack in case of DataParallel wrapping
        model = WrapperModule(model, forward_resnet)

    elif opt['arch'] == 'fbresnet152':
        model = torch7_models.__dict__[opt['arch']](num_classes=1000,
                                                    pretrained='imagenet')

        #  ugly hack in case of DataParallel wrapping
        model = WrapperModule(model, forward_resnet)

    elif opt['arch'] in torch7_resnet_names:
        model = torch7_models.__dict__[opt['arch']](num_classes=1000,
                                                    pretrained='imagenet')

        #  ugly hack in case of DataParallel wrapping
        model = WrapperModule(model, forward_resnext)

    elif opt['arch'] == 'resnet18_med':
        filename = 'data/image_models/best_resnet18_crossentropyloss_med.pth.tar'
        model = pytorch_models.resnet18()
        checkpoint = torch.load(filename)
        state_dict = checkpoint['state_dict']
        state_dict = rename_key(state_dict)
        model.load_state_dict(state_dict)
        #  ugly hack in case of DataParallel wrapping
        model = WrapperModule(model, forward_resnet)

    elif opt['arch'] == 'resnet152_med':
        filename = 'data/image_models/best_resnet152_crossentropyloss_med.pth.tar'
        model = pytorch_models.resnet152()
        checkpoint = torch.load(filename)
        state_dict = checkpoint['state_dict']
        state_dict = rename_key(state_dict)
        model.load_state_dict(state_dict)
        #  ugly hack in case of DataParallel wrapping
        model = WrapperModule(model, forward_resnet)

    else:
        raise ValueError

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model
