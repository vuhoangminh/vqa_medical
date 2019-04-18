import argparse
import os
import time
import h5py

import torch
import torch.nn.parallel
from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import vqa.models.convnets_idrid as convnets_idrid
import vqa.models.convnets_breast as convnets_breast
import vqa.models.convnets_tools as convnets_tools
import vqa.models.convnets_med as convnets_med
import vqa.datasets as datasets
import datasets.utils.augment_utils as augment_utils
import vqa.lib.utils as gen_utils
from vqa.lib.dataloader import DataLoader
from vqa.lib.logger import AvgMeter
import datasets.utils.print_utils as print_utils

parser = argparse.ArgumentParser(description='Extract')
parser.add_argument('--dataset', default='med',
                    choices=['coco', 'vgenome', 'idrid',
                             'tools', 'breast', 'med'],
                    help='dataset type: coco (default) | vgenome')
parser.add_argument('--dir_data', default='data/raw/vqa_med/preprocessed',
                    help='dir dataset to download or/and load images')
parser.add_argument('--data_split', default='train', type=str,
                    help='Options: (default) train | val | test')
parser.add_argument('--arch', '-a', default='fbresnet152',
                    help='model architecture: ' +
                    ' | '.join(convnets_idrid.model_names) +
                    ' (default: fbresnet152)')
parser.add_argument('--workers', default=0, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', '-b', default=4, type=int,
                    help='mini-batch size (default: 80)')
parser.add_argument('--mode', default='both', type=str,
                    help='Options: att | noatt | (default) both')
parser.add_argument('--size', default=448, type=int,
                    help='Image size (448 for noatt := avg pooling to get 224) (default:448)')
parser.add_argument('--is_augment_image', default='0',
                    help='whether to augment images at the beginning of every epoch?')


def main():
    global args
    args = parser.parse_args()

    print("=> using pre-trained model '{}'".format(args.arch))
    # model = convnets.factory({'arch':}, cuda=True, data_parallel=True)

    # model = convnets.factory({'arch':args.arch}, cuda=True, data_parallel=True)
    # model = convnets.factory({'arch':'resnet18'}, cuda=False, data_parallel=False)

    # if debug:
    if args.dataset == "idrid":
        model = convnets_idrid.factory(
            {'arch': args.arch}, cuda=True, data_parallel=True)
    elif args.dataset == "tools":
        model = convnets_tools.factory(
            {'arch': args.arch}, cuda=True, data_parallel=True)
    elif args.dataset == "breast":
        model = convnets_breast.factory(
            {'arch': args.arch}, cuda=True, data_parallel=True)
    elif args.dataset == "med":
        model = convnets_med.factory(
            {'arch': args.arch}, cuda=True, data_parallel=True)

    extract_name = 'arch,{}_size,{}'.format(args.arch, args.size)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.dataset == 'coco':
        if 'coco' not in args.dir_data:
            raise ValueError('"coco" string not in dir_data')
        dataset = datasets.COCOImages(args.data_split, dict(dir=args.dir_data),
                                      transform=transforms.Compose([
                                          transforms.Resize(args.size),
                                          transforms.CenterCrop(args.size),
                                          transforms.ToTensor()
                                      ]))
    elif args.dataset == 'vgenome':
        if args.data_split != 'train':
            raise ValueError('train split is required for vgenome')
        if 'vgenome' not in args.dir_data:
            raise ValueError('"vgenome" string not in dir_data')
        dataset = datasets.VisualGenomeImages(args.data_split, dict(dir=args.dir_data),
                                              transform=transforms.Compose([
                                                  transforms.Resize(args.size),
                                                  transforms.CenterCrop(
                                                      args.size),
                                                  transforms.ToTensor(),
                                                  normalize,
                                              ]))
    elif args.dataset == 'idrid':
        dataset = datasets.IDRIDImages(args.data_split, dict(dir=args.dir_data),
                                       transform=transforms.Compose([
                                           transforms.Resize(args.size),
                                           transforms.CenterCrop(args.size),
                                           transforms.ToTensor(),
                                           normalize,
                                       ]))
    elif args.dataset == 'tools':
        dataset = datasets.TOOLSImages(args.data_split, dict(dir=args.dir_data),
                                       transform=transforms.Compose([
                                           transforms.Resize(args.size),
                                           transforms.CenterCrop(args.size),
                                           transforms.ToTensor(),
                                           normalize,
                                       ]))
    elif args.dataset == 'breast':
        dataset = datasets.BREASTImages(args.data_split, dict(dir=args.dir_data),
                                        transform=transforms.Compose([
                                            transforms.Resize(args.size),
                                            transforms.CenterCrop(args.size),
                                            transforms.ToTensor(),
                                            normalize,
                                        ]))
    elif args.dataset == 'med':
        if gen_utils.str2bool(args.is_augment_image):
            transform = transforms.Compose([
                transforms.Resize(args.size),
                # transforms.CenterCrop(args.size),
                # transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(-30, 30)),
                augment_utils.PowerPILMed(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(args.size),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                normalize,
            ])
        dataset = datasets.MEDImages(args.data_split, dict(dir=args.dir_data),
                                     transform=transform)

    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    dir_extract = os.path.join(args.dir_data, 'extract', extract_name)
    path_file = os.path.join(dir_extract, args.data_split + 'set')
    os.system('mkdir -p ' + dir_extract)

    # if args.dataset == "med":
    #     extract_med(data_loader, model, path_file, args.mode)
    # else:
    extract(data_loader, model, path_file, args.mode, args.is_augment_image)


def extract(data_loader, model, path_file, mode, is_augment_image):
    path_hdf5 = path_file + '.hdf5'
    path_txt = path_file + '.txt'
    if os.path.exists(path_hdf5):
        print("remove existing", path_hdf5)
        os.remove(path_hdf5)
    hdf5_file = h5py.File(path_hdf5, 'w')

    # estimate output shapes
    output, hidden = model(Variable(torch.ones(1, 3, args.size, args.size)))

    nb_images = len(data_loader.dataset)
    if mode == 'both' or mode == 'att':
        shape_att = (nb_images, output.size(1), output.size(2), output.size(3))
        print('Warning: shape_att={}'.format(shape_att))
        hdf5_att = hdf5_file.create_dataset('att', shape_att,
                                            dtype='f')  # , compression='gzip')
    if mode == 'both' or mode == 'noatt':
        shape_noatt = (nb_images, output.size(1))
        print('Warning: shape_noatt={}'.format(shape_noatt))
        hdf5_noatt = hdf5_file.create_dataset('noatt', shape_noatt,
                                              dtype='f')  # , compression='gzip')

    model.eval()

    batch_time = AvgMeter()
    data_time = AvgMeter()
    begin = time.time()
    end = time.time()

    idx = 0
    if gen_utils.str2bool(is_augment_image):
        print("\n>> extract augmented images\n")
    else:
        print("\n>> extract original images\n")

    with torch.no_grad():
        for i, input in enumerate(data_loader):
            print_utils.print_tqdm(i, len(data_loader), cutoff=10)
            input_var = Variable(input['visual'])
            output_att, _ = model(input_var)

            nb_regions = output_att.size(2) * output_att.size(3)
            output_noatt = output_att.sum(3).sum(2).div(nb_regions).view(-1, 2048)

            batch_size = output_att.size(0)
            if mode == 'both' or mode == 'att':
                hdf5_att[idx:idx+batch_size] = output_att.data.cpu().numpy()
            if mode == 'both' or mode == 'noatt':
                hdf5_noatt[idx:idx+batch_size] = output_noatt.data.cpu().numpy()
            idx += batch_size

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

        hdf5_file.close()

    # Saving image names in the same order than extraction
    with open(path_txt, 'w') as handle:
        for name in data_loader.dataset.dataset.imgs:
            handle.write(name + '\n')

    end = time.time() - begin
    print('Finished in {}m and {}s'.format(int(end/60), int(end % 60)))


def extract_med(data_loader, model, path_file, mode):
    path_hdf5 = path_file + '.hdf5'
    path_txt = path_file + '.txt'
    hdf5_file = h5py.File(path_hdf5, 'w')

    # estimate output shapes
    _, output = model(
        Variable(torch.ones(1, 3, args.size, args.size)))

    nb_images = len(data_loader.dataset)
    if mode == 'both' or mode == 'att':
        shape_att = (nb_images, output.size(1), output.size(2), output.size(3))
        print('Warning: shape_att={}'.format(shape_att))
        hdf5_att = hdf5_file.create_dataset('att', shape_att,
                                            dtype='f')  # , compression='gzip')

    model.eval()

    batch_time = AvgMeter()
    data_time = AvgMeter()
    begin = time.time()
    end = time.time()

    idx = 0
    for i, input in enumerate(data_loader):
        input_var = Variable(input['visual'], volatile=True)
        _, output_att = model(input_var)

        nb_regions = output_att.size(2) * output_att.size(3)

        batch_size = output_att.size(0)
        if mode == 'both' or mode == 'att':
            hdf5_att[idx:idx+batch_size] = output_att.data.cpu().numpy()
        idx += batch_size

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1 == 0:
            print('Extract: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i, len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,))

    hdf5_file.close()

    # Saving image names in the same order than extraction
    with open(path_txt, 'w') as handle:
        for name in data_loader.dataset.dataset.imgs:
            handle.write(name + '\n')

    end = time.time() - begin
    print('Finished in {}m and {}s'.format(int(end/60), int(end % 60)))


if __name__ == '__main__':
    main()
