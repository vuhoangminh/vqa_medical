import torch
import cv2
from imgaug import augmenters as iaa
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from PIL import Image
import PIL.ImageEnhance as ie
import PIL.Image as im


RGB_MEAN = [0.5, 0.5, 0.5]
RGB_STD = [0.5, 0.5, 0.5]


def separate_resnet_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    paras_wo_bn_to_finetune = []
    for index, layer in enumerate(modules):
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
        if index > 390:
            if 'batchnorm' in str(layer.__class__):
                continue
            else:
                paras_wo_bn_to_finetune.extend([*layer.parameters()])


    return paras_only_bn, paras_wo_bn, paras_wo_bn_to_finetune


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight, weight_per_class



def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def augment(images):
    def sometimes(aug): return iaa.Sometimes(0.5, aug)

    def rare(aug): return iaa.Sometimes(0.25, aug)

    seq = iaa.Sequential(
        [
            sometimes(iaa.CropAndPad(
                percent=(-0.1, 0.1),
            )),
            sometimes(iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                # translate by -20 to +20 percent (per axis)
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-20, 20),  # rotate by -45 to +45 degrees
                shear=(-15, 15),  # shear by -16 to +16 degrees
                # use nearest neighbour or bilinear interpolation (fast)
                order=[0, 1],

            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((1, 2),
                       [
                iaa.OneOf([
                    # blur images with a sigma between 0 and 3.0
                    iaa.GaussianBlur((0, 5.0)),
                    # blur image using local means with kernel sizes between 2 and 7
                    iaa.AverageBlur(k=(5, 7)),
                    # blur image using local medians with kernel sizes between 2 and 7
                    # iaa.MedianBlur(k=(3, 11)),
                ]),
                # add gaussian noise to images
                iaa.AdditiveGaussianNoise(loc=0, scale=(
                    0.0, 0.03*255), per_channel=True),
                # iaa.Invert(0.001, per_channel=True),  # invert color channels
                iaa.Add((-5, 5), per_channel=0.5),
                iaa.OneOf([
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
                    iaa.PerspectiveTransform(scale=(0.01, 0.1))
                ]),
            ],
                random_order=True
            )
        ],
        random_order=True
    )

    images_aug = seq.augment_images(images)

    num_images_convert_to_gray = int(0.2*images_aug.shape[0])

    random_index__convert_to_gray = random.sample(
        range(0, images_aug.shape[0]), num_images_convert_to_gray)

    for i in range(num_images_convert_to_gray):
        index = random_index__convert_to_gray[i]
        rgb = images_aug[index, :, :, :]
        gray = rgb2gray(rgb)
        rgb = np.zeros((rgb.shape[0], rgb.shape[1], rgb.shape[2]))
        rgb[:, :, 0] = gray
        rgb[:, :, 1] = gray
        rgb[:, :, 2] = gray
        images_aug[index, :, :, :] = rgb

    return images_aug


def random_cropping(image, target_shape=(128, 128), p=0.5):
    zeros = np.zeros(target_shape)
    target_w, target_h = target_shape
    width, height = image.shape
    if random.random() < p:
        start_x = random.randint(0, target_w - width)
        start_y = random.randint(0, target_h - height)
        zeros[start_x:start_x+width, start_y:start_y+height] = image
    else:
        start_x = (target_w - width)//2
        start_y = (target_h - height)//2
        zeros[start_x:start_x+width, start_y:start_y+height] = image
    return zeros


def TTA_cropps(image, target_shape=(128, 128, 3)):
    width, height, d = image.shape
    target_w, target_h, d = target_shape
    start_x = (target_w - width) // 2
    start_y = (target_h - height) // 2
    starts = [[start_x, start_y], [0, 0], [2 * start_x, 0],
              [0, 2 * start_y], [2 * start_x, 2 * start_y]]
    images = []
    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        zeros = np.zeros(target_shape)
        zeros[x:x + width, y: y+height, :] = image_
        image_ = zeros.copy()
        image_ = (torch.from_numpy(image_).div(255)).float()
        image_ = image_.permute(2, 0, 1)
        images.append(image_)

        zeros = np.fliplr(zeros)
        image_ = zeros.copy()
        image_ = (torch.from_numpy(image_).div(255)).float()
        image_ = image_.permute(2, 0, 1)
        images.append(image_)

    return images


def random_erase(image, p=0.5):
    if random.random() < p:
        width, height, d = image.shape
        x = random.randint(0, width)
        y = random.randint(0, height)
        b_w = random.randint(5, 10)
        b_h = random.randint(5, 10)
        image[x:x+b_w, y:y+b_h] = 0
    return image


def random_cropping3d(image, target_shape=(8, 128, 128), p=0.5):
    zeros = np.zeros(target_shape)
    target_l, target_w, target_h = target_shape
    length, width, height = image.shape
    if random.random() < p:
        start_x = random.randint(0, target_w - width)
        start_y = random.randint(0, target_h - height)
    else:
        start_x = (target_w - width) // 2
        start_y = (target_h - height) // 2
    zeros[:target_l, start_x:start_x+width, start_y:start_y+height] = image
    return zeros


def random_shift(image, p=1):
    if random.random() < p:
        width, height, d = image.shape
        zero_image = np.zeros_like(image)
        w = random.randint(0, 20) - 10
        h = random.randint(0, 30) - 15
        zero_image[max(0, w): min(w+width, width), max(h, 0): min(h+height, height)] = \
            image[max(0, -w): min(-w+width, width),
                  max(-h, 0): min(-h+height, height)]
        image = zero_image.copy()
    return image


def random_scale(image, p=0.5):
    if random.random() < p:
        scale = random.random() * 0.1 + 0.9
        assert 0.9 <= scale <= 1
        width, height, d = image.shape
        zero_image = np.zeros_like(image)
        new_width = round(width * scale)
        new_height = round(height * scale)
        image = cv2.resize(image, (new_height, new_width))
        start_w = random.randint(0, width - new_width)
        start_h = random.randint(0, height - new_height)
        zero_image[start_w: start_w + new_width,
                   start_h:start_h+new_height] = image
        image = zero_image.copy()
    return image


def change_scale(image, scale=1):
    if 1:
        assert 0.9 <= scale <= 1
        width, height, d = image.shape
        zero_image = np.zeros_like(image)
        new_width = round(width * scale)
        new_height = round(height * scale)
        image = cv2.resize(image, (new_height, new_width))
        start_w = (width - new_width)//2
        start_h = (height - new_height)//2
        zero_image[start_w: start_w + new_width,
                   start_h:start_h+new_height] = image
        image = zero_image.copy()
    return image


def random_flip(image, p=0.5):
    if random.random() < p:
        if len(image.shape) == 2:
            image = np.flip(image, 1)
        elif len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
            image = np.flip(image, 1)
            image = np.transpose(image, (2, 0, 1))
    return image


def do_gaussian_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H, W = gray.shape

    noise = np.random.normal(0, sigma, (H, W))
    noisy = gray + noise

    noisy = (np.clip(noisy, 0, 1)*255).astype(np.uint8)
    lab = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


def do_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H, W = gray.shape

    noise = sigma*np.random.randn(H, W)
    noisy = gray + gray * noise

    noisy = (np.clip(noisy, 0, 1)*255).astype(np.uint8)
    lab = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


def do_inv_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H, W = gray.shape

    noise = sigma*np.random.randn(H, W)
    noisy = gray + (1-gray) * noise

    noisy = (np.clip(noisy, 0, 1)*255).astype(np.uint8)
    lab = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


def random_angle_rotate(image, angles=[-30, 30]):
    angle = random.randint(0, angles[1]-angles[0]) + angles[0]
    image = rotate(image, angle)
    return image


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

# illumination ====================================================================================


def do_brightness_shift(image, alpha=0.125):
    image = image.astype(np.float32)
    image = image + alpha*255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_brightness_multiply(image, alpha=1):
    image = image.astype(np.float32)
    image = alpha*image
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_contrast(image, alpha=1.0):
    image = image.astype(np.float32)
    gray = image * np.array([[[0.114, 0.587,  0.299]]])  # rgb to gray (YCbCr)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    image = alpha*image + gray
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/


def do_gamma(image, gamma=1.0):

    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def do_clahe(image, clip=2, grid=16):
    grid = int(grid)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = cv2.createCLAHE(
        clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)
    lab = cv2.merge((gray, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


def do_flip_transpose(image, type=0):
    # choose one of the 8 cases

    if type == 1:  # rotate90
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 1)

    if type == 2:  # rotate180
        image = cv2.flip(image, -1)

    if type == 3:  # rotate270
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 0)

    if type == 4:  # flip left-right
        image = cv2.flip(image, 1)

    if type == 5:  # flip up-down
        image = cv2.flip(image, 0)

    if type == 6:
        image = cv2.flip(image, 1)
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 1)

    if type == 7:
        image = cv2.flip(image, 0)
        image = image.transpose(1, 0, 2)
        image = cv2.flip(image, 1)

    return image


def bgr_to_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def do_flip_transpose_4(image, type=0):
    # choose one of the 8 cases

    if type == 0:  # rotate180
        image = cv2.flip(image, -1)

    if type == 1:  # flip left-right
        image = cv2.flip(image, 1)

    if type == 2:  # flip up-down
        image = cv2.flip(image, 0)

    return image


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data


class RandomErasing(object):
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    #img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                    #img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[1]
                    # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
                return img

        return img


def random_crop(img, boxes):
    '''Crop the given PIL image to a random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.
    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
    Returns:
      img: (PIL.Image) randomly cropped image.
      boxes: (tensor) randomly cropped boxes.
    '''
    success = False
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2

    img = img.crop((x, y, x+w, y+h))
    boxes -= torch.Tensor([x, y, x, y])
    boxes[:, 0::2].clamp_(min=0, max=w-1)
    boxes[:, 1::2].clamp_(min=0, max=h-1)
    return img, boxes


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))


class PILColorBalance(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Color(img).enhance(alpha)


class PILContrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Contrast(img).enhance(alpha)


class PILBrightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Brightness(img).enhance(alpha)


class PILSharpness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        alpha = random.uniform(1 - self.var, 1 + self.var)
        return ie.Sharpness(img).enhance(alpha)


class Gaussian(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img, sigma=0.5):
        img = np.array(img)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray, a, b = cv2.split(lab)
        gray = gray.astype(np.float32)/255
        H, W = gray.shape

        noise = np.random.normal(0, sigma, (H, W))
        noisy = gray + noise

        noisy = (np.clip(noisy, 0, 1)*255).astype(np.uint8)
        lab = cv2.merge((noisy, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return Image.fromarray(image)


class SpeckleNoise(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img, sigma=0.5):
        img = np.array(img)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray, a, b = cv2.split(lab)
        gray = gray.astype(np.float32)/255
        H, W = gray.shape

        noise = sigma*np.random.randn(H, W)
        noisy = gray + gray * noise

        noisy = (np.clip(noisy, 0, 1)*255).astype(np.uint8)
        lab = cv2.merge((noisy, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return Image.fromarray(image)


class RandomErase(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img, sigma=0.5):
        img = np.array(img)
        width, height, d = img.shape
        x = random.randint(0, width)
        y = random.randint(0, height)
        b_w = random.randint(5, 10)
        b_h = random.randint(5, 10)
        img[x:x+b_w, y:y+b_h] = 0
        return Image.fromarray(img)


class RandomShift(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img, sigma=0.5):
        img = np.array(img)
        width, height, d = img.shape
        zero_image = np.zeros_like(img)
        w = random.randint(0, 20) - 10
        h = random.randint(0, 30) - 15
        zero_image[max(0, w): min(w+width, width), max(h, 0): min(h+height, height)] = \
            img[max(0, -w): min(-w+width, width),
                max(-h, 0): min(-h+height, height)]
        image = zero_image.copy()
        return Image.fromarray(img)


class RandomScale(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img, sigma=0.5):
        img = np.array(img)
        scale = random.random() * 0.1 + 0.9
        assert 0.9 <= scale <= 1
        width, height, d = img.shape
        zero_image = np.zeros_like(img)
        new_width = round(width * scale)
        new_height = round(height * scale)
        img = cv2.resize(img, (new_height, new_width))
        start_w = random.randint(0, width - new_width)
        start_h = random.randint(0, height - new_height)
        zero_image[start_w: start_w + new_width,
                   start_h:start_h+new_height] = img
        img = zero_image.copy()
        return Image.fromarray(img)


# Check ImageEnhancer effect: https://www.youtube.com/watch?v=_7iDTpTop04
# Not documented but all enhancements can go beyond 1.0 to 2
# Image must be RGB
# Use Pillow-SIMD because Pillow is too slow
class PowerPIL(RandomOrder):
    def __init__(self,
                 #  rotate=True,
                 #  flip=True,
                 colorbalance=0.4,
                 contrast=0.4,
                 brightness=0.4,
                 sharpness=0.4,
                 noise=0.3,
                 erase=0.1,
                 shift=0.1,
                 scale=0.1):
        self.transforms = []
        # if rotate:
        #     self.transforms.append(RandomRotate())
        # if flip:
        #     self.transforms.append(RandomFlip())
        if brightness != 0:
            self.transforms.append(PILBrightness(brightness))
        if contrast != 0:
            self.transforms.append(PILContrast(contrast))
        if colorbalance != 0:
            self.transforms.append(PILColorBalance(colorbalance))
        if sharpness != 0:
            self.transforms.append(PILSharpness(sharpness))
        if random.random() < noise:
            index = random.randint(0, 1)
            if index == 0:
                self.transforms.append(Gaussian(1))
            elif index == 1:
                self.transforms.append(SpeckleNoise(1))
        if random.random() < erase:
            self.transforms.append(RandomErase(1))
        if random.random() < shift:
            self.transforms.append(RandomShift(1))
        if random.random() < scale:
            self.transforms.append(RandomScale(1))


class PowerPILMed(RandomOrder):
    def __init__(self,
                 contrast=0.3,
                 brightness=0.3,
                 sharpness=0.4,
                 erase=0.1,
                 shift=0.2,
                 scale=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(PILBrightness(brightness))
        if contrast != 0:
            self.transforms.append(PILContrast(contrast))
        if sharpness != 0:
            self.transforms.append(PILSharpness(sharpness))
        if random.random() < erase:
            self.transforms.append(RandomErase(1))
        if random.random() < shift:
            self.transforms.append(RandomShift(1))
        if random.random() < scale:
            self.transforms.append(RandomScale(1))


def default_loader(input_path):
    input_image = (Image.open(input_path)).convert('RGB')
    return input_image
