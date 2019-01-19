import numpy as np
from PIL import Image
from scipy.misc import imsave
import cv2


def fill_groundtruth(image, coordinates, color=255):
    cv2.fillPoly(image, coordinates, color=color)
    return image


def convert_gray_to_rgb(img_gray):
    shape = img_gray.shape
    new_shape = (shape[0], shape[1], 3)
    img_rgb = np.zeros(new_shape, dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_gray[i, j] == 1:
                img_rgb[i,j,0], img_rgb[i,j,1], img_rgb[i,j,2] = 255, 0, 0
            elif img_gray[i, j] == 2:
                img_rgb[i,j,0], img_rgb[i,j,1], img_rgb[i,j,2] = 0, 255, 0
            elif img_gray[i, j] == 3:
                img_rgb[i,j,0], img_rgb[i,j,1], img_rgb[i,j,2] = 0, 0, 255
    return img_rgb


def convert_rgb_to_gray(img_rgb):
    shape = img_rgb.shape
    new_shape = (shape[0], shape[1])
    img_gray = np.zeros(new_shape, dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_rgb[i,j,0] == 255 and img_rgb[i,j,1] == 0 and img_rgb[i,j,2] == 0:
                img_gray[i, j] = 1
            elif img_rgb[i,j,0] == 0 and img_rgb[i,j,1] == 255 and img_rgb[i,j,2] == 0:
                img_gray[i, j] = 2
            elif img_rgb[i,j,0] == 0 and img_rgb[i,j,1] == 0 and img_rgb[i,j,2] == 255:
                img_gray[i, j] = 3
    return img_gray


def generate_groundtruth(filename, dims, coordinates, labels, sample=4, is_color=False):
    #red is 'benign', green is 'in situ' and blue is 'invasive'
    if is_color:
        colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
        image_size = (dims[1], dims[0], 3)
    else:
        colors = [0, 1, 2, 3]
        image_size = (dims[1], dims[0])
    img = np.zeros(image_size, dtype=np.uint8)

    for c, l in zip(coordinates, labels):
        img1 = fill_groundtruth(img, [np.int32(np.stack(c))], color=colors[l])
        if is_color:
            img2 = img1[::sample, ::sample, :]
        else:
            img2 = img1[::sample, ::sample]
    return img2


def generate_groundtruth_from_xml(filename, dims, coordinates, labels, sample=4, is_debug=False, is_save=False):
    print(">> generating groungtruth from xml")
    gt = generate_groundtruth(
        filename, dims, coordinates, labels, sample=1, is_color=False)
    if is_debug:
        img2 = generate_groundtruth(
            filename, dims, coordinates, labels, sample=64, is_color=True)
        img2 = Image.fromarray(img2)
        img2.show()
    if is_save:
        img2 = generate_groundtruth(
            filename, dims, coordinates, labels, sample=64, is_color=True)
        imsave(filename, img2)

    return gt


def compute_patch_indices(image_shape, patch_size, overlap,
                          start=None, is_extract_patch_agressive=True):
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
    if start is None:
        if is_extract_patch_agressive:
            n_patches = np.ceil(image_shape / (patch_size - overlap))
            overflow = (patch_size - overlap) * \
                n_patches - image_shape + overlap
            start = -np.ceil(overflow/2)
        else:
            n_patches = np.round(image_shape / (patch_size - overlap))
            n_patches = np.maximum(n_patches, [1, 1, 1])
            overflow = (patch_size - overlap) * \
                n_patches - image_shape + overlap
            start = -np.ceil(overflow/2)
    elif isinstance(start, int):
        start = np.asarray([start] * len(image_shape))
    if is_extract_patch_agressive:
        stop = image_shape + start
    else:
        stop = image_shape + np.floor(overflow/2)
    step = patch_size - overlap

    return get_set_of_patch_indices(start, stop, step)


def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)


def get_patch_from_3d_data(data, patch_shape, patch_index):
    """
    Returns a patch from a numpy array.
    :param data: numpy array from which to get the patch.
    :param patch_shape: shape/size of the patch.
    :param patch_index: corner index of the patch.
    :return: numpy array take from the data with the patch shape specified.
    """
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[-3:]

    return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                patch_index[2]:patch_index[2]+patch_shape[2]]


def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    :param data:
    :param patch_shape:
    :param patch_index:
    :return: padded data, fixed patch index
    """
    image_shape = data.shape[-ndim:]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > image_shape)
                       * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * \
            (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index
