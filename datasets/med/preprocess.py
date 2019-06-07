import cv2
import os
if os.path.isdir("C:\\Users\\minhm\\Documents\\GitHub\\vqa_idrid"):
    sys.path.append("C:\\Users\\minhm\\Documents\\GitHub\\vqa_idrid")
import numpy as np
import datasets.utils.paths_utils as path_utils
import glob
from random import shuffle
import scipy.ndimage


def show_image(img, caption):
    imS = cv2.resize(img, (256, 256))
    cv2.imshow(caption, imS)


def stack_2_images_horizontally(img1, img2, size=(512, 512), caption="image"):
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    numpy_horizontal = np.hstack((img1, img2))
    return numpy_horizontal


def stack_4_images_horizontally(img1, img2, img3, img4, size=(256, 256), caption="image"):
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)
    img3 = cv2.resize(img3, size)
    img4 = cv2.resize(img4, size)
    numpy_horizontal = np.hstack((img1, img2, img3, img4))
    return numpy_horizontal
    # cv2.imshow('Numpy Horizontal', numpy_horizontal)


def normalize_0_255(img):
    img_norm = (img-np.min(img))/(np.max(img)-np.min(img))
    im = np.array(img_norm * 255, dtype=np.uint8)
    return im


def otsu_binarize(img, threshold=10):
    img = normalize_0_255(img)
    # ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, th = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    # ret, th = cv2.threshold(img, 10, 255, 10)
    return th


def convert_to_3_channels(img):
    if len(img.shape) < 3:
        img_rgb = np.zeros((img.shape[0], img.shape[1], 3))
        img_rgb[:, :, 0] = img
        img_rgb[:, :, 1] = img
        img_rgb[:, :, 2] = img
    else:
        img_rgb = img
    return img_rgb


def remove_connected_objects(img, factor=2, is_remove_small=True):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    # min_size = img.shape[0]*img.shape[0]/factor
    min_size = np.max(sizes)/factor

    # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if is_remove_small and sizes[i] >= min_size:
            img2[output == i + 1] = 255
        if not is_remove_small and sizes[i] < min_size:
            img2[output == i + 1] = 255

    return np.array(img2, dtype=np.uint8)


def fill_holes(img):
    img = img/255
    img = scipy.ndimage.morphology.binary_fill_holes(img)
    img = np.array(img * 255, dtype=np.uint8)
    return img


def remove_text(img, factor):
    kernel = np.ones(
        (int(img.shape[0]/factor),
         int(img.shape[1]/factor)),
        np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening


def get_bounding_box(img, label=255):
    label_im, nb_labels = scipy.ndimage.label(img)

    # Find the largest connected component
    sizes = scipy.ndimage.sum(img, label_im, range(nb_labels + 1))
    mask_size = sizes < 10
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)

    # Now that we have only one connected component, extract it's bounding box
    slice_x, slice_y = scipy.ndimage.find_objects(label_im != 0)[0]
    # roi = im[slice_x, slice_y]
    # slice_x, slice_y = scipy.ndimage.find_objects(label_im == label_im)[0]
    return slice_x, slice_y


def process(index, img_dirs, is_draw=False):
    img_path = img_dirs[index]
    img = cv2.imread(img_path)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(img.shape) > 3:
        assert (len(img.shape) > 3)

    # equ = cv2.equalizeHist(img)
    img = normalize_0_255(img)
    th = otsu_binarize(img, threshold=5)
    img_removed_text = remove_text(th, factor=40)
    img_filled_holes = fill_holes(img_removed_text)
    img_removed_small = remove_connected_objects(
        img_filled_holes, factor=3)

    slice_x, slice_y = get_bounding_box(img_removed_small, label=255)

    img_1st = img[slice_x, slice_y]
    th_roi = img_removed_small[slice_x, slice_y]
    final_mask = remove_text(th_roi, factor=50)
    final_mask_binary = np.array(final_mask/255, dtype=np.uint8)
    W, H = final_mask_binary.shape[0], final_mask_binary.shape[1]
    # final_mask_binary[int(0.1*W):int(0.9*W), int(0.1*H):int(0.9*H)] = 1

    remove_percent = 0.1
    final_mask_binary[int(remove_percent*W):int((1-remove_percent)*W), :] = 1
    final_mask_binary[:, int(remove_percent*H):int((1-remove_percent)*H)] = 1
    final = np.multiply(img_1st, final_mask_binary)

    if is_draw:
        h1 = stack_4_images_horizontally(
            img, th, img_removed_text, img_filled_holes)
        h2 = stack_4_images_horizontally(
            img_removed_small, img_1st, final_mask_binary*255, final)

        numpy_vertical = np.vstack((h1, h2))

        h3 = stack_2_images_horizontally(
            img, final)

        cv2.imshow(path_utils.get_filename_without_extension(img_path), numpy_vertical)
        cv2.imshow("Done", numpy_vertical)
        # cv2.imshow('Done', h3)
        cv2.waitKey(0)

    return final


def main():
    img_dirs = glob.glob(os.path.join(
        "data/raw/vqa_med/ImageClef-2019-VQA-Med-Training/Train_images", "*.jpg"))
    img_dirs = glob.glob(os.path.join(
        "data/raw/vqa_med/ImageClef-2019-VQA-Med-Validation/Val_images", "*.jpg"))
    shuffle(img_dirs)

    for index in range(3000):
        process(index, img_dirs, is_draw=True)


if __name__ == "__main__":
    main()
