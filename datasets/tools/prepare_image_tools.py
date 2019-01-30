import datasets.utils.svs_utils as svs_utils
import datasets.utils.normalization_utils as normalization_utils
import datasets.utils.print_utils as print_utils
import datasets.utils.paths_utils as path_utils
import datasets.utils.xml_utils as xml_utils
import datasets.utils.image_utils as image_utils
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import glob
from scipy.misc import imsave, imresize
from scipy.misc import toimage
from scipy.misc.pilutil import imshow
from xml.dom import minidom
import shutil
import random
random.seed(1988)
import os
OPENSLIDE_PATH = "C:/Users/minhm/Documents/GitHub/bin/openslide-win64-20171122/bin"
if os.path.exists(OPENSLIDE_PATH):
    os.environ['PATH'] = OPENSLIDE_PATH + ";" + os.environ['PATH']
import openslide


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
DATASETS_PHOTOS_DIR = PROJECT_DIR + \
    "/data/raw/m2cai16-tool-locations/JPEGImages/"
PREPROCESSED_IMAGE_DIR = PROJECT_DIR + \
    "/data/raw/m2cai16-tool-locations/preprocessed/full/"
CLASSIFICATION_IMAGE_DIR = PROJECT_DIR + \
    "/data/raw/m2cai16-tool-locations/preprocessed/classification/"
SEGMENTATION_IMAGE_DIR = PROJECT_DIR + \
    "/data/raw/m2cai16-tool-locations/preprocessed/segmentation/"
TRAIN_SEGMENTATION_IMAGE_DIR = PROJECT_DIR + \
    "/data/raw/m2cai16-tool-locations/preprocessed/segmentation/train/"
VAL_SEGMENTATION_IMAGE_DIR = PROJECT_DIR + \
    "/data/raw/m2cai16-tool-locations/preprocessed/segmentation/val/"
IMAGE_SET_DIR = PROJECT_DIR + \
    "/data/raw/m2cai16-tool-locations/preprocessed/imagesets/"
DATASETS_DIR = PROJECT_DIR + "/data/raw/m2cai16-tool-locations/Annotations/"
RAW_DIR = PROJECT_DIR + "/data/vqa_tools/raw/raw/"

path_utils.make_dir(PREPROCESSED_IMAGE_DIR)
path_utils.make_dir(CLASSIFICATION_IMAGE_DIR)
path_utils.make_dir(SEGMENTATION_IMAGE_DIR)
path_utils.make_dir(IMAGE_SET_DIR)


list_tool = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag"
]
list_tool = [x.lower() for x in list_tool]
list_tools = list_tool, list_tool


def normalize(path_in, path_out=None, is_debug=False, is_save=False, is_resize=True, is_normalize=True):
    im = Image.open(path_in)
    imarray = np.array(im)
    if is_normalize:
        im_norm = normalization_utils.normalize_rgb(imarray)
    else:
        im_norm = imarray

    im_norm = Image.fromarray(im_norm)

    if is_resize:
        im_resized = im_norm.resize((256, 256), Image.ANTIALIAS)
    else:
        im_resized = im_norm

    if is_debug:
        im.show()
        im_norm.show()
        im_resized.show()

    if is_save:
        im_resized.save(path_out)


def prepare_image_model(overwrite=False, is_debug=False):
    folder_in = DATASETS_PHOTOS_DIR
    folder_out = PREPROCESSED_IMAGE_DIR
    path_utils.make_dir(folder_out)
    img_dirs = glob.glob(os.path.join(folder_in, "*.jpg"))
    for index, path_in in enumerate(img_dirs):
        filename = path_utils.get_filename_without_extension(path_in)
        path_out = folder_out + "/{}.jpg".format(filename)
        if not os.path.exists(path_out) or overwrite:
            print(">> processing {}/{}".format(index+1, len(img_dirs)))
            normalize(path_in, path_out=path_out, is_debug=is_debug,
                        is_save=True, is_resize=True, is_normalize=True)
        else:
            print("skip {}/{}".format(index+1, len(img_dirs)))


def split_images_for_image_model_and_vqa(N=100, P=0.6):
    paths = glob.glob(DATASETS_DIR + "*.xml")
    random.shuffle(paths)
    
    num_images = len(paths)
    num_images_segmentation = num_images - N*len(list_tool)
    num_images_segmentation_train = int(round(num_images_segmentation*P))
    num_images_segmentation_val = num_images_segmentation - num_images_segmentation_train

    rows = list()
    rows_temp = list()
    count_dict_tool_multi = dict(zip(list_tool, [0] * 7))
    count_train_images_segmentation = 0
    for index, path in enumerate(paths):
        filename = path_utils.get_filename_without_extension(path)
        print(">> processing {}/{}".format(index+1, len(paths)))
        mydoc = minidom.parse(path)
        boxes = xml_utils.get_object_xml(mydoc)
        for i in range(len(boxes)):
            tool, xmin, xmax, ymin, ymax = boxes[i]
            tool = tool.lower()
        if len(boxes)==1 and count_dict_tool_multi[tool] < N:
            src = PREPROCESSED_IMAGE_DIR + filename + ".jpg"
            dst_dir = CLASSIFICATION_IMAGE_DIR + tool
            path_utils.make_dir(dst_dir)
            dst = "{}/{}.jpg".format(dst_dir, filename)
            shutil.copyfile(src, dst)
            count_dict_tool_multi[tool] += 1
            print(count_dict_tool_multi)
        else:
            src = PREPROCESSED_IMAGE_DIR + filename + ".jpg"
            if count_train_images_segmentation < num_images_segmentation_train:
                dst_dir = "{}{}".format(SEGMENTATION_IMAGE_DIR, "train")
                path_utils.make_dir(dst_dir)
                count_train_images_segmentation += 1
            else:
                dst_dir = "{}{}".format(SEGMENTATION_IMAGE_DIR, "val")
                path_utils.make_dir(dst_dir)
            dst = "{}/{}.jpg".format(dst_dir, filename)
            shutil.copyfile(src, dst)

def main():
    print_utils.print_section("image model")
    # prepare_image_model(overwrite=False, is_debug=False)
    print_utils.print_section("split images")
    split_images_for_image_model_and_vqa()


if __name__ == "__main__":
    main()
