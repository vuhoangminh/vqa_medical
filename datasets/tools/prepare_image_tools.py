import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import datasets.utils.paths_utils as path_utils
import datasets.utils.normalization_utils as normalization_utils

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
DATASETS_PHOTOS_DIR = PROJECT_DIR + "/data/raw/breast-cancer/ICIAR2018_BACH_Challenge/Photos/"
DATASETS_WSI_DIR = PROJECT_DIR + "/data/raw/breast-cancer/ICIAR2018_BACH_Challenge/WSI/"
PREPROCESSED_IMAGE_PHOTOS_DIR = PROJECT_DIR + "/data/raw/breast-cancer/preprocessed/Photos/"
PREPROCESSED_IMAGE_WSI_DIR = PROJECT_DIR + "/data/raw/breast-cancer/preprocessed/WSI/"

path_utils.make_dir(PREPROCESSED_IMAGE_PHOTOS_DIR)
path_utils.make_dir(PREPROCESSED_IMAGE_WSI_DIR)

LIST_BREAST_CLASS = ["Benign", "InSitu", "Invasive", "Normal"]


def normalize(path_in, path_out=None, is_debug=False, is_save=False, is_resize=True):
    im = Image.open(path_in)
    imarray = np.array(im)
    im_norm = normalization_utils.normalize_staining(imarray)
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


def prepare_image_model():
    for breast in LIST_BREAST_CLASS:
        folder_in = DATASETS_PHOTOS_DIR + breast
        folder_out = PREPROCESSED_IMAGE_PHOTOS_DIR + breast
        path_utils.make_dir(folder_out)
        img_dirs = glob.glob(os.path.join(folder_in, "*.tif"))
        for index, path_in in enumerate(img_dirs):
            print(">> processing {}/{}".format(index+1, len(img_dirs)))
            filename = path_utils.get_filename_without_extension(path_in)
            path_out = folder_out + "/{}.jpg".format(filename)
            normalize(path_in, path_out=path_out, is_debug=False, is_save=True, is_resize=True)


def prepare_question_model():
    print("ok")


def main():
    # prepare_image_model()
    prepare_question_model()


if __name__ == "__main__":
    main()
