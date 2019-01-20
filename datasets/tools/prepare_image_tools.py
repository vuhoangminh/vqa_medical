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

path_utils.make_dir(PREPROCESSED_IMAGE_DIR)


def normalize(path_in, path_out=None, is_debug=False, is_save=False, is_resize=True, is_normalize=True):
    im = Image.open(path_in)
    imarray = np.array(im)
    if is_normalize:
        im_norm = normalization_utils.normalize_staining(imarray)
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


def prepare_image_model(overwrite=False):
    folder_in = DATASETS_PHOTOS_DIR
    folder_out = PREPROCESSED_IMAGE_DIR
    path_utils.make_dir(folder_out)
    img_dirs = glob.glob(os.path.join(folder_in, "*.jpg"))
    for index, path_in in enumerate(img_dirs):
        filename = path_utils.get_filename_without_extension(path_in)
        path_out = folder_out + "/{}.jpg".format(filename)
        if not os.path.exists(path_out) or overwrite:
            print(">> processing {}/{}".format(index+1, len(img_dirs)))
            normalize(path_in, path_out=path_out, is_debug=False,
                        is_save=True, is_resize=True, is_normalize=False)
        else:
            print("skip {}/{}".format(index+1, len(img_dirs)))


def main():
    print_utils.print_section("image model")
    prepare_image_model()


if __name__ == "__main__":
    main()
