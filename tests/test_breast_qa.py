import os
import glob
import pandas as pd
import numpy as np
from collections import OrderedDict
from xml.dom import minidom
from scipy.ndimage import imread
import datasets.utils.breast_qa_utils as qa_utils
import datasets.utils.paths_utils as path_utils
import datasets.utils.xml_utils as xml_utils
import datasets.utils.image_utils as image_utils
from PIL import Image


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
PREPROCESSED_IMAGE_WSI_DIR = PROJECT_DIR + \
    "/data/raw/breast-cancer/preprocessed/WSI/"
PREPROCESSED_IMAGE_WSI_PATCH_DIR = PREPROCESSED_IMAGE_WSI_DIR + "patch/"
PREPROCESSED_IMAGE_WSI_GT_DIR = PREPROCESSED_IMAGE_WSI_DIR + "patch_gt/"


list_class = [
    "benign",
    "in situ",
    "invasive"
]
list_class = [x.lower() for x in list_class]
list_classes = list_class, list_class


gt_path = PREPROCESSED_IMAGE_WSI_GT_DIR + "A01_idx-12864-14928_ps-16384-16384.png"
# gt_path = PREPROCESSED_IMAGE_WSI_GT_DIR + "A01_idx-10816-2640_ps-4096-4096.jpg"

gt = Image.open(gt_path)
gt = np.array(gt)

gt = imread(gt_path)
# gt.show()
# gt = np.array(gt)
gt = image_utils.convert_rgb_to_gray(gt)

num = qa_utils.get_ans_how_many_classes(gt)
print(num)
