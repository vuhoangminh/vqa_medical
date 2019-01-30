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

DICT_CLASS = {
    0: "normal",
    1: "benign",
    2: "in situ",
    3: "invasive"
}


list_class = [
    "normal",
    "benign",
    "in situ",
    "invasive"
]

list_class = [x.lower() for x in list_class]
list_classes = list_class, list_class


gt_path = PREPROCESSED_IMAGE_WSI_GT_DIR + "A01_idx-12864-14928_ps-16384-16384.png"
gt_path = PREPROCESSED_IMAGE_WSI_GT_DIR + "A01_idx-10816-2640_ps-4096-4096.png"
# gt_path = PREPROCESSED_IMAGE_WSI_GT_DIR + "A09_idx-5072-32095_ps-8192-8192.png"
# gt_path = PREPROCESSED_IMAGE_WSI_GT_DIR + "A09_idx-5072-23903_ps-8192-8192.png"


gt = imread(gt_path)
gt = image_utils.convert_rgb_to_gray(gt)

num = qa_utils.get_ans_how_many_classes(gt)
print(num)



for i in range(4):
    print(qa_utils.get_name_class_from_number(i, DICT_CLASS))


print(qa_utils.get_ans_major_class(gt, DICT_CLASS))
print(qa_utils.get_ans_minor_class(gt, DICT_CLASS))

print(qa_utils.get_ans_major_tumor(gt, DICT_CLASS))
print(qa_utils.get_ans_minor_tumor(gt, DICT_CLASS))


q, combinations = qa_utils.generate_ques_is_x_larger_or_smaller_than_y(list_classes)
a = qa_utils.get_ans_is_x_larger_or_smaller_than_y(combinations, gt, DICT_CLASS)
print(a)


a = qa_utils.get_ans_is_there_any_x(list_class, gt, DICT_CLASS)
print(a)


q, encoded_locations = qa_utils.generate_ques_is_x_in_z(
    list_class, (256, 256))
a = qa_utils.get_ans_is_x_in_z(list_class, encoded_locations, gt, DICT_CLASS)
 
b = 2


print(qa_utils.get_ans_which_patient(gt_path))


print(qa_utils.generate_ques_how_many_pixels_of_x(list_class))
print(qa_utils.get_ans_how_many_pixels_of_x(list_class, gt, DICT_CLASS))

print(qa_utils.generate_ques_how_many_percent_of_x(list_class))
print(qa_utils.get_ans_how_many_percent_of_x(list_class, gt, DICT_CLASS))