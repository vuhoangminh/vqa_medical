import os
import datasets.utils.print_utils as print_utils
import datasets.utils.paths_utils as path_utils
import datasets.utils.image_utils as image_utils
import datasets.utils.io_utils as io_utils
from PIL import Image
import numpy as np
import pandas as pd
import glob
import shutil
import random
random.seed(1988)


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
DATASETS_TRAIN_DIR = PROJECT_DIR + \
    "/data/raw/vqa_med/ImageClef-2019-VQA-Med-Training/Train_images/"
DATASETS_VALID_DIR = PROJECT_DIR + \
    "/data/raw/vqa_med/ImageClef-2019-VQA-Med-Validation/Val_images/"
DATASETS_TRAIN_DIR = PROJECT_DIR + \
    "/data/raw/vqa_med/VQAMed2019Test/VQAMed2019_Test_Images/"

DATASETS_TRAIN_TXT = PROJECT_DIR + \
    "/data/raw/vqa_med/ImageClef-2019-VQA-Med-Training/All_QA_Pairs_train.txt"


# PREPROCESSED_IMAGE_WSI_DIR = PROJECT_DIR + \
#     "/data/raw/breast-cancer/preprocessed/WSI/"
# IMAGE_SET_DIR = PROJECT_DIR + \
#     "/data/raw/breast-cancer/preprocessed/imagesets/"
# PREPROCESSED_IMAGE_WSI_RAW_DIR = PREPROCESSED_IMAGE_WSI_DIR + "raw/"
# PREPROCESSED_IMAGE_WSI_PATCH_DIR = PREPROCESSED_IMAGE_WSI_DIR + "patch/"
# PREPROCESSED_IMAGE_WSI_GT_DIR = PREPROCESSED_IMAGE_WSI_DIR + "patch_gt/"


# path_utils.make_dir(PREPROCESSED_IMAGE_PHOTOS_DIR)
# path_utils.make_dir(PREPROCESSED_IMAGE_WSI_DIR)
# path_utils.make_dir(PREPROCESSED_IMAGE_WSI_PATCH_DIR)
# path_utils.make_dir(PREPROCESSED_IMAGE_WSI_GT_DIR)
# path_utils.make_dir(IMAGE_SET_DIR)
# path_utils.make_dir(PREPROCESSED_IMAGE_WSI_RAW_DIR)


LIST_BREAST_CLASS = ["Benign", "InSitu", "Invasive", "Normal"]

LIST_PLANE = {
    "axial": "axial",
    "sagittal": "sagittal",
    "coronal": "coronal",
    "ap": "ap",
    "lateral": "lateral",
    "frontal": "frontal",
    "pa": "pa",
    "transverse": "transverse",
    "oblique": "oblique",
    "longitudinal": "longitudinal",
    "decubitus": "decubitus",
    "3d reconstruction": "reconstruction",
    "mammo - mlo": "mlo",
    "mammo - cc": "cc",
    "mammo - mag cc": "mag",
    "mammo - xcc": "xcc",
}

LIST_ORGAN = {
    "breast": "breast",
    "skull": "skull",
    "face": "face",
    "spine": "spine",
    "musculoskeletal": "musculoskeletal",
    "heart": "heart",
    "lung": "lung",
    "gastrointestinal": "gastrointestinal",
    "genitourinary": "genitourinary",
    "vascular": "vascular",
}

LIST_MODALITY = {
    "xr - plain film": "xr_plain",
    "ct - noncontrast": "ct_noncontrast",
    "ct w/contrast (iv)": "ct_wcontrast",
    "ct - gi & iv contrast": "ct_giiv",
    "cta - ct angiography": "cta",
    "ct - gi contrast": "ct_gi",
    "ct - myelogram": "ct_myelogram",
    "tomography": "tomography",
    "mr - t1w w/gadolinium": "mr_t1w_wgadolinium",
    "mr - t1w - noncontrast": "mr_t1w_noncontrast",
    "mr - t2 weighted": "mr_t2_weighted",
    "mr - flair": "mr_flair",
    "mr - t1w w/gd (fat suppressed)": "mr_t1w_wfat",
    "mr t2* gradient,gre,mpgr,swan,swi": "mr_t2_mpgr",
    "mr - dwi diffusion weighted": "mr_dwi",
    "mra - mr angiography/venography": "mra_mr_angiography",
    "mr - other pulse seq.": "mr_other",
    "mr - adc map (app diff coeff)": "mr_adc",
    "mr - pdw proton density": "mr_pdw",
    "mr - stir": "mr_stir",
    "mr - fiesta": "mr_fiesta",
    "mr - flair w/gd": "mr_flair_wgd",
    "mr - t1w spgr": "mr_t1w_spgr",
    "mr - t2 flair w/contrast": "mr_t2_flair",
    "mr t2* gradient gre": "mr_t2_gradientgre",
    "us - ultrasound": "us",
    "us-d - doppler ultrasound": "usd",
    "mammograph": "mammograph",
    "bas - barium swallow": "bas",
    "ugi - upper gi": "ugi",
    "be - barium enema": "be",
    "sbft - small bowel": "sbft",
    "an - angiogram": "angiogram",
    "venogram": "venogram",
    "nm - nuclear medicine": "nm",
    "pet - positron emission": "pet",
}


def get_class_image_model(df, line):
    line = line.split("|")
    image, question, answer = line[0], line[1], line[2].split("\n")[0]
    plane_keys = list(LIST_PLANE.keys())
    organ_keys = list(LIST_ORGAN.keys())
    modality_keys = list(LIST_MODALITY.keys())

    modality, plane, organ,  = "", "", ""

    for key in plane_keys:
        if key in answer and "plane" in question:
            plane = LIST_PLANE[key]
    for key in organ_keys:
        if key in answer:
            organ = LIST_ORGAN[key]
    for key in modality_keys:
        if key in answer:
            modality = LIST_MODALITY[key]

    index = df.index[df['image'] == image].tolist()
    if plane != "":
        if len(index) == 0:
            df = df.append(pd.DataFrame({"image": [image],
                                         "modality": [""],
                                         "plane": [plane],
                                         "organ": [""]}), ignore_index=True)
        else:
            df.at[index[0], 'plane'] = plane
    if organ != "":
        if len(index) == 0:
            df = df.append(pd.DataFrame({"image": [image],
                                          "modality": [""],
                                          "plane": [""],
                                          "organ": [organ]}), ignore_index=True)
        else:
            df.at[index[0], 'organ'] = organ
    if modality != "":
        if len(index) == 0:
            df = df.append(pd.DataFrame({"image": [image],
                                          "modality": [modality],
                                          "plane": [""],
                                          "organ": [""]}), ignore_index=True)
        else:
            df.at[index[0], 'modality'] = modality

    return df


def main(overwrite=False):
    with open(DATASETS_TRAIN_TXT) as f:
        lines = f.readlines()

    df = pd.DataFrame(columns=['image', 'modality', 'plane', 'organ'])
    for line in lines:
        df = get_class_image_model(df, line)

    print(df)

    n_group = len(set(zip(df['plane'], df['organ'], df['modality'])))
    print(n_group)


if __name__ == "__main__":
    main(overwrite=False)
