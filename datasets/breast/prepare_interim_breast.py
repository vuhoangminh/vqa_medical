import datasets.utils.svs_utils as svs_utils
import datasets.utils.normalization_utils as normalization_utils
import datasets.utils.print_utils as print_utils
import datasets.utils.paths_utils as path_utils
import datasets.utils.image_utils as image_utils
import datasets.utils.io_utils as io_utils
import datasets.utils.panda_utils as pd_utils
import numpy as np
import pandas as pd
import glob
import os


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
IMAGE_SET_DIR = PROJECT_DIR + \
    "/data/raw/breast-cancer/preprocessed/imagesets/"
RAW_DIR = PROJECT_DIR + "/data/vqa_breast/raw/raw/"
QA_PATH = RAW_DIR + "breast_qa_full.csv"
processed_qa_per_question_path = RAW_DIR + "breast_qa_per_question.csv"

INTERIM_DIR = PROJECT_DIR + "data/vqa_idrid/interim/"
train_annotations_filename = INTERIM_DIR + 'train_questions_annotations.json'
val_annotations_filename = INTERIM_DIR + 'val_questions_annotations.json'


def read_train_val_split():
    segmentation_dict_tool = {k: [] for k in ["train", "val"]}
    
    # read files
    for dataset in segmentation_dict_tool.keys():
        path_write = "{}{}.txt".format(IMAGE_SET_DIR, dataset)
        temp = io_utils.read_file_to_list(path_write)
        my_list = list()
        for i in range(len(temp)):
            my_list.append(temp[i].strip('\n'))
        segmentation_dict_tool[dataset] = my_list
    return segmentation_dict_tool


def process_df_to_build_json(segmentation_dict_tool):
    df = pd.read_csv(QA_PATH)
    df.rename(columns={'File_id': 'file_id'}, inplace=True) 
    df = df.drop(columns = ['Unnamed: 0'])     
    print(">> insert dataset column")
    df = pd_utils.insert_dataset_to_df(df, segmentation_dict_tool)

    print(">> build full df")
    if os.path.exists(processed_qa_per_question_path):
        df = pd.read_csv(processed_qa_per_question_path)  
    else:          
        df = pd_utils.create_full_imageid_quesid_questype(df, INTERIM_DIR)
        df.to_csv(processed_qa_per_question_path)
    return df


def main():
    print(">> read train val split")
    segmentation_dict_tool = read_train_val_split()
    df = process_df_to_build_json(segmentation_dict_tool)
    print("a")

if __name__ == "__main__":
    main()