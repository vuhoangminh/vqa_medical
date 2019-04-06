import datasets.utils.print_utils as print_utils
import datasets.utils.paths_utils as path_utils
import datasets.utils.image_utils as image_utils
import datasets.utils.io_utils as io_utils
import datasets.utils.panda_utils as pd_utils
import numpy as np
import pandas as pd
import glob
import os
import json


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
RAW_DIR = PROJECT_DIR + "/data/vqa_med/raw/raw/"
PROCESSED_QA_PER_QUESTION_PATH = RAW_DIR + "med_qa_per_question.csv"

INTERIM_DIR = PROJECT_DIR + "/data/vqa_med/interim/"
path_utils.make_dir(INTERIM_DIR)
train_annotations_filename = INTERIM_DIR + 'train_questions_annotations.json'
val_annotations_filename = INTERIM_DIR + 'val_questions_annotations.json'


def process_df_to_build_json(df):
    train_questions_annotations = pd_utils.create_questions_annotations(
        df, 'train', INTERIM_DIR)
    val_questions_annotations = pd_utils.create_questions_annotations(
        df, 'val', INTERIM_DIR)
    val_questions_annotations = pd_utils.create_questions_annotations(
        df, 'trainval', INTERIM_DIR)        
    test_questions = pd_utils.create_questions(
        df, 'test', INTERIM_DIR, dataset_default="test")
    testdev_questions = pd_utils.create_questions(
        df, 'testdev', INTERIM_DIR, dataset_default="test")


def main():
    print(">> read train val split")
    df = pd.read_csv(PROCESSED_QA_PER_QUESTION_PATH)
    process_df_to_build_json(df)


if __name__ == "__main__":
    main()
