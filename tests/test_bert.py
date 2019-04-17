from vqa.models import sen2vec
import datasets.utils.print_utils as print_utils
import datasets.utils.io_utils as io_utils
import datasets.utils.paths_utils as path_utils
import os
import pandas as pd
import sys
sys.path.append("C:\\Users\\minhm\\Documents\\GitHub\\vqa_idrid")


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
RAW_DIR = PROJECT_DIR + "/data/vqa_med/raw/raw/"
PROCESSED_QA_PER_QUESTION_PATH = RAW_DIR + "med_qa_per_question.csv"
EXTRACTED_QUES_FEATURES_PATH = RAW_DIR + "question_features.pickle"
BASE_EXTRACTED_QUES_FEATURES_PATH = RAW_DIR + "question_features_base.pickle"

e = io_utils.read_pickle(EXTRACTED_QUES_FEATURES_PATH)
b = io_utils.read_pickle(BASE_EXTRACTED_QUES_FEATURES_PATH)

print("a")
