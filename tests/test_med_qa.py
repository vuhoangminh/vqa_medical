import os
import glob
import re
import unidecode
import pandas as pd
from tqdm import tqdm
import datasets.utils.paths_utils as path_utils
import datasets.utils.image_utils as image_utils


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
RAW_DIR = PROJECT_DIR + "/data/vqa_med/raw/raw/"
PROCESSED_QA_PER_QUESTION_PATH = RAW_DIR + "med_qa_per_question.csv"

df = pd.read_csv(PROCESSED_QA_PER_QUESTION_PATH)

print(df["answer"].value_counts())