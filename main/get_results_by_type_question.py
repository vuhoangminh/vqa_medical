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
RAW_DIR = PROJECT_DIR + "/data/vqa_breast/raw/raw/"
QA_PATH = RAW_DIR + "breast_qa_full.csv"
processed_qa_per_question_path = RAW_DIR + "breast_qa_per_question.csv"


# keep num = 20 please
def get_val_acc1_from_json(json_path, num=20):
    with open(json_path) as f:
        data = json.load(f)
    val = data["logged"]["val"]["acc1"]
    # pprint(val)
    list_acc1 = []
    to = len(val)
    fr = max((to + 1 - num, 1))
    for i in range(fr, to+1):
        acc1 = val["{}".format(str(i))]
        list_acc1.append(acc1)
    return list_acc1, to


def main():
    print(">> read train val split")
    df = pd.read_csv(processed_qa_per_question_path)
    print(df.head())

    print(df.question.unique())


if __name__ == "__main__":
    main()
