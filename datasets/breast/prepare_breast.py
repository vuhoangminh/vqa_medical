import os
import glob
import pandas as pd
import datasets.utils.paths_utils as path_utils


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
DATASETS_DIR = PROJECT_DIR + "/data/raw/breast-cancer/ICIAR2018_BACH_Challenge/"
PREPROCESSED_IMAGE_DIR = PROJECT_DIR + "/data/raw/breast-cancer/preprocessed/"
RAW_DIR = PROJECT_DIR + "/data/vqa_breast/raw/raw/"
path_utils.make_dir(PREPROCESSED_IMAGE_DIR)
path_utils.make_dir(RAW_DIR)


def test():
    print("lets go")

def main():
    test()

if __name__ == "__main__":
    main()
