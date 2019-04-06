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

DATASETS_TRAIN_DIR = PROJECT_DIR + \
    "/data/raw/vqa_med/ImageClef-2019-VQA-Med-Training/Train_images/"
DATASETS_VALID_DIR = PROJECT_DIR + \
    "/data/raw/vqa_med/ImageClef-2019-VQA-Med-Validation/Val_images/"
DATASETS_TEST_DIR = PROJECT_DIR + \
    "/data/raw/vqa_med/VQAMed2019Test/VQAMed2019_Test_Images/"

QA_TRAIN_TXT = PROJECT_DIR + \
    "/data/raw/vqa_med/ImageClef-2019-VQA-Med-Training/All_QA_Pairs_train.txt"
QA_VALID_TXT = PROJECT_DIR + \
    "/data/raw/vqa_med/ImageClef-2019-VQA-Med-Validation/All_QA_Pairs_val.txt"
QA_TEST_TXT = PROJECT_DIR + \
    "/data/raw/vqa_med/VQAMed2019Test/VQAMed2019_Test_Questions.txt"
DATASETS_TRAIN_TXT = PROJECT_DIR + \
    "/data/raw/vqa_med/ImageClef-2019-VQA-Med-Training/All_QA_Pairs_train.txt"
RAW_DIR = PROJECT_DIR + "/data/vqa_med/raw/raw/"
PROCESSED_QA_PER_QUESTION_PATH = RAW_DIR + "med_qa_per_question.csv"
IMAGEID_PATH = RAW_DIR + "image_id.csv"
path_utils.make_dir(RAW_DIR)


def generate_image_id():
    img_train_paths = glob.glob(os.path.join(DATASETS_TRAIN_DIR, "*.jpg"))
    img_val_paths = glob.glob(os.path.join(DATASETS_VALID_DIR, "*.jpg"))
    img_test_paths = glob.glob(os.path.join(DATASETS_TEST_DIR, "*.jpg"))

    img_paths = img_train_paths + img_val_paths + img_test_paths

    df = pd.DataFrame(columns=['file_id', 'image_id'])
    image_id_path = RAW_DIR

    for index, path in enumerate(img_paths):
        file_id = path_utils.get_filename(path)
        image_id = index
        df = df.append(pd.DataFrame({"file_id": [file_id],
                                     "image_id": [index]}), ignore_index=True)

    df.to_csv(IMAGEID_PATH, index=False)
    return df


def prepare_qa_per_question(full_df, cols, df_id, dataset="train"):

    if dataset == "train":
        qa_path = QA_TRAIN_TXT
    elif dataset == "val":
        qa_path = QA_VALID_TXT
    else:
        qa_path = QA_TEST_TXT

    with open(qa_path, encoding='UTF-8') as f:
        lines = f.readlines()

    temp_df = pd.DataFrame(columns=cols)
    for index in tqdm(range(len(lines))):
        line = lines[index]
        line = line.split("|")

        if dataset in ["train", "val"]:
            image, question, answer = line[0], line[1], line[2].split("\n")[0]
            question = question.encode('ascii', 'ignore').decode('ascii')
            answer = answer.encode('ascii', 'ignore').decode('ascii')
        else:
            image, question = line[0], line[1].split("\n")[0]
            question = question.encode('ascii', 'ignore').decode('ascii')

        file_id = image + ".jpg"
        image_id = str(int(df_id.at[df_id.index[df_id['file_id'] == file_id].tolist()[
            0], "image_id"])).zfill(6)

        question_i = len(full_df.index[full_df['file_id'] == file_id].tolist())
        question_id = str(image_id + str(question_i).zfill(6)).zfill(12)

        if dataset in ["train", "val"]:
            row_df = pd.DataFrame({'file_id': [file_id],
                                   'image_id': [image_id],
                                   'question': [question],
                                   'question_id': [question_id],
                                   'question_type': ['is the'],
                                   'answer': [answer],
                                   'multiple_choice_answer': [answer],
                                   'answer_confidence': ['yes'],
                                   'answer_id': [1],
                                   'dataset': [dataset]
                                   })
        else:
            row_df = pd.DataFrame({'file_id': [file_id],
                                   'image_id': [image_id],
                                   'question': [question],
                                   'question_id': [question_id],
                                   'question_type': [""],
                                   'answer': [""],
                                   'multiple_choice_answer': [""],
                                   'answer_confidence': [""],
                                   'answer_id': [""],
                                   'dataset': [dataset]
                                   })

        if len(temp_df) == 0:
            temp_df = row_df
        else:
            temp_df = temp_df.append(row_df, ignore_index=True)

        if index % 100 == 0 and index > 0:
            if len(full_df) == 0:
                full_df = temp_df
            else:
                full_df = full_df.append(temp_df)
            temp_df = pd.DataFrame(columns=cols)

    full_df = full_df.append(temp_df)

    return full_df


def process(dataset):
    if dataset == "train":
        qa_path = QA_TRAIN_TXT
    elif dataset == "val":
        qa_path = QA_VALID_TXT
    else:
        qa_path = QA_TEST_TXT

    with open(qa_path) as f:
        lines = f.readlines()

    df = pd.DataFrame(columns=['image', 'modality', 'plane', 'organ'])
    for index in tqdm(range(len(lines))):
        line = lines[index]


def main():
    if os.path.exists(IMAGEID_PATH):
        df_id = pd.read_csv(IMAGEID_PATH)
    else:
        df_id = generate_image_id()

    if os.path.exists(PROCESSED_QA_PER_QUESTION_PATH):
        full_df = pd.read_csv(PROCESSED_QA_PER_QUESTION_PATH)
    else:
        cols = ['file_id', 'image_id', 'question', 'question_id',
                'question_type', 'answer', 'multiple_choice_answer',
                'answer', 'answer_confidence', 'answer_id',
                'dataset']
        full_df = pd.DataFrame(columns=cols)
        full_df = prepare_qa_per_question(
            full_df, cols, df_id, dataset="val")
        full_df.to_csv(PROCESSED_QA_PER_QUESTION_PATH, index=False)
        full_df = prepare_qa_per_question(
            full_df, cols, df_id, dataset="test")
        full_df = prepare_qa_per_question(
            full_df, cols, df_id, dataset="train")
        full_df.to_csv(PROCESSED_QA_PER_QUESTION_PATH, index=False)


if __name__ == "__main__":
    main()
