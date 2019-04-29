import sys
import os
if os.path.isdir("C:\\Users\\minhm\\Documents\\GitHub\\vqa_idrid"):
    sys.path.append("C:\\Users\\minhm\\Documents\\GitHub\\vqa_idrid")
import datasets.utils.print_utils as print_utils
import datasets.utils.paths_utils as path_utils
import pandas as pd
import numpy as np
from pprint import pprint
import glob
import json


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs/med/trainval")
RAW_DIR = PROJECT_DIR + "/data/vqa_med/raw/raw/"
PROCESSED_QA_PER_QUESTION_PATH = RAW_DIR + "med_qa_per_question.csv"
TEST_DIR = PROJECT_DIR + \
    "/data/raw/vqa_med/VQAMed2019Test/VQAMed2019_Test_ImageList.txt"
SUB_DIR = PROJECT_DIR + "/data/vqa_med/submission/"
path_utils.make_dir(SUB_DIR)


# SUB_QC_MLB = [
#     "minhmul_att_trainval_imagenet_h200_g4_relu",
#     "minhmul_att_trainval_imagenet_h200_g4_relu_bert_uncased",
#     "minhmul_att_trainval_imagenet_h200_g4_relu_bert_cased",
#     "minhmul_att_trainval_imagenet_h200_g4_relu_bert_uncased_768",
#     "minhmul_att_trainval_imagenet_h200_g4_relu_bert_cased_768",
#     "minhmul_att_trainval_imagenet_h200_g8_relu",
#     "minhmul_att_trainval_imagenet_h400_g8_relu",
#     "minhmul_att_trainval_imagenet_h100_g8_relu",
#     "minhmul_att_trainval_imagenet_h100_g8_relu_bert_uncased",
#     "minhmul_att_trainval_imagenet_h100_g8_relu_bert_cased",
#     "minhmul_att_trainval_imagenet_h100_g8_relu_bert_uncased_768",
#     "minhmul_att_trainval_imagenet_h100_g8_relu_bert_cased_768",
# ]


# SUB_BILINEAR = [
#     "bilinear_att_trainval_imagenet_h200_g4_relu",
#     "bilinear_att_trainval_imagenet_h200_g4_relu_bert_uncased",
#     "bilinear_att_trainval_imagenet_h200_g4_relu_bert_cased",
#     "bilinear_att_trainval_imagenet_h200_g4_relu_bert_uncased_768",
#     "bilinear_att_trainval_imagenet_h200_g4_relu_bert_cased_768",
#     "bilinear_att_trainval_imagenet_h100_g8_relu",
#     "bilinear_att_trainval_imagenet_h100_g8_relu_bert_uncased",
#     "bilinear_att_trainval_imagenet_h100_g8_relu_bert_cased",
#     "bilinear_att_trainval_imagenet_h100_g8_relu_bert_uncased_768",
#     "bilinear_att_trainval_imagenet_h100_g8_relu_bert_cased_768",
# ]


# SUB_SKIP_THOUGHTS = [
#     "minhmul_att_trainval_imagenet_h200_g4_relu",
#     "minhmul_att_trainval_imagenet_h200_g8_relu",
#     "minhmul_att_trainval_imagenet_h400_g8_relu",
#     "minhmul_att_trainval_imagenet_h100_g8_relu",
#     "bilinear_att_trainval_imagenet_h200_g4_relu",
#     "bilinear_att_trainval_imagenet_h100_g8_relu",
# ]


# SUB_BERT_3072 = [
#     "minhmul_att_trainval_imagenet_h200_g4_relu_bert_uncased",
#     "minhmul_att_trainval_imagenet_h200_g4_relu_bert_cased",
#     "minhmul_att_trainval_imagenet_h100_g8_relu_bert_uncased",
#     "minhmul_att_trainval_imagenet_h100_g8_relu_bert_cased",
#     "bilinear_att_trainval_imagenet_h200_g4_relu_bert_uncased",
#     "bilinear_att_trainval_imagenet_h200_g4_relu_bert_cased",
#     "bilinear_att_trainval_imagenet_h100_g8_relu_bert_uncased",
#     "bilinear_att_trainval_imagenet_h100_g8_relu_bert_cased",
# ]


# SUB_BERT_768 = [
#     "minhmul_att_trainval_imagenet_h200_g4_relu_bert_uncased_768",
#     "minhmul_att_trainval_imagenet_h200_g4_relu_bert_cased_768",
#     "minhmul_att_trainval_imagenet_h100_g8_relu_bert_uncased_768",
#     "minhmul_att_trainval_imagenet_h100_g8_relu_bert_cased_768",
#     "bilinear_att_trainval_imagenet_h200_g4_relu_bert_uncased_768",
#     "bilinear_att_trainval_imagenet_h200_g4_relu_bert_cased_768",
#     "bilinear_att_trainval_imagenet_h100_g8_relu_bert_uncased_768",
#     "bilinear_att_trainval_imagenet_h100_g8_relu_bert_cased_768",
# ]


SUB_SKIP_THOUGHTS = [
    "mutan_att_trainval_imagenet_relu",
    "mlb_att_trainval_imagenet_h200_g4_relu",
    "mlb_att_trainval_imagenet_h100_g8_relu",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu",
]


SUB_BERT_3072 = [
    "mutan_att_trainval_imagenet_relu_bert_cased",
    "mutan_att_trainval_imagenet_relu_bert_uncased",
    "mlb_att_trainval_imagenet_h200_g4_relu_bert_uncased",
    "mlb_att_trainval_imagenet_h200_g4_relu_bert_cased",
    "mlb_att_trainval_imagenet_h100_g8_relu_bert_uncased",
    "mlb_att_trainval_imagenet_h100_g8_relu_bert_cased",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_uncased",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_cased",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_uncased",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_cased",
]


SUB_BERT_768 = [
    "mutan_att_trainval_imagenet_relu_bert_cased_768",
    "mutan_att_trainval_imagenet_relu_bert_uncased_768",
    "mlb_att_trainval_imagenet_h200_g4_relu_bert_uncased_768",
    "mlb_att_trainval_imagenet_h200_g4_relu_bert_cased_768",
    "mlb_att_trainval_imagenet_h100_g8_relu_bert_uncased_768",
    "mlb_att_trainval_imagenet_h100_g8_relu_bert_cased_768",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_uncased_768",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_cased_768",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_uncased_768",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_cased_768",
]


SUB_GLOBAL_BILINEAR = [
    "globalbilinear_att_trainval_imagenet_h200_g4_relu",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_uncased",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_cased",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_uncased_768",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_cased_768",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_uncased",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_cased",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_uncased_768",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_cased_768",    
]


SUB_ALL = [
    "mutan_att_trainval_imagenet_relu_bert_cased_768",
    "mutan_att_trainval_imagenet_relu_bert_uncased_768",
    "mutan_att_trainval_imagenet_relu_bert_cased",
    "mutan_att_trainval_imagenet_relu_bert_uncased",
    "mutan_att_trainval_imagenet_relu",
    "mlb_att_trainval_imagenet_h200_g4_relu",
    "mlb_att_trainval_imagenet_h200_g4_relu_bert_uncased",
    "mlb_att_trainval_imagenet_h200_g4_relu_bert_cased",
    "mlb_att_trainval_imagenet_h200_g4_relu_bert_uncased_768",
    "mlb_att_trainval_imagenet_h200_g4_relu_bert_cased_768",
    "mlb_att_trainval_imagenet_h100_g8_relu",
    "mlb_att_trainval_imagenet_h100_g8_relu_bert_uncased",
    "mlb_att_trainval_imagenet_h100_g8_relu_bert_cased",
    "mlb_att_trainval_imagenet_h100_g8_relu_bert_uncased_768",
    "mlb_att_trainval_imagenet_h100_g8_relu_bert_cased_768",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_uncased",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_cased",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_uncased_768",
    "globalbilinear_att_trainval_imagenet_h200_g4_relu_bert_cased_768",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_uncased",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_cased",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_uncased_768",
    "globalbilinear_att_trainval_imagenet_h100_g8_relu_bert_cased_768",
]

SUB_BEST = [
    "globalbilinear_att_trainval_imagenet_h64_g8_relu",
]


DICT_METHOD = {
    "globalbilinear": SUB_GLOBAL_BILINEAR,
    "skipthoughts": SUB_SKIP_THOUGHTS,
    "bert3072": SUB_BERT_3072,
    "bert768": SUB_BERT_768,
    "all": SUB_ALL,
    "best": SUB_BEST
}


DICT_SCORE_MAP = {
    'answer1': 81,
    'answer2': 27,
    'answer3': 9,
    'answer4': 3,
    'answer5': 1,
}

DICT_SCORE_MAP = {
    'answer1': 16,
    'answer2': 8,
    'answer3': 4,
    'answer4': 2,
    'answer5': 1,
}


def show_sample(df, question, answer):
    return 0


def get_info(df, question_id):
    index = df.index[df['question_id'] == int(question_id)].tolist()
    row = df.iloc[index[0]]
    return row


def keys_exists(element, *keys):
    '''
    Check if *keys (nested) exists in `element` (dict).
    '''
    if type(element) is not dict:
        raise AttributeError('keys_exists() expects dict as first argument.')
    if len(keys) == 0:
        raise AttributeError(
            'keys_exists() expects at least two arguments, one given.')

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


def get_final_answer(dict_score, file_id):
    dict_ans_score = {}
    for a in range(1, 6):
        list_answer = list(dict_score[file_id]['answer{}'.format(a)].keys())
        for answer in list_answer:
            if not keys_exists(dict_ans_score, answer):
                dict_ans_score[answer] = DICT_SCORE_MAP['answer{}'.format(
                    a)] * dict_score[file_id]['answer{}'.format(a)][answer]
            else:
                dict_ans_score[answer] += DICT_SCORE_MAP['answer{}'.format(
                    a)] * dict_score[file_id]['answer{}'.format(a)][answer]

    final_answer = ""
    max_value = 0
    for key, value in dict_ans_score.items():
        if value > max_value:
            final_answer = key
            max_value = value

    return dict_ans_score, final_answer


def get_ans(dict_score, folder, df, fr=79, to=99, weight=1):
    list_epochs = list(range(fr, to+1))
    for epoch in list_epochs:
        folder_epoch_json = "{}/epoch_{}/vqa_OpenEnded_mscoco_test2015_model_results.json".format(
            folder, epoch)
        with open(folder_epoch_json) as json_file:
            data = json.load(json_file)
        for q in range(len(data)):
            question_id = data[q]["question_id"]
            row_info = get_info(df, question_id)
            file_id = row_info["file_id"]
            file_id = path_utils.get_filename_without_extension(file_id)
            for a in range(1, 6):
                ans = data[q]['answer{}'.format(a)]

                if not keys_exists(dict_score, file_id):
                    dict_score[file_id] = {'answer{}'.format(a): {ans: 1*weight}}
                elif not keys_exists(dict_score, file_id, 'answer{}'.format(a)):
                    dict_score[file_id]['answer{}'.format(a)] = {ans: 1*weight}
                elif not keys_exists(dict_score, file_id, 'answer{}'.format(a), ans):
                    dict_score[file_id]['answer{}'.format(a)][ans] = 1*weight
                else:
                    dict_score[file_id]['answer{}'.format(a)][ans] += 1*weight
    return dict_score


def main():
    folders = glob.glob(os.path.join(LOGS_DIR, "*"))
    df = pd.read_csv(PROCESSED_QA_PER_QUESTION_PATH)
    with open(TEST_DIR, encoding='UTF-8') as f:
        lines = f.readlines()
    dict_score = {}

    # for method in ["qcmlb", "bilinear", "skipthoughts", "bert3072", "bert768", "all"]:
    for method in ["best", "globalbilinear", "skipthoughts", "bert3072", "bert768", "all"]:     
    # for method in []:
        sub_path  = SUB_DIR + method + ".txt"
        sub = DICT_METHOD[method]

        method_folders = []
        for item in sub:
            for folder in folders:
                if item == path_utils.get_filename_without_extension(folder):
                    method_folders.append(folder)
        method_folders = list(set(method_folders))

        for folder in method_folders:
            print('>> processing {} of {}'.format(folder, method))
            weight = 1 if "bert" in folder else 5
            dict_score = get_ans(dict_score, folder, df, weight=weight)

        f = open(sub_path, 'w')
        for line in lines:
            file_id = line.split('\n')[0]
            dict_ans_score, final_answer = get_final_answer(
                dict_score, file_id)
            f.write('{}|{}\n'.format(file_id, final_answer))  # python will convert \n to os.linesep
        f.close()  # you can omit in most cases as the destructor will call it


if __name__ == "__main__":
    main()
