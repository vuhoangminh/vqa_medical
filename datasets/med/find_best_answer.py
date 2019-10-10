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
LOGS_DIR = os.path.join(PROJECT_DIR, "logs/med")
RAW_DIR = PROJECT_DIR + "/data/vqa_med/raw/raw/"
PROCESSED_QA_PER_QUESTION_PATH = RAW_DIR + "med_qa_per_question.csv"
TEST_DIR = PROJECT_DIR + \
    "/data/raw/vqa_med/VQAMed2019Test/VQAMed2019_Test_ImageList.txt"

TEST_GT = PROJECT_DIR + \
    "/data/raw/vqa_med/VQAMed2019Test/VQAMed2019_Test_Questions_w_Ref_Answers.txt"

SUB_DIR = PROJECT_DIR + "/data/vqa_med/submission/"
path_utils.make_dir(SUB_DIR)


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
    # "minhmul_att_trainval_imagenet_relu", # super-overfit
    # "minhmul_att_trainval_imagenet_h200_g8_relu",
    "minhmul_att_trainval_imagenet_h200_g4_relu", # 59.6
    # "minhmul_att_trainval_imagenet_h100_g8_relu",
    # "minhmul_att_trainval_imagenet_h100_g4_relu",
    # "minhmul_att_trainval_imagenet_h64_g8_relu", # 59.2
    # "minhmul_att_trainval_imagenet_h64_g4_relu",
    # "minhmul_att_trainval_imagenet_h32_g8_relu",
    # "minhmul_att_trainval_imagenet_h32_g4_relu",    
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
    for a in range(1, 4):
        list_answer = list(dict_score[file_id]['answer{}'.format(a)].keys())
        for answer in list_answer:
            if not keys_exists(dict_ans_score, answer):
                dict_ans_score[answer] = DICT_SCORE_MAP['answer{}'.format(
                    a)] * dict_score[file_id]['answer{}'.format(a)][answer]
            else:
                dict_ans_score[answer] += DICT_SCORE_MAP['answer{}'.format(
                    a)] * dict_score[file_id]['answer{}'.format(a)][answer]

    final_answer = ""
    second_answer = ""
    max_value = 0
    for key, value in dict_ans_score.items():
        if value > max_value:
            second_answer = final_answer
            final_answer = key
            max_value = value

    return dict_ans_score, final_answer, second_answer


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
            for a in range(1, 4):
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


def get_top1_ans(dict_score, folder, df, fr=1, to=1, weight=1):
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
            for a in range(1, 4):
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
    with open(TEST_GT, encoding='UTF-8') as f:
        lines = f.readlines()
    list_epoch = list()

    for method in ["best"]:
    # for method in []:
        sub_path  = SUB_DIR + method + ".txt"
        sub = DICT_METHOD[method]

        method_folders = []
        for item in sub:
            for folder in folders:
                if item == path_utils.get_filename_without_extension(folder):
                    method_folders.append(folder)
        method_folders = list(set(method_folders))

        best_acc = 0

        for folder in method_folders:

            try:
                print('>> processing {} of {}'.format(folder, method))
                weight = 1 if "bert" in folder else 5

                for epoch in range(1,100):

                    count = 0
                    correct = 0

                    dict_score = {}
                    # dict_score = get_top1_ans(dict_score, folder, df, weight=weight, fr=90, to=99)
                    dict_score = get_top1_ans(dict_score, folder, df, weight=weight, fr=epoch, to=epoch)

                    for line in lines:
                        line = line.split("|")
                        image, qtype, question, answer = line[0], line[1], line[2], line[3].split("\n")[0]
                        question = question.encode('ascii', 'ignore').decode('ascii')
                        answer = answer.encode('ascii', 'ignore').decode('ascii')

                        file_id = image
                        dict_ans_score, final_answer, second_answer = get_final_answer(
                            dict_score, file_id)

                        count += 1
                        if answer == final_answer:
                            correct += 1
                        # else:
                        #     print ("{} / {}".format(answer, final_answer))
                        # if answer == second_answer:
                        #     correct += 1

                    acc = round(correct/count*100,2)
                    print("Method: {} \t | Epoch: {} \t | Acc: {}".format(folder, epoch, acc))

                    # print("Method: {} \t | Acc: {}".format(folder, acc))
            
            except:
                pass

                    

                    



if __name__ == "__main__":
    main()
