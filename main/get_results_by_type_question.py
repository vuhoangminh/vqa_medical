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
import re
import unidecode
import math


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")


def get_path_by_project(project):
    RAW_DIR = PROJECT_DIR + "/data/vqa_{}/raw/raw/".format(project)
    QA_PATH = RAW_DIR + "{}_qa_full.csv".format(project)
    processed_qa_per_question_path = RAW_DIR + \
        "{}_qa_per_question.csv".format(project)
    return processed_qa_per_question_path


def get_question_by_type_med(input):
    # if project == "med":
    All_QA_Pairs_val = PROJECT_DIR + \
        "/data/raw/vqa_med/ImageClef-2019-VQA-Med-Validation/All_QA_Pairs_val.txt"
    qa_path = All_QA_Pairs_val

    with open(qa_path, encoding='UTF-8') as f:
        lines = f.readlines()

    list_question = list()

    for index in range(len(lines)):
        line = lines[index]
        line = line.split("|")

        answer = ""

        image, question, answer = line[0], line[1], line[2].split("\n")[0]
        question = question.encode('ascii', 'ignore').decode('ascii')
        answer = answer.encode('ascii', 'ignore').decode('ascii')

        list_question.append(question)

        if input in question:
            return math.floor(index/500)


def get_question_by_type_tools(input):
    if "how many" in input:
        return 2
    elif "which tool" in input:
        return 3
    else:
        return 1


def get_question_by_type_breast(input):
    if "how many" in input:
        return 2
    elif "what is the" in input:
        return 3
    else:
        return 1


def get_question_type(df, question_id):
    row = df.loc[df['question_id'] == question_id].index.values
    return row


def compute_accuracy_per_epoch(project, results_json, df):
    if project == "med":
        score = [0 for i in range(4)]
        count = [0 for i in range(4)]
    elif project == "breast":
        score = [0 for i in range(3)]
        count = [0 for i in range(3)]
    elif project == "tools":
        score = [0 for i in range(3)]
        count = [0 for i in range(3)]

    for i_result in range(len(results_json)):

        if i_result > 0 and i_result % 10000 == 0:
            print("process {}/{}".format(i_result, len(results_json)))
            print(score)

        case = results_json[i_result]
        pred = case["answer1"]
        question_id = int(case["question_id"])
        loc = df.loc[df['question_id'] == question_id].index.values
        row = df.iloc[loc[0]]
        question = row["question"]
        ans = row["answer"]

        question_type = get_question_by_type_tools(question)

        if ans == pred:
            score[question_type] = score[question_type] + 1
        count[question_type] = count[question_type] + 1

    accuracy_per_type = [i / j for i, j in zip(score, count)]

    return accuracy_per_type


def get_accuracy_per_type_per_epoch(project, df, path, epoch):
    json_path = "{}/epoch_{}/OpenEnded_mscoco_val2014_model_results.json".format(
        path, str(epoch))
    with open(json_path) as f:
        results_json = json.load(f)

    print("="*30)
    print("processing epoch", epoch)
    print("="*30)
    return compute_accuracy_per_epoch(project, results_json, df)


def compute_mean_std_per_type_per_project(project, path, from_epoch, to_epoch):
    processed_qa_per_question_path = get_path_by_project(project)
    df = pd.read_csv(processed_qa_per_question_path)

    results = list()
    for epoch in range(from_epoch, to_epoch, 1):    
        accuracy_per_type_per_epoch = get_accuracy_per_type_per_epoch(project, df, path, epoch)
        results.append(accuracy_per_type_per_epoch)

    if project == "med":
        modality_list = [item[0] for item in results]
        plane_list = [item[1] for item in results]
        organ_list = [item[2] for item in results]
        abnor_list = [item[3] for item in results]
        return [modality_list, plane_list, organ_list, abnor_list]

    elif project == "breast":
        yesno_list = [item[0] for item in results]
        number_list = [item[1] for item in results]
        name_list = [item[2] for item in results]
        return [yesno_list, number_list, name_list]

    elif project == "tools":
        yesno_list = [item[0] for item in results]
        number_list = [item[1] for item in results]
        position_list = [item[2] for item in results]
        return [yesno_list, number_list, position_list]


def main():
    print(">> read train val split")

    project = "breast"
    # project = "med"
    # project = "tools"

    path = "{}/{}".format(PROJECT_DIR + project)

    compute_mean_std_per_type_per_project(project, path, from_epoch=10, to_epoch=20)


if __name__ == "__main__":
    main()
