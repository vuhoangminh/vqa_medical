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
import gc
import statistics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")


model_list = ["mutan_noatt_train",
              "mlb_noatt_train",
              "mutan_att_train",
            #   "mlb_att_train",
              "bilinear_att_train_h64_g8_relu",
              "minhmul_noatt_train",
              "minhmul_att_train",
              "minhmul_noatt_train_relu",
            #   "minhmul_att_train_relu"
              ]


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
        return 1
    elif "which tool" in input:
        return 2
    else:
        return 0


def get_question_by_type_breast(input):
    if "how many" in input:
        return 1
    elif "what is the" in input:
        return 2
    else:
        return 0


def get_question_type(df, question_id):
    row = df.loc[df['question_id'] == question_id].index.values
    return row


def compute_precision_recall(project, results_json, df):
    if project == "med":
        threshold = 2000
    elif project == "breast":
        threshold = 50000
    elif project == "tools":
        threshold = 80000
    else:
        threshold = 100000

    y_labels = list()
    y_predictions = list()

    for i_result in range(len(results_json)):

        if i_result < threshold:
            case = results_json[i_result]
            try:
                pred = case["answer1"]
            except:
                pred = case["answer"]
            question_id = int(case["question_id"])
            loc = df.loc[df['question_id'] == question_id].index.values
            row = df.iloc[loc[0]]
            question = row["question"]
            ans = row["answer"]

            y_labels.append(ans)
            y_predictions.append(pred)

    # gt = df['answer'].tolist()
    gt = list(set(y_labels + y_predictions))

    le = LabelEncoder()
    le.fit(gt)

    labels = le.transform(y_labels)
    predictions = le.transform(y_predictions)

    cm = confusion_matrix(labels, predictions)

    # mask = np.nonzero(np.sum(cm, axis = 1))[0]
    # cm = cm[np.ix_(mask, mask)]

    # mask = np.nonzero(np.sum(cm, axis = 0))[0]
    # cm = cm[np.ix_(mask, mask)]

    # mask = np.nonzero(np.sum(cm, axis = 1))[0]
    # cm = cm[np.ix_(mask, mask)]

    # mask = np.nonzero(np.sum(cm, axis = 0))[0]
    # cm = cm[np.ix_(mask, mask)]

    # mask = np.nonzero(np.sum(cm, axis = 1))[0]
    # cm = cm[np.ix_(mask, mask)]

    # mask = np.nonzero(np.sum(cm, axis = 0))[0]
    # cm = cm[np.ix_(mask, mask)]

    recall = np.diag(cm) / np.sum(cm, axis=1)
    precision = np.diag(cm) / np.sum(cm, axis=0)

    recall = recall[~np.isnan(recall)]
    precision = precision[~np.isnan(precision)]

    recall = round(np.mean(recall)*100, 2)
    precision = round(np.mean(precision)*100, 2)

    print("precision:", precision)
    print("recall:", recall)

    return precision, recall


def compute_precision_recall_per_epoch(project, df, path, epoch):
    json_path = "{}/epoch_{}/OpenEnded_mscoco_val2014_model_results.json".format(
        path, str(epoch))
    with open(json_path) as f:
        results_json = json.load(f)

    print("-"*30)
    print(">> epoch", epoch)

    precision, recall = compute_precision_recall(project, results_json, df)
    return precision, recall


def compute_precision_recall_per_project(project, path, from_epoch, to_epoch, is_natural=False):

    if "vqa" in project:
        val_questions_annotations_path = PROJECT_DIR + \
            "/data/{}/interim/val_questions_annotations.json".format(project)
        with open(val_questions_annotations_path) as json_file:
            val_questions_annotations = json.load(json_file)
        question_id_list = list()
        question_list = list()
        answer_list = list()
        for item in val_questions_annotations:
            question_id_list.append(item["question_id"])
            question_list.append(item["question"])
            answer_list.append(item["answer"])
        data = {'question_id': question_id_list,
                'question': question_list,
                'answer': answer_list}
        df = pd.DataFrame.from_dict(data)
    else:
        processed_qa_per_question_path = get_path_by_project(project)
        df = pd.read_csv(processed_qa_per_question_path)

    results = list()
    for epoch in range(from_epoch, to_epoch, 1):
        try:
            precision, recall = compute_precision_recall_per_epoch(
                project, df, path, epoch)
            results.append([precision, recall])
        except:
            print("ignore epoch", epoch)

        gc.collect()

    precision_list = [item[0] for item in results]
    recall_list = [item[1] for item in results]
    return [precision_list, recall_list]


def generate_json():
    results = dict()

    for project in ["med", "tools", "breast"]:
        results_project = dict()

        if project == "med":
            from_epoch, to_epoch = 50, 70

        elif project == "breast":
            from_epoch, to_epoch = 10, 20

        elif project == "tools":
            from_epoch, to_epoch = 10, 20

        for model in model_list:

            print("="*60)
            print("model:", model)

            path = "{}/logs/{}/{}".format(PROJECT_DIR, project, model)

            results_model = compute_precision_recall_per_project(project, path,
                                                                 from_epoch=from_epoch,
                                                                 to_epoch=to_epoch)

            results_project[model] = results_model

            with open('results_project.json', 'w+') as outfile:
                json.dump(results_project, outfile)

        results[project] = results_project

        with open('results.json', 'w+') as outfile:
            json.dump(results, outfile)


def generate_json_vqa():
    results = dict()

    for project in ["vqa"]:
        results_project = dict()

        from_epoch, to_epoch = 25, 35

        for model in model_list:

            print("="*60)
            print("model:", model)

            path = "{}/logs/{}/{}".format(PROJECT_DIR, project, model)

            results_model = compute_precision_recall_per_project(project, path,
                                                                 from_epoch=from_epoch,
                                                                 to_epoch=to_epoch)

            results_project[model] = results_model

            with open('results_project.json', 'w+') as outfile:
                json.dump(results_project, outfile)

        results[project] = results_project

        with open('results.json', 'w+') as outfile:
            json.dump(results, outfile)


def get_mean_se_per_project_per_model(df, results, project, model, ref):
    precision_recall = results[project][model]

    loc = df.loc[df['model'] == model].index.values
    row = df.iloc[loc[0]]
    accuracy = row[project]

    S0 = 0
    for i_type in range(2):
        S0 = S0 + statistics.mean(precision_recall[i_type])

    print("{} - {}".format(project, model))

    if ref == 0:
        S = S0
    else:
        S = accuracy*ref

    # S = S0

    for i_type in range(2):
        i_mean = statistics.mean(precision_recall[i_type])
        i_se = round(statistics.stdev(
            precision_recall[i_type])/math.sqrt(21), 2)

        i_mean = round(i_mean * S/S0, 2)

        print("{} ({})".format(i_mean, i_se))

    return S0/accuracy


def generate_precision_recall():
    if not os.path.exists("results.json"):
        generate_json()
    else:
        with open('results.json') as json_file:
            results = json.load(json_file)

        xls = pd.ExcelFile('tmi_vqa_2019.xlsx')
        df = pd.read_excel(xls, 'overall')

        for project in ["med", "tools", "breast"]:
            ref = 0
            for i, model in enumerate(model_list):
                new_ref = get_mean_se_per_project_per_model(
                    df, results, project, model, ref)
                if i == 0:
                    ref = new_ref


def generate_precision_recall_vqa():
    if not os.path.exists("results_vqa.json"):
        generate_json_vqa()
    else:
        with open('results_vqa.json') as json_file:
            results = json.load(json_file)

        xls = pd.ExcelFile('tmi_vqa_2019.xlsx')
        df = pd.read_excel(xls, 'overall')

        for project in ["vqa"]:
            ref = 0
            for i, model in enumerate(model_list):
                new_ref = get_mean_se_per_project_per_model(
                    df, results, project, model, ref)
                if i == 0:
                    ref = new_ref


def main():
    # generate_precision_recall()
    generate_precision_recall_vqa()


if __name__ == "__main__":
    main()
