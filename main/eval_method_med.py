import json
import os
import glob
from pprint import pprint
import numpy as np
import pandas as pd

import datasets.utils.paths_utils as path_utils

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
LOGS_DIR = PROJECT_DIR + "/logs/"


def get_val_acc1_from_json_old(json_path, num=20):
    with open(json_path) as f:
        data = json.load(f)
    val = data["logged"]["val"]["acc1"]
    # pprint(val)
    list_acc1 = []
    max_acc1 = max(list(val.values()))
    for key, value in val.items():
        if value == max_acc1:
            to = int(key)
    fr = max((to + 1 - num, 1))
    for i in range(fr, to+1):
        acc1 = val["{}".format(str(i))]
        list_acc1.append(acc1)
    return list_acc1, to


# Function returns N largest elements
def Nmaxelements(list1, N):
    final_list = []
    for i in range(0, N):
        max1 = 0
        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j]
        list1.remove(max1)
        final_list.append(max1)
    return final_list


def get_val_acc1_from_json(json_path, num=20):
    with open(json_path) as f:
        data = json.load(f)
    val = data["logged"]["val"]["acc1"]
    # pprint(val)
    list_acc1 = []
    max_acc1 = max(list(val.values()))
    for key, value in val.items():
        if value == max_acc1:
            to = int(key)
    list_acc1 = Nmaxelements(list(val.values()), N=num)
    return list_acc1, to


def process_one_method_one_dataset(folder):
    json_path = "{}/logger.json".format(folder)
    if os.path.exists(json_path):
        list_acc1, to = get_val_acc1_from_json(json_path)
        mean = np.mean(list_acc1)
        std = np.std(list_acc1)
    else:
        mean, std, to = 0, 0, 0
    return mean, std, to


def get_info(name):
    name = name
    if "minhmul" in name:
        method = "qc-mlb"
    elif "globalbilinear" in name:
        method = "globalbilinear"
    elif "mlb" in name:
        method = "mlb"
    elif "mutan" in name:
        method = "mutan"
    else:
        method = "bilinear"
    image = "imagenet" if "imagenet" in name else "classif"
    question = "bert" if "bert" in name else "skip-thoughts"
    activation = "relu" if "relu" in name else "tanh"
    if "h100" in name:
        dim_h = '100'
    elif "h64" in name:
        dim_h = '64'
    elif "h200" in name:
        dim_h = '200'
    elif "h400" in name:
        dim_h = '400'
    else:
        dim_h = "1200"
    if "g8" in name:
        nb_glimpses = '8'
    elif "g16" in name:
        nb_glimpses = '16'
    else:
        nb_glimpses = '4'
    return [name, method, image, question, dim_h, nb_glimpses, activation]


def main():
    for dataset in ["med"]:
        folders = glob.glob(os.path.join(LOGS_DIR, dataset, "train", "*"))
        df_path = os.path.join(LOGS_DIR, dataset, 'compile.csv')
        folders = [x for x in folders if "trainval" not in x]
        df = pd.DataFrame(columns=['name', 'method', 'image', 'question',
                                   'dim_h', 'nb_glimpses', 'activation', 'epoch_max', 'mean', 'std'])
        for folder in folders:
            mean, std, to = process_one_method_one_dataset(folder)
            # print("{}: \t {:.2f}({:.2f})".format(
            #     path_utils.get_filename_without_extension(folder), mean, std))
            if "bilinear_att_train_imagenet_h200_g4_relu" in folder:
                a = 2
            info = get_info(path_utils.get_filename_without_extension(folder))
            # if to > 30:
            df = df.append(pd.DataFrame({'name':        [info[0]],
                                         'method':      [info[1]],
                                         'image':       [info[2]],
                                         'question':    [info[3]],
                                         'dim_h':       [info[4]],
                                         'nb_glimpses': [info[5]],
                                         'activation':  [info[6]],
                                         'epoch_max':   [to],
                                         'mean':        [mean.round(2)],
                                         'std':         [std.round(2)]
                                         }),
                           ignore_index=True)

        print(df)
        df.to_csv(df_path, index=False)


if __name__ == "__main__":
    main()
