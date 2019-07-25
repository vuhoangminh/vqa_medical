import json, os
from pprint import pprint
import numpy as np

LIST_DATASET = [
    # "breast",
    "idrid",
    # "tools",
    # "vqa",
    # "vqa2"
]

LIST_METHOD = [
    "mutan_noatt_train",
    "mlb_noatt_train",
    "mutan_att_train",
    "mlb_att_train",
    "minhmul_noatt_train",
    "minhmul_att_train",

    "minhmul_noatt_train_relu",
    "minhmul_noatt_train_selu",

    "minhmul_att_train_relu",
    "minhmul_att_train_selu",

    "minhmul_noatt_train_relu_h200_g8",
    "minhmul_noatt_train_selu_h200_g8",

    "minhmul_att_train_relu_h200_g8",
    "minhmul_att_train_selu_h200_g8",
    "minhmul_att_train_relu_h200_g4",
    "minhmul_att_train_selu_h200_g4",    

    # "minhmul_att_train_leakyrelu",
    # "minhmul_att_train_celu",
    
    # "minhmul_att_train_leakyrelu",
    # "minhmul_att_train_relu_wrong",
    # "minhmul_att_train_wrong",
    # "minhmul_att_train_selu_wrong",    
    # "minhmul_att_train_h600_g4_relu",

    # "minhmul_noatt_train_2048",
    # "minhmul_att_train_2048",
]

import datasets.utils.paths_utils as path_utils

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
LOGS_DIR = PROJECT_DIR + "/logs/"


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


def get_val_acc1_from_json_new(json_path, num=20):
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


def process_one_method_one_dataset(method, dataset):
    json_path = "{}{}/{}/logger.json".format(LOGS_DIR, dataset, method)
    try:
        list_acc1, to = get_val_acc1_from_json(json_path)
        mean = np.mean(list_acc1)
        std = np.std(list_acc1)
    except:
        mean, std, to = 0, 0, 0

    return mean, std, to


def main():
    for dataset in LIST_DATASET:
        for method in LIST_METHOD:
            print("\n{}, {}".format(dataset, method))
            mean, std, to = process_one_method_one_dataset(method, dataset)
            print("mean={:.2f}({:.2f}) \t {}".format(mean, std, to))


if __name__ == "__main__":
    main()