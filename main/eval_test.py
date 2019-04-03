import json, os
from pprint import pprint
import numpy as np

LIST_DATASET = [
    "breast",
    "idrid",
    "tools",
    "vqa_v1",
    "vqa_v2"
]

LIST_METHOD = [
    "minhmul_att_train",
    "minhmul_noatt_train",
    "mlb_att_train",
    "mlb_noatt_train",
    "mutan_att_train",
    "mutan_noatt_train",
    "minhmul_noatt_train_2048",
    "minhmul_att_train_2048",
]

import datasets.utils.paths_utils as path_utils

CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
LOGS_DIR = PROJECT_DIR + "/logs/"


def get_val_acc1_from_json(json_path, from_pt=1, to_pt=69):
    with open(json_path) as f:
        data = json.load(f)
    val = data["logged"]["val"]["acc1"]
    # pprint(val)
    list_acc1 = []
    for i in range(from_pt, to_pt+1):
        acc1 = val["{}".format(str(i))]
        list_acc1.append(acc1)
    return list_acc1


def process_one_method_one_dataset(method, dataset):
    json_path = "{}{}/{}/logger.json".format(LOGS_DIR, dataset, method)
    if os.path.exists(json_path):
        list_acc1 = get_val_acc1_from_json(json_path, from_pt=1, to_pt=67)
        list_last_40 = list_acc1[len(list_acc1)-39:len(list_acc1)]
        mean = np.mean(list_last_40)
        std = np.std(list_last_40)
    else:
        mean, std = 0, 0

    return mean, std


def main():
    for dataset in LIST_DATASET:
        for method in LIST_METHOD:
            print("\n{}, {}".format(dataset, method))
            mean, std = process_one_method_one_dataset(method, dataset)
            print("mean={:.2f}({:.2f})".format(mean, std))


if __name__ == "__main__":
    main()