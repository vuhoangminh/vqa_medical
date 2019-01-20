import os
import glob
import pandas as pd
from collections import OrderedDict
from scipy.ndimage import imread
import datasets.utils.breast_qa_utils as qa_utils
import datasets.utils.paths_utils as path_utils
import datasets.utils.xml_utils as xml_utils
import datasets.utils.image_utils as image_utils


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
PREPROCESSED_IMAGE_WSI_DIR = PROJECT_DIR + \
    "/data/raw/breast-cancer/preprocessed/WSI/"
PREPROCESSED_IMAGE_WSI_PATCH_DIR = PREPROCESSED_IMAGE_WSI_DIR + "patch/"
PREPROCESSED_IMAGE_WSI_GT_DIR = PREPROCESSED_IMAGE_WSI_DIR + "patch_gt/"
RAW_DIR = PROJECT_DIR + "/data/vqa_breast/raw/raw/"
path_utils.make_dir(RAW_DIR)


DICT_CLASS = {
    0: "normal",
    1: "benign",
    2: "in situ",
    3: "invasive"
}


list_class = [
    "normal",
    "benign",
    "in situ",
    "invasive"
]

list_class = [x.lower() for x in list_class]
list_classes = list_class, list_class


def add_case(path_case, image_id):
    gt = imread(path_case)
    gt = image_utils.convert_rgb_to_gray(gt)

    file_id = path_utils.get_filename(path_case)

    cols = ['file_id', 'image_id']
    rows = [file_id, image_id]

    q_row = list()
    a_row = list()


    # which patient?
    q_row.append(qa_utils.generate_ques_which_patient())
    a_row.append(qa_utils.get_ans_which_patient(path_case))

    # add how many classes are there?
    q_row.append(qa_utils.generate_ques_how_many_classes())
    a_row.append(qa_utils.get_ans_how_many_classes(gt))

    # add how many tumor classes are there?
    q_row.append(qa_utils.generate_ques_how_many_tumor_classes())
    a_row.append(qa_utils.get_ans_how_many_tumor_classes(gt))

    # add major class?
    q_row.append(qa_utils.generate_ques_major_class())
    a_row.append(qa_utils.get_ans_major_class(gt, DICT_CLASS))

    # add minor class?
    q_row.append(qa_utils.generate_ques_minor_class())
    a_row.append(qa_utils.get_ans_minor_class(gt, DICT_CLASS))

    # add major class?
    q_row.append(qa_utils.generate_ques_major_tumor())
    a_row.append(qa_utils.get_ans_major_tumor(gt, DICT_CLASS))

    # add minor class?
    q_row.append(qa_utils.generate_ques_minor_tumor())
    a_row.append(qa_utils.get_ans_minor_tumor(gt, DICT_CLASS))    

    # is there any x in?
    q_row.extend(qa_utils.generate_ques_is_there_any_x(list_class))
    a_row.extend(qa_utils.get_ans_is_there_any_x(list_class, gt, DICT_CLASS))

    # is x larger/smaller than y?
    q, combinations = qa_utils.generate_ques_is_x_larger_or_smaller_than_y(list_classes)
    a = qa_utils.get_ans_is_x_larger_or_smaller_than_y(combinations, gt, DICT_CLASS)
    q_row.extend(q)
    a_row.extend(a)

    # is x in z?
    q, encoded_locations = qa_utils.generate_ques_is_x_in_z(list_class, (256, 256))
    a = qa_utils.get_ans_is_x_in_z(list_class, encoded_locations, gt, DICT_CLASS)
    q_row.extend(q)
    a_row.extend(a)

    # extend cols rows
    cols.extend(q_row)
    rows.extend(a_row)

    return cols, rows


def main():
    paths = glob.glob(PREPROCESSED_IMAGE_WSI_PATCH_DIR + "*.png")
    rows = list()
    rows_temp = list()
    for index, patch_path in enumerate(paths):
        # if index<100:
        print(">> processing {}/{}".format(index+1, len(paths)))
        col, row = add_case(patch_path, str(index).zfill(5))
        rows_temp.append(row)
        if index % 20 == 0 and index > 0:
            rows.extend(rows_temp)
            rows_temp = list()
    if len(rows_temp) > 0:
        rows.extend(rows_temp)

    rows = list(map(list, zip(*rows)))
    # dictionary = dict(zip(col, rows))
    dictionary = OrderedDict(zip(col, rows))

    df = pd.DataFrame.from_dict(dictionary)

    save_dir = RAW_DIR + "breast_qa_full.csv"
    df.to_csv(save_dir)
    print(df)


if __name__ == "__main__":
    main()
