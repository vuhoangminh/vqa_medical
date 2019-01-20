import os
import glob
import pandas as pd
from collections import OrderedDict
from xml.dom import minidom
import datasets.utils.breast_qa_utils as qa_utils
import datasets.utils.paths_utils as path_utils
import datasets.utils.xml_utils as xml_utils


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
PREPROCESSED_IMAGE_WSI_DIR = PROJECT_DIR + \
    "/data/raw/breast-cancer/preprocessed/WSI/"
PREPROCESSED_IMAGE_WSI_PATCH_DIR = PREPROCESSED_IMAGE_WSI_DIR + "patch/"
PREPROCESSED_IMAGE_WSI_GT_DIR = PREPROCESSED_IMAGE_WSI_DIR + "patch_gt/"


list_class = [
    "benign",
    "in situ",
    "invasive"
]
list_class = [x.lower() for x in list_class]
list_classes = list_class, list_class


def add_case(path_case, image_id):
    mydoc = minidom.parse(path_case)
    file_id = xml_utils.get_data_xml(mydoc, "filename")
    boxes = xml_utils.get_object_xml(mydoc)

    cols = ['file_id', 'image_id']
    rows = [file_id, image_id]

    q_row = list()
    a_row = list()
    # add how many tools are there?
    q_row.append(qa_utils.generate_ques_how_many_tools())
    a_row.append(qa_utils.get_ans_how_many_tools(boxes))

    # what is the key tool?
    q_row.append(qa_utils.generate_ques_major_tool())
    a_row.append(qa_utils.get_ans_major_tool(boxes))

    # what is the minor tool?
    q_row.append(qa_utils.generate_ques_minor_tool())
    a_row.append(qa_utils.get_ans_minor_tool(boxes))

    # is x larger/smaller than y?
    q, combinations = qa_utils.generate_ques_is_x_larger_or_smaller_than_y(
        list_classes, data="bounding box")
    a = qa_utils.get_ans_is_x_larger_or_smaller_than_y(combinations, boxes)
    q_row.extend(q)
    a_row.extend(a)

    # is x larger/smaller than y?
    q, encoded_locations = qa_utils.generate_ques_is_x_in_z(
        list_class, (596, 334))
    a = qa_utils.get_ans_is_x_in_z(list_class, encoded_locations, boxes)
    q_row.extend(q)
    a_row.extend(a)

    # extend cols rows
    cols.extend(q_row)
    rows.extend(a_row)
    # rows = [[i] for i in rows]

    # dictionary = dict(zip(cols, rows))
    # dictionary = OrderedDict(zip(cols, rows))

    # df = pd.DataFrame.from_dict(dictionary)
    return cols, rows


def main():
    paths = glob.glob(PREPROCESSED_IMAGE_WSI_PATCH_DIR + "*.jpg")
    rows = list()
    rows_temp = list()
    for index, patch_path in enumerate(paths):
        # if index<100:
        print(">> processing {}/{}".format(index+1, len(paths)))
        col, row = add_case(patch_path, str(index).zfill(5))
    #     rows_temp.append(row)
    #     if index % 20 == 0 and index > 0:
    #         rows.extend(rows_temp)
    #         rows_temp = list()
    # if len(rows_temp) > 0:
    #     rows.extend(rows_temp)

    # rows = list(map(list, zip(*rows)))
    # # dictionary = dict(zip(col, rows))
    # dictionary = OrderedDict(zip(col, rows))

    # df = pd.DataFrame.from_dict(dictionary)

    # save_dir = RAW_DIR + "tools_qa_full.csv"
    # df.to_csv(save_dir)
    # print(df)


if __name__ == "__main__":
    main()
