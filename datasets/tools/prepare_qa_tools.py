import os
import glob
import pandas as pd
from collections import OrderedDict
from xml.dom import minidom
from datasets.utils.images import is_image_file
import datasets.utils.tools_qa_utils as qa_utils
import datasets.utils.paths_utils as path_utils
import datasets.utils.xml_utils as xml_utils


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
DATASETS_DIR = PROJECT_DIR + "/data/raw/m2cai16-tool-locations/Annotations/"
RAW_DIR = PROJECT_DIR + "/data/vqa_tools/raw/raw/"
path_utils.make_dir(RAW_DIR)


list_tool = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag"
]
list_tool = [x.lower() for x in list_tool]
list_tools = list_tool, list_tool


def fix_bb_coordinates_after_resize(boxes, old_shape, new_shape):
    new_boxes = list()
    for box in boxes:
        object_name, xmin, xmax, ymin, ymax = box
        Rx, Ry = new_shape[0]/old_shape[0], new_shape[1]/old_shape[1]
        new_xmin, new_xmax, new_ymin, new_ymax = round(xmin*Rx), round(xmax*Rx), round(ymin*Ry), round(ymax*Ry)
        new_boxes.append([object_name, new_xmin, new_xmax, new_ymin, new_ymax])
    return new_boxes



def add_case(path_case, image_id):
    mydoc = minidom.parse(path_case)
    file_id = xml_utils.get_data_xml(mydoc, "filename")
    boxes = xml_utils.get_object_xml(mydoc)

    width, height = xml_utils.get_size_image_xml(mydoc)

    boxes = fix_bb_coordinates_after_resize(boxes, (width, height), (256, 256))


    cols = ['file_id', 'image_id']
    rows = [file_id, image_id]

    q_row = list()
    a_row = list()

    # add how many tools are there?
    q_row.append(qa_utils.generate_ques_how_many_tools())
    a_row.append(qa_utils.get_ans_how_many_tools(boxes))

    # is there any x in?
    q_row.extend(qa_utils.generate_ques_is_there_any_x(list_tool))
    a_row.extend(qa_utils.get_ans_is_there_any_x(list_tool, boxes))

    """
    removed by R and T's suggestions     
    # what is the key tool?
    q_row.append(qa_utils.generate_ques_major_tool())
    a_row.append(qa_utils.get_ans_major_tool(boxes))

    # what is the minor tool?
    q_row.append(qa_utils.generate_ques_minor_tool())
    a_row.append(qa_utils.get_ans_minor_tool(boxes))

    # is x larger/smaller than y?
    q, combinations = qa_utils.generate_ques_is_x_larger_or_smaller_than_y(
        list_tools, data="bounding box")
    a = qa_utils.get_ans_is_x_larger_or_smaller_than_y(combinations, boxes)
    q_row.extend(q)
    a_row.extend(a) 
    """

    # is x in z?
    q, encoded_locations = qa_utils.generate_ques_is_x_in_z(
        list_tool, (256, 256))
    a = qa_utils.get_ans_is_x_in_z(list_tool, encoded_locations, boxes)
    q_row.extend(q)
    a_row.extend(a)

    # which tool having pointed tip position?
    q = qa_utils.generate_ques_which_tool_pointed_tip_position()
    a = qa_utils.get_ans_which_tool_pointed_tip_position(boxes)
    q_row.extend(q)
    a_row.extend(a)


    # extend cols rows
    cols.extend(q_row)
    rows.extend(a_row)
    
    return cols, rows


def main():
    paths = glob.glob(DATASETS_DIR + "*.xml")
    rows = list()
    rows_temp = list()
    for index, path in enumerate(paths):
        # if index<100:
        print(">> processing {}/{}".format(index+1, len(paths)))
        col, row = add_case(path, str(index).zfill(5))
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

    save_dir = RAW_DIR + "tools_qa_full.csv"
    df.to_csv(save_dir)
    print(df)


def test():
    paths = glob.glob(DATASETS_DIR + "*.xml")
    rows = list()
    rows_temp = list()
    count_dict_tool = dict(zip(list_tool, [0] * 7))
    count_dict_tool_multi = dict(zip(list_tool, [0] * 7))
    count_multitool_per_frame = 0
    for index, path in enumerate(paths):
        # if index<100:
        print(">> processing {}/{}".format(index+1, len(paths)))
        mydoc = minidom.parse(path)
        boxes = xml_utils.get_object_xml(mydoc)
        for i in range(len(boxes)):
            tool, xmin, xmax, ymin, ymax = boxes[i]
            tool = tool.lower()
            count_dict_tool[tool] += 1   
        if len(boxes)>1:
            count_multitool_per_frame += 1
        else:
            count_dict_tool_multi[tool] += 1
    print(count_dict_tool)
    print(count_multitool_per_frame)
    print(count_dict_tool_multi)

if __name__ == "__main__":
    # main()
    test()
