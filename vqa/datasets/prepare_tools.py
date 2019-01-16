import os
import pandas as pd
from xml.dom import minidom

def get_project_dir(path, project_name):
    paths = path.split(project_name)
    return paths[0] + project_name

def get_data_xml(mydoc, tag):
    items = mydoc.getElementsByTagName(tag)
    value = items[0].firstChild.data
    return value

def get_object_xml(mydoc):
    items = mydoc.getElementsByTagName("object")
    boxes = list()
    for i in range(len(items)):
        item = items[i]
        boxes.append(get_bndbox_xml(item))
    return boxes

def get_bndbox_xml(item):
    object_name = get_data_xml(item, "name")
    object_box = item.getElementsByTagName("bndbox")[0]
    xmin = int(get_data_xml(object_box, "xmin"))
    ymin = int(get_data_xml(object_box, "ymin"))
    xmax = int(get_data_xml(object_box, "xmax"))
    ymax = int(get_data_xml(object_box, "ymax"))
    bndbox_list = [object_name, xmin, xmax, ymin, ymax]
    return bndbox_list


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
DATASETS_DIR = PROJECT_DIR + "/datasets/m2cai16-tool-locations/Annotations/"
filename = "v08_012825.xml"
path_xml = DATASETS_DIR + filename


def main():
    # parse an xml file by name
    mydoc = minidom.parse(path_xml)
    
    value = get_data_xml(mydoc, "filename")
    print(value)
    value = get_object_xml(mydoc)
    print(value)



if __name__ == "__main__":
    main()  