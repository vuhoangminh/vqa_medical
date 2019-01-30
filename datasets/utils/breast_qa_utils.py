import itertools
from itertools import product
import numpy as np


DICT_CLASS = {
    0: "normal",
    1: "benign",
    2: "in situ",
    3: "invasive"
}


def get_count_from_class_number(number, unique_elements, counts_elements):
    index = np.where(unique_elements == number)
    if index[0].size == 0:
        return 0
    else:
        return counts_elements[index]


def get_name_class_from_number(number, dict_class):
    return dict_class[number]


'''
Get a list of keys from dictionary which has value that matches with any value in given list of values
'''


def get_number_from_name_class(values, dict_class):
    keys = list()
    items = dict_class.items()
    for item in items:
        if item[1] in values:
            keys.append(item[0])
    return keys


def find_all_combination_of_two_lists(list1, is_unique=True):
    if is_unique:
        return [(x, y) for x, y in product(*list1) if x != y]
    else:
        return [(x, y) for x, y in product(*list1)]


def generate_ques_how_many_classes():
    return "how many classes are there?"


def get_ans_how_many_classes(gt):
    unique_classes = np.unique(gt)
    return len(unique_classes)


def generate_ques_how_many_tumor_classes():
    return "how many tumor classes are there?"


def get_ans_how_many_tumor_classes(gt):
    unique_classes = np.unique(gt)
    unique_classes_without_normal = [x for x in unique_classes if x > 0]
    return len(unique_classes_without_normal)


def generate_ques_major_class():
    return "what is the major class in the image?"


def get_ans_major_class(gt, dict_class):
    unique_elements, counts_elements = np.unique(gt, return_counts=True)
    return get_name_class_from_number(unique_elements[np.argmax(counts_elements)], dict_class)


def generate_ques_minor_class():
    return "what is the minor class in the image?"


def get_ans_minor_class(gt, dict_class):
    unique_elements, counts_elements = np.unique(gt, return_counts=True)
    if unique_elements.size == 1:
        return "na"
    else:
        return get_name_class_from_number(unique_elements[np.argmin(counts_elements)], dict_class)


def generate_ques_major_tumor():
    return "what is the major tumor in the image?"


def get_ans_major_tumor(gt, dict_class):
    unique_elements, counts_elements = np.unique(gt, return_counts=True)
    index_normal = np.where(unique_elements == 0)
    unique_elements = np.delete(unique_elements, index_normal)
    counts_elements = np.delete(counts_elements, index_normal)
    if unique_elements.size >= 1:
        return get_name_class_from_number(unique_elements[np.argmax(counts_elements)], dict_class)
    else:
        return "na"


def generate_ques_minor_tumor():
    return "what is the minor class in the image?"


def get_ans_minor_tumor(gt, dict_class):
    unique_elements, counts_elements = np.unique(gt, return_counts=True)
    index_normal = np.where(unique_elements == 0)
    unique_elements = np.delete(unique_elements, index_normal)
    counts_elements = np.delete(counts_elements, index_normal)
    if unique_elements.size > 1:
        return get_name_class_from_number(unique_elements[np.argmin(counts_elements)], dict_class)
    else:
        return "na"


def generate_ques_is_x_larger_or_smaller_than_y(x, data=""):
    combinations = find_all_combination_of_two_lists(x)
    q = list()
    for i in range(len(combinations)):
        t1 = combinations[i][0]
        t2 = combinations[i][1]
        if data == "":
            ques = "is {} larger than {}".format(t1, t2)
            q.append(ques.lower())
            ques = "is {} smaller than {}".format(t1, t2)
            q.append(ques.lower())
        else:
            ques = "is {}'s {} larger than {}'s".format(t1, data, t2)
            q.append(ques.lower())
            ques = "is {}'s {} smaller than {}'s".format(t1, data, t2)
            q.append(ques.lower())
    return q, combinations


def get_one_ans_is_x_larger_than_y(combination, gt, dict_class):
    unique_elements, counts_elements = np.unique(gt, return_counts=True)
    t1 = combination[0]
    t2 = combination[1]
    t1 = get_number_from_name_class(t1, dict_class)
    t2 = get_number_from_name_class(t2, dict_class)

    n1 = get_count_from_class_number(t1, unique_elements, counts_elements)
    n2 = get_count_from_class_number(t2, unique_elements, counts_elements)

    if n1 > n2:
        return "yes"
    elif n2 > n1:
        return "no"
    elif n1 == n2 and n1 > 0:
        return "no"
    else:
        return "na"


def get_ans_is_x_larger_or_smaller_than_y(combinations, gt, dict_class):
    a = list()
    for i in range(len(combinations)):
        t1 = combinations[i][0]
        t2 = combinations[i][1]
        ans_larger = get_one_ans_is_x_larger_than_y([t1, t2], gt, dict_class)
        ans_smaller = get_one_ans_is_x_larger_than_y([t2, t1], gt, dict_class)
        a.append(ans_larger)
        a.append(ans_smaller)
    return a


def generate_ques_is_there_any_x(x):
    q = list()
    for i in range(len(x)):
        ques = "is there any {} class in the image?".format(x[i])
        q.append(ques.lower())
    return q


def get_ans_is_there_any_x(x, gt, dict_class):
    a = list()
    unique_elements, counts_elements = np.unique(gt, return_counts=True)
    for i in range(len(x)):
        t = get_number_from_name_class(x[i], dict_class)
        n = get_count_from_class_number(t, unique_elements, counts_elements)

        if n > 0:
            a.append("yes")
        else:
            a.append("no")

    return a


def generate_encoded_box_location(image_shape, box_size=32):
    h, w = image_shape
    _h = list(range(0, h, box_size))
    _w = list(range(0, w, box_size))
    list1 = _h, _w
    combinations = find_all_combination_of_two_lists(list1, is_unique=False)
    encoded_locations = list()
    for i in range(len(combinations)):
        x, y = combinations[i]
        ques = str("{}_{}_{}_{}".format(x, y, box_size, box_size))
        encoded_locations.append(ques)
    return encoded_locations


def decode_encoded_location(encoded_location):
    split = encoded_location.split('_')
    split = [int(x) for x in split]
    xmin1, ymin1, box_size, box_size = split
    return [xmin1, xmin1+box_size, ymin1, ymin1+box_size]


def generate_ques_is_x_in_z(x, image_shape):
    encoded_locations = generate_encoded_box_location(image_shape, box_size=32)
    q = list()
    for i in range(len(encoded_locations)):
        for j in range(len(x)):
            ques = "is {} in {} location?".format(x[j], encoded_locations[i])
            q.append(ques.lower())
    return q, encoded_locations


def get_ans_is_x_in_z(x, encoded_locations, gt, dict_class):
    a = list()
    for i in range(len(encoded_locations)):
        for j in range(len(x)):
            t = get_number_from_name_class(x[j], dict_class)
            decoded_location = decode_encoded_location(encoded_locations[i])
            xmin1, xmax1, ymin1, ymax1 = decoded_location
            gt_extracted = gt[xmin1:xmax1, ymin1:ymax1]
            index = np.where(gt_extracted == t)
            if index[0].size == 0:
                a.append("no")
            else:
                a.append("yes")
    return a


def generate_ques_which_patient():
    return "which patient is it?"


def get_ans_which_patient(filename):
    for i in range(10):
        patient = "A{}".format(str(i+1).zfill(2))
        if patient in filename:
            return patient


def generate_ques_how_many_pixels_of_x(x):
    q = list()
    for i in range(len(x)):
        ques = "how many percent of {} class in the image?".format(x[i])
        q.append(ques.lower())
    return q


def get_ans_how_many_pixels_of_x(x, gt, dict_class):
    a = list()
    unique_elements, counts_elements = np.unique(gt, return_counts=True)
    for i in range(len(x)):
        t = get_number_from_name_class(x[i], dict_class)
        n = get_count_from_class_number(t, unique_elements, counts_elements)
        if type(n) is np.ndarray:
            a.append(n[0])
        else:
            a.append(n)
    return a
