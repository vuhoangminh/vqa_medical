import itertools
from itertools import product


def find_all_combination_of_two_lists(list1):
    return [(x, y) for x, y in product(*list1) if x != y]


def generate_ques_how_many_tools():
    return "how many tools are there?"


def generate_ques_major_tool():
    return "what is the major tool in the image?"


def generate_ques_minor_tool():
    return "what is the minor tool in the image?"

# def generate_ques_is_x_larger_than_y(x,y):


def generate_ques_is_there_any_x(x):
    q = list()
    for i in range(len(x)):
        ques = "is there any {} in the image?".format(x[i])
        q.append(ques.lower())
    return q
