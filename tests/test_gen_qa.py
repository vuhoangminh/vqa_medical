from datasets.utils.generate_qa import *

list1 = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag"
]

list2 = list1, list1


print(generate_ques_how_many_tools())

print(generate_ques_is_there_any_x(list1))

print(find_all_combination_of_two_lists(list2))
