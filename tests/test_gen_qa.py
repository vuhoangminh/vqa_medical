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


# print(generate_ques_how_many_tools())

# print(generate_ques_is_there_any_x(list1))

# print(find_all_combination_of_two_lists(list2))

# print(generate_ques_is_x_larger_or_smaller_than_y(list2, data="bounding box"))

# print(generate_ques_is_x_larger_or_smaller_than_y(list2))

box1 = [0,10,0,10]
box2 = [11,12,5,6]
print(is_two_boxes_overlap(box1, box2))


print(generate_encoded_box_location(16, (596,334)))


print(decode_encoded_location("0_0_16_16"))