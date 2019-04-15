import glob
import os
import shutil
import datasets.utils.paths_utils as paths_utils


LIST_QUESTION_BREAST = [
    "how many classes are there",
    "is there any benign class in the image",
    "is there any in situ class in the image",
    "is there any invasive class in the image",
    "what is the major class in the image",
    "what is the minor class in the image",
    "is benign in 64_64_32_32 location",
    "is invasive in 96_96_32_32 location",
]

LIST_QUESTION_TOOLS = [
    "how many tools are there",
    "is scissors in 64_32_32_32 location",
    "is irrigator in 64_96_32_32 location",
    "is grasper in 64_96_32_32 location"
    "is bipolar in 64_96_32_32 location"
    "is hook in 64_96_32_32 location"
    "is clipper in 64_96_32_32 location"
    "is specimenbag in 64_96_32_32 location"
    "is there any grasper in the image",
    "is there any bipolar in the image",
    "is there any hook in the image",
    "is there any scissors in the image",
    "is there any clipper in the image",
    "is there any irrigator in the image",
    "is there any specimenbag in the image",
]

LIST_QUESTION_IDRID = [
    "is there haemorrhages in the fundus",
    "is there microaneurysms in the fundus",
    "is there soft exudates in the fundus",
    "is there hard exudates in the fundus",
    "is hard exudates larger than soft exudates",
    "is haemorrhages smaller than microaneurysms",
    "is there haemorrhages in the region 32_32_16_16",
    "is there microaneurysms in the region 96_96_16_16",
]


BREAST = {
    "A02_idx-31344-39648_ps-8192-8192": "how many classes are there",
    "A03_idx-23568-14336_ps-4096-4096": "is invasive in 96_96_32_32 location",
    "A04_idx-22904-32351_ps-4096-4096": "what is the major class in the image",
    "A05_idx-35648-3248_ps-4096-4096": "is there any benign class in the image",
    "A10_idx-20000-2576_ps-4096-4096": "what is the minor class in the image",
    "A08_idx-34864-6672_ps-4096-4096": "is there any invasive class in the image",
}

TOOLS = {
    "v04_020125": "how many tools are there",
    "v04_020400": "is there any grasper in the image",
    "v05_056350": "how many tools are there",
    "v04_020200": "is there any specimenbag in the image",
    "v04_020300": "is there any irrigator in the image",
    "v06_070725": "is there any bipolar in the image",
}

IDRID = {
    "IDRiD_44": "is there hard exudates in the fundus",
    "IDRiD_46": "is there microaneurysms in the fundus",
    "IDRiD_47": "is there hard exudates in the fundus",
    "IDRiD_47": "is there microaneurysms in the region 96_96_16_16",
    "IDRiD_51": "is there haemorrhages in the region 32_32_16_16",
    "IDRiD_52": "is there microaneurysms in the region 96_96_16_16",
    "IDRiD_53": "is there haemorrhages in the fundus",
    "IDRiD_54": "is there haemorrhages in the fundus",
    "IDRiD_49": "is there microaneurysms in the region 96_96_16_16",
}


VQA = {
    "img1_what_color_is_the_hydrant_red": "what color is the hydrant",
    "img2_what_color_is_the_hydrant_black_and_yellow": "what color is the hydrant",
    "img3_why_are_the_men_jumping_to_catch_frisbee": "why are the men jumping to catch",
    "img4_why_are_the_men_jumping_trick": "why are the men jumping to catch",
    "img5_is_the_water_still_no": "is the water still",
    "img6_is_the_water_still_yes": "is the water still",
    "img7_how_many_people_are_in_the_image_4": "how many people are in the image",
    "img8_how_many_people_are_in_the_image_1": "how many people are in the image"
}


def find_in_dir(name, src_dir):
    paths = glob.glob(os.path.join(src_dir, "gradcam", "*", "*.jpg")) + \
        glob.glob(os.path.join(src_dir, "occlusion", "*", "*.jpg"))
    for path in paths:
        if name in path:
            print(name, "found")
            return path


def main(dataset="breast"):
    if dataset == "breast":
        dataset_dict = BREAST
    elif dataset == "idrid":
        dataset_dict = IDRID
    elif dataset == "tools":
        dataset_dict = TOOLS
    elif dataset == "vqa":
        dataset_dict = VQA

    src_dir = "temp"
    dst_dir = "temp/sup"
    paths_utils.make_dir(dst_dir)
    wrt_dir = "figures/sup"

    str = []
    for key, value in dataset_dict.items():
        img_name = key
        question_str = value
        question_str = question_str.replace(' ', '-')
        noatt_name = "{}_noatt_question_{}.jpg".format(img_name,
                                                       question_str)
        att_name = "{}_att_question_{}.jpg".format(img_name,
                                                   question_str)
        cnn_name = "{}_cnn.jpg".format(img_name)
        in_name = "{}_in.jpg".format(img_name)
        occ_name = "{}_{}_w_{:0}_s_{:0}_color.jpg".format(
            img_name, value.replace(' ', '_'), 32, 2)

        str.append("Question: {} - Answer: {}\n\n".format(value, " "))

        for name in [in_name, cnn_name, att_name, noatt_name, occ_name]:
            path = find_in_dir(name, src_dir)
            try:
                shutil.copy(path, os.path.join(dst_dir, name))
                str.append(
                    "\\includegraphics[width=\\subfigsize]{figures/sup/" + name + "}\n")
            except:
                str.append(
                    "\\includegraphics[width=\\subfigsize]{figures/sup/" + in_name + "}\n")
        str.append("\n")

    file1 = open("temp/myfile.txt", "a+")
    file1.writelines(str)
    file1.close()


if __name__ == '__main__':
    # dataset = "idrid"
    # main(dataset)
    # file1 = open("temp/myfile.txt", "a+")
    # file1.writelines(["\n\n\n"])
    # file1.close()
    # dataset = "tools"
    # main(dataset)
    # file1 = open("temp/myfile.txt", "a+")
    # file1.writelines(["\n\n\n"])
    # file1.close()    
    # dataset = "breast"
    # main(dataset)
    # file1 = open("temp/myfile.txt", "a+")
    # file1.writelines(["\n\n\n"])
    # file1.close()    
    dataset = "vqa"
    main(dataset)