import os, glob
import pandas as pd
import pprint
from sklearn.model_selection import train_test_split
from shutil import copyfile
import shutil
import itertools
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

debug = 1
pp = pprint.PrettyPrinter(indent=4)

question_list = [
    'Is there retinal hemorrhage in the fundus?', 
    'Is there hard exudate in the fundus?', 
    'Is there microaneurysm in the fundus?', 
    'Is there soft exudate in the fundus?', 
    'Is the retinal hemorrhage larger than the hard exudate?', 
    'Is the retinal hemorrhage larger than the microaneurysm?', 
    'Is the retinal hemorrhage larger than the soft exudate?', 
    'Is the hard exudate larger than the microaneurysm?', 
    'Is the hard exudate larger than the soft exudate?', 
    'Is the microaneurysm larger than the soft exudate?',
    'Is the retinal hemorrhage smaller than the hard exudate?', 
    'Is the retinal hemorrhage smaller than the microaneurysm?', 
    'Is the retinal hemorrhage smaller than the soft exudate?', 
    'Is the hard exudate smaller than the microaneurysm?', 
    'Is the hard exudate smaller than the soft exudate?', 
    'Is the microaneurysm smaller than the soft exudate?'
]

disease_list = [
    'haemorrhages', 'hard exudates', 'soft exudates', 'microaneurysms'
]

columns = question_list + disease_list + ['file_id']

key_dict = {
    'haemorrhages': '_HE', 
    'hard exudates': '_EX', 
    'soft exudates': '_SE', 
    'microaneurysms': '_MA'
}

pair_list = list(itertools.combinations(disease_list, 2))

debug = 1

def main():
    generate_basic_qa_all_plus_nar()
    

def generate_basic_qa_all_plus_nar():
    df = pd.read_csv('idrid_qa_gt_generated.csv')

    columns = list(df.columns)
    print (columns)

    new_columns = columns[-5:] + columns[:-5]
    print (new_columns)

    for i in range(90):
        img_dict = {}
        file_id = 'IDRiD_NAR_' + str(i).zfill(3)
        print(file_id) 
        img_dict['file_id'] = file_id 
        for disease in disease_list:
            img_dict[disease] = 0
            question_is_there = get_question_is_there(disease)
            img_dict[question_is_there] = 'no'
            temp = pd.DataFrame([img_dict]) 

        if len(df)==0:
            df = temp
        else:
            df = df.append(temp)   

        if debug: 
            print(img_dict)
            print(temp)                

    df = df.fillna('undefined')

    df = df[new_columns]
    print(df.tail())    

    df.to_csv('idrid_qa_gt.csv', index=False)              




def generate_basic_qa_all():
    df_train = generate_basic_qa('train') 
    df_test = generate_basic_qa('test')

    df_train = df_train.append(df_test)

    print(df_train.head())                    
    print(df_train.tail()) 

    df_train.to_csv('idrid_qa_gt_generated.csv', index=False) 


def generate_basic_qa(dataset):
    dir_folder_original_images = 'original_images/'    
    dir_folder_groundtruth_images = 'groundtruth/' 

    df = pd.DataFrame()
    
    for file in os.listdir(dir_folder_original_images + dataset + '/'):
        if file.endswith(".jpg"):
            img_dict = {}

            name_image, _ = os.path.splitext(file)
            img_dict['file_id'] = name_image 

            for disease in disease_list:
                num_pixel = count_pixel_hemorrhage(dir_folder_groundtruth_images, 
                        dataset, disease, name_image)
                img_dict[disease] = num_pixel                        

                question_is_there = get_question_is_there(disease)
                answer_is_there = get_answer_is_there(disease, num_pixel)
                if debug==2: print(question_is_there, answer_is_there)

                img_dict[question_is_there] = answer_is_there
 
            for pair in pair_list:
                if debug==2:
                    print(pair)
                    print(pair[0], pair[1])
                q1 = 'Is the {} larger than the {}?'.format(pair[0], pair[1])
                q2 = 'Is the {} smaller than the {}?'.format(pair[0], pair[1])
                q3 = 'Is the {} larger than the {}?'.format(pair[1], pair[0])
                q4 = 'Is the {} smaller than the {}?'.format(pair[1], pair[0])
                a1 = get_answer_is_the(pair[0], pair[1], img_dict, 1)
                a2 = get_answer_is_the(pair[0], pair[1], img_dict, 0)
                a3 = get_answer_is_the(pair[1], pair[0], img_dict, 1)
                a4 = get_answer_is_the(pair[1], pair[0], img_dict, 0)
                if debug==2: 
                    print(q1, a1)
                    print(q2, a2)
                    print(q3, a3)
                    print(q4, a4)

                img_dict[q1] = a1
                img_dict[q2] = a2
                img_dict[q3] = a3
                img_dict[q4] = a4

            if debug==2:
                print('--------------------------------------------------------')
                for key, value in img_dict.items():
                    print(key, value)

            temp = pd.DataFrame([img_dict]) 

            if len(df)==0:
                df = temp
            else:
                df = df.append(temp)   

    print (df.head())    

    return df

def get_question_is_there(disease):
    return 'Is there {} in the fundus?'.format(disease)

def get_answer_is_there(disease, num_pixel):
    if num_pixel>0:
        return 'yes'
    else:
        return 'no'        

def get_answer_is_the(disease1, disease2, img_dict, is_larger):
    count1 = img_dict[disease1]
    count2 = img_dict[disease2]

    if is_larger and count1>=count2:
        return 'yes'
    elif is_larger and count1<count2:        
        return 'no'
    elif not is_larger and count1>=count2:
        return 'no'
    elif not is_larger and count1<count2:        
        return 'yes'

def get_dir_image(dir_folder_original_images, dataset, name_image):
    dir_image = '{}{}/{}.jpg'.format(dir_folder_original_images, dataset, name_image)
    return dir_image

def get_dir_image_disease(dir_folder_groundtruth_images, dataset, disease, name_image):
    dir_image = '{}{}/{}/{}{}.tif'.format(dir_folder_groundtruth_images, 
            dataset, disease, name_image, key_dict[disease])
    return dir_image    

def count_pixel_hemorrhage(dir_folder_groundtruth_images, dataset, disease, name_image):
    dir_image_disease = get_dir_image_disease(dir_folder_groundtruth_images, dataset, disease, name_image)

    if os.path.exists(dir_image_disease):
        img = cv2.imread(dir_image_disease, 1)

        if debug==2:
            plt.imshow(img)
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            plt.show()

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        num_nonzeros = cv2.countNonZero(gray_image)
        return num_nonzeros
    else:
        return 0        

 

def copy_to_dataset(from_dir, to_dir, img_name):
    if not os.path.exists(to_dir):
        print('make dir', to_dir)
        os.makedirs(to_dir) 

    from_dir_filename = from_dir + '/' + img_name + '.jpg'
    to_dir_filename = to_dir + '/' + img_name + '.jpg'
    print('copying', from_dir_filename, 'to', to_dir_filename)          
    shutil.copy2(from_dir_filename, to_dir_filename)   

def split_df(df):
    df_train, df_test= train_test_split(df, test_size=0.2, random_state=1988)
    df_train, df_val= train_test_split(df_train, test_size=0.1, random_state=1988)
    return df_train, df_test, df_val

def merge_df(df_yes, df_no, df_undefined, dataset):
    df = df_yes.append([df_no, df_undefined])
    df['dataset'] = dataset
    return df

def split_train_val_test_write_to_df():
    dir_csv = 'data/idrid/raw/idrid_questions_gt.csv'
    df = pd.read_csv(dir_csv)
    if debug==2:
        print('\nhead:')
        pp.pprint(df.head())
        print('\ninfo:')
        pp.pprint(df.info())

    for question in df:
        sub = df[question].value_counts()
        # print(question)
        print(sub)

    print('\n>> start split df based on <Is the microaneurysm larger than the soft exudate?>')        
    question = 'Is the microaneurysm larger than the soft exudate?'
    df_yes = df.loc[df[question] == 'yes']
    df_no = df.loc[df[question] == 'no']
    df_undefined = df.loc[df[question] == 'undefined']

    if debug==2:
        print('\ndf_yes:')
        print(df_yes.head())
        print('\ndf_no:')
        print(df_no.head())
        print('\ndf_undefined:')
        print(df_undefined.head())

    print('>> split yes')
    df_yes_train, df_yes_test, df_yes_val = split_df(df_yes) 
    if debug==2:
        print('\ndf_yes_train:')
        print(df_yes_train.info())
        print('\ndf_yes_test:')
        print(df_yes_test.info())
        print('\ndf_yes_val:')
        print(df_yes_val.info())

    print('>> split no')
    df_no_train, df_no_test, df_no_val = split_df(df_no) 
    if debug==2:
        print('\ndf_no_train:')
        print(df_no_train.info())
        print('\ndf_no_test:')
        print(df_no_test.info())
        print('\ndf_no_val:')
        print(df_no_val.info())

    print('>> split undefined')
    df_undefined_train, df_undefined_test, df_undefined_val = split_df(df_undefined) 
    if debug==2:
        print('\ndf_undefined_train:')
        print(df_undefined_train.info())
        print('\ndf_undefined_test:')
        print(df_undefined_test.info())
        print('\ndf_undefined_val:')
        print(df_undefined_val.info())

    df_train = merge_df(df_yes_train, df_no_train, df_undefined_train, 'train')
    df_val = merge_df(df_yes_val, df_no_val, df_undefined_val, 'val')
    df_test = merge_df(df_yes_test, df_no_test, df_undefined_test, 'test')

    if debug==2:
        print('\ndf_train:')
        print(df_train.info())
        print('\ndf_val:')
        print(df_val.info())
        print('\ndf_test:')
        print(df_test.info())    

    df = df_train.append([df_val, df_test])
    # print(df)

    if debug==2:
        for question in df:
            sub = df[question].value_counts()
            # print(question)
            print(sub)

    filename = 'data/idrid/raw/idrid_questions_gt_split.csv'
    if not os.path.exists(filename):
        df.to_csv(filename)

    return df

def test():
    pair_list = list(itertools.combinations(disease_list, 2))
    for pair in pair_list:
        print(pair)
        print(pair[0], pair[1])
        q1 = 'Is the {} larger than the {}?'.format(pair[0], pair[1])
        q2 = 'Is the {} smaller than the {}?'.format(pair[0], pair[1])
        q3 = 'Is the {} larger than the {}?'.format(pair[1], pair[0])
        q4 = 'Is the {} smaller than the {}?'.format(pair[1], pair[0])
        print(q1, q2, q3, q4)

if __name__ == '__main__':
    main()
    # test()
