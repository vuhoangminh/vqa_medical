import json
import os
import pandas as pd
import argparse
from collections import Counter, OrderedDict
import pprint

pp = pprint.PrettyPrinter(indent=4)
# with open('data.txt', 'w') as outfile:
#     json.dump(data, outfile)

debug = 1

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
    'Is the microaneurysm larger than the soft exudate?'
]

dir_raw = '../../data/vqa_idrid/raw/raw/'
df_filename = dir_raw + 'idrid_questions_gt_split.csv'
fulldf_filename = dir_raw + 'idrid_questions_answer_full.csv'

dir_interim = '../../data/vqa_idrid/interim/'
train_filename = dir_interim + 'train_questions_annotations.json'
val_filename = dir_interim + 'val_questions_annotations.json'
test_filename = dir_interim + 'test_questions_annotations.json'
trainval_filename = dir_interim + 'trainval_questions_annotations.json'
test_filename = dir_interim + 'test_questions.json'
testdev_filename = dir_interim + 'testdev_questions.json'

def main():
    df = pd.read_csv(df_filename)
    if debug==2:
        print(df.head())

    df = create_image_id(df)        
    if debug==2:
        print(df.head())

    if os.path.exists(fulldf_filename):
        df = pd.read_csv(fulldf_filename)  
    else:          
        df = create_full_imageid_quesid_questype(df)

    if debug==2:
        print('\nhead')
        print(df.head())
        print('\ntail')
        print(df.tail())

    train_questions_annotations = create_questions_annotations(df, 'train')
    val_questions_annotations = create_questions_annotations(df, 'val')
    test_questions_annotations = create_questions_annotations(df, 'test')

    if os.path.exists(trainval_filename):
        print('>> loading', trainval_filename)
        with open(trainval_filename) as f:
            trainval_questions_annotations = json.load(f)
    else:
        trainval_questions_annotations = train_questions_annotations + val_questions_annotations
        print('>> saving', trainval_filename)
        with open(trainval_filename, 'w') as fp:
            json.dump(trainval_questions_annotations, fp) 

    test_questions = create_questions(df, 'test')
    testdev_questions = create_questions(df, 'testdev')


def create_questions(df, dataset):
    filename = dir_interim + dataset + '_questions.json'
    if dataset=='testdev':
        dataset='test'

    if os.path.exists(filename):
        print('>> loading', filename)
        with open(filename) as f:
            dataset_questions = json.load(f)

    else:
        dataset_questions = []
        for index, row in df.iterrows():
            if row['dataset'] == dataset:
                question_id = str(row['question_id']).zfill(9)
                image_name = row['file_id'] + '.jpg'
                question = row['question']
                
                row_dict = OrderedDict()
                row_dict['question_id'] = question_id
                row_dict['image_name'] = image_name
                row_dict['question'] = question
                dataset_questions = dataset_questions + [row_dict]

        json_data = dataset_questions

        with open(filename, 'w') as fp:
            print('>> saving', filename)
            json.dump(json_data, fp)       

    return dataset_questions


def create_questions_annotations(df, dataset):
    filename = dir_interim + dataset + '_questions_annotations.json'

    if os.path.exists(filename):
        print('>> loading', filename)
        with open(filename) as f:
            dataset_questions_annotations = json.load(f)

    else:
        dataset_questions_annotations = []
        for index, row in df.iterrows():
            if row['dataset'] == dataset:
                question_id = str(row['question_id']).zfill(9)
                image_name = row['file_id'] + '.jpg'
                question = row['question']
                answer = row['answer']
                answers_occurence = [[answer,10]]
                
                row_dict = OrderedDict()
                row_dict['question_id'] = question_id
                row_dict['image_name'] = image_name
                row_dict['question'] = question
                row_dict['answer'] = answer
                row_dict['answers_occurence'] = answers_occurence

                dataset_questions_annotations = dataset_questions_annotations + [row_dict]

        json_data = dataset_questions_annotations

        with open(filename, 'w') as fp:
            print('>> saving', filename)
            json.dump(json_data, fp)       

    return dataset_questions_annotations

def create_full_imageid_quesid_questype(df):
    cols = ['file_id', 'image_id', 'question', 'question_id', 
            'question_type', 'answer', 'multiple_choice_answer',
            'answer', 'answer_confidence', 'answer_id', 
            'dataset']
    full_df = pd.DataFrame(columns=cols)
    for index, row in df.iterrows():
        file_id = row['file_id']
        dataset = row['dataset']
        image_id = row['image_id']
        question_i = 0
        for question in question_list:
            question_id = str(image_id + str(question_i).zfill(3)).zfill(9)
            answer = row[question]
            temp = pd.DataFrame({'file_id': [file_id],
                                'image_id': [image_id],
                                'question': [question],
                                'question_id': [question_id],
                                'question_type': ['is the'],
                                'answer': [answer],
                                'multiple_choice_answer': [answer],
                                'answer_confidence': ['yes'],
                                'answer_id': [1],
                                'dataset': [dataset]
                                })
            if debug==2: print(temp)    
            if len(full_df)==0:
                full_df = temp
            else:                
                full_df = full_df.append(temp)                                        
            question_i = question_i + 1
    full_df.to_csv(fulldf_filename, index=False)         
    return full_df

def create_image_id(df):
    # df = df.drop(['Unnamed: 0'],axis=0)
    temp = pd.DataFrame(columns=['image_id', 'File_id'])
    temp['File_id'] = df['File_id'].values
    for index, _ in df.iterrows():
        image_id = str(index)
        image_id = image_id.zfill(6)
        if debug == 2:
            print (index)
            print (image_id)
        temp['image_id'][index] = image_id             
    df = pd.merge(df, temp, on='File_id')   
    df.rename(columns={'File_id': 'file_id'}, inplace=True) 
    df = df.drop(columns = ['Unnamed: 0'])             
    return df

if __name__ == "__main__":
    main()