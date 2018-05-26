import json
import os
import pandas as pd
import argparse
from collections import Counter

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

filename = '../../data/vqa_idrid/raw/raw/idrid_questions_gt_split.csv'
filenamefull = '../../data/vqa_idrid/raw/raw/idrid_questions_answer_full.csv'

def main():

    df = pd.read_csv(filename)
    if debug==2:
        print(df.head())

    df = create_image_id(df)        
    if debug==2:
        print(df.head())

    if os.path.exists(filenamefull):
        df = pd.read_csv(filenamefull)  
    else:          
        df = create_full_imageid_quesid_questype(df)

    if debug:
        print('\nhead')
        print(df.head())
        print('\ntail')
        print(df.tail())



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
    full_df.to_csv(filenamefull, index=False)         
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

def create_questions(df, dataset):
    return 0

def create_annotations(df, dataset):    
    return 0

if __name__ == "__main__":
    main()      
