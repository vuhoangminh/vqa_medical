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

pp = pprint.PrettyPrinter(indent=4)

disease_list = [
    'haemorrhages', 'hard exudates', 'soft exudates', 'microaneurysms'
]

key_dict = {
    'haemorrhages': '_HE', 
    'hard exudates': '_EX', 
    'soft exudates': '_SE', 
    'microaneurysms': '_MA'
}

pair_list = list(itertools.combinations(disease_list, 2))

debug = 1

def main():
    process_groundtruth('train')
    

def process_groundtruth(dataset):
    dir_folder_original_images = 'original_images/'    
    dir_folder_groundtruth_images = 'groundtruth/' 

    for file in os.listdir(dir_folder_original_images + dataset + '/'):
        if file.endswith(".jpg"):
            name_image, _ = os.path.splitext(file)
            for disease in disease_list:
                process_one_image(name_image, dataset, disease)

def process_one_image(name_image, dataset, disease):
    dir_folder_original_images = 'original_images'    
    dir_folder_groundtruth_images = 'groundtruth'
    dir_folder_processed_images = 'processed'
    dir_folder_processed_groundtruth = 'processed_groundtruth'
    
    dir_image_original = '{}/{}/{}.jpg'.format(
                                dir_folder_original_images, 
                                dataset, 
                                name_image)
    dir_image_processed = '{}/{}.jpg'.format(
                                dir_folder_processed_images,  
                                name_image)                          
    dir_image_haemorrhages = '{}/{}/{}/{}{}.tif'.format(
                                dir_folder_groundtruth_images, 
                                dataset, 
                                disease,
                                name_image,
                                key_dict[disease])
    dir_gt_haemorrhages = '{}/{}/{}/{}{}.tif'.format(
                                dir_folder_processed_groundtruth, 
                                dataset, 
                                disease,
                                name_image,
                                key_dict[disease])                                    

    if os.path.exists(dir_image_haemorrhages):
        print('>> processing', dir_image_haemorrhages)
        image_original = cv2.imread(dir_image_original, 1)
        gray_image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        retval, thresh_gray_image_original = \
            cv2.threshold(gray_image_original, thresh=30, maxval=255, type=cv2.THRESH_BINARY)

        image_haemorrhages = cv2.imread(dir_image_haemorrhages, 1)
        gray_image_haemorrhages = cv2.cvtColor(image_haemorrhages, cv2.COLOR_BGR2GRAY)
        retval, thresh_gray_image_haemorrhages = \
            cv2.threshold(gray_image_haemorrhages, thresh=30, maxval=255, type=cv2.THRESH_BINARY)

        crop_image_original, crop_image_haemorrhages = crop_image_remove_vertical_border(
                thresh_gray_image_original, thresh_gray_image_haemorrhages)

        image_processed = cv2.imread(dir_image_processed, 1)


        padded_image_original, padded_image_haemorrhages, resized_image_original, resized_image_haemorrhages = \
                pad_resize(crop_image_original, crop_image_haemorrhages)

        cv2.imwrite(dir_gt_haemorrhages,resized_image_haemorrhages)

        if debug==2:    
            plt.figure()

            plt.subplot(5,3,1)
            plt.imshow(gray_image_original)
            # plt.xticks([]), plt.yticks([])

            plt.subplot(5,3,2)
            plt.imshow(gray_image_haemorrhages)
            # plt.xticks([]), plt.yticks([])

            plt.subplot(5,3,3)
            plt.imshow(image_processed)
            # plt.xticks([]), plt.yticks([])
            
            plt.subplot(5,3,4)
            plt.imshow(thresh_gray_image_original)
            # plt.xticks([]), plt.yticks([])

            plt.subplot(5,3,5)
            plt.imshow(thresh_gray_image_haemorrhages)
            # plt.xticks([]), plt.yticks([])

            plt.subplot(5,3,7)
            plt.imshow(crop_image_original)
            
            plt.subplot(5,3,8)
            plt.imshow(crop_image_haemorrhages)

            plt.subplot(5,3,10)
            plt.imshow(padded_image_original)
            
            plt.subplot(5,3,11)
            plt.imshow(padded_image_haemorrhages)

            plt.subplot(5,3,13)
            plt.imshow(resized_image_original)
            
            plt.subplot(5,3,14)
            plt.imshow(resized_image_haemorrhages)        

            plt.show()
    else:
        print(dir_image_haemorrhages, 'not existed')

def pad_resize(crop_image_original, crop_image_haemorrhages):
    desired_size = 256
    # print(crop_image_original.shape)
    w = crop_image_original.shape[1]
    h = crop_image_original.shape[0]
    delta_w = 0
    delta_h = w-h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    padded_image_original = cv2.copyMakeBorder(crop_image_original, 
            top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    padded_image_haemorrhages = cv2.copyMakeBorder(crop_image_haemorrhages, 
            top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    resized_image_original = cv2.resize(padded_image_original, (256, 256))
    resized_image_haemorrhages = cv2.resize(padded_image_haemorrhages, (256, 256))

    return padded_image_original, padded_image_haemorrhages, \
            resized_image_original, resized_image_haemorrhages


def crop_image_remove_vertical_border(
        thresh_gray_image_original, thresh_gray_image_haemorrhages):
    points = np.argwhere(thresh_gray_image_original==255) # find where the black pixels are
    points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
    # x, y, w, h = x-10, y-10, w+20, h+20 # make the box a little bigger
    crop_image_original = thresh_gray_image_original[y:y+h, x:x+w] # create a cropped region of the gray image
    crop_image_haemorrhages = thresh_gray_image_haemorrhages[y:y+h, x:x+w] # create a cropped region of the gray image
    return crop_image_original, crop_image_haemorrhages

def generate_basic_qa(dataset):
    dir_folder_original_images = 'original_images/'    
    dir_folder_groundtruth_images = 'groundtruth/' 

    df = pd.DataFrame()
    
    i=0
    for file in os.listdir(dir_folder_original_images + dataset + '/'):
        if file.endswith(".jpg"):
            img_dict = {}

            name_image, _ = os.path.splitext(file)
            img_dict['file_id'] = name_image 

            for disease in disease_list:
                gray_image = transform_groundtruth(dir_folder_groundtruth_images, 
                        dataset, disease, name_image)  

            i = i+1
            if debug: break                                            
    return df    

def get_dir_image(dir_folder_original_images, dataset, name_image):
    dir_image = '{}{}/{}.jpg'.format(dir_folder_original_images, dataset, name_image)
    return dir_image

def get_dir_image_disease(dir_folder_groundtruth_images, dataset, disease, name_image):
    dir_image = '{}{}/{}/{}{}.tif'.format(dir_folder_groundtruth_images, 
            dataset, disease, name_image, key_dict[disease])
    return dir_image   

def transform_groundtruth(dir_folder_groundtruth_images, dataset, disease, name_image):
    dir_image_disease = get_dir_image_disease(dir_folder_groundtruth_images, dataset, disease, name_image)

    if os.path.exists(dir_image_disease):
        img = cv2.imread(dir_image_disease, 1)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if debug==2:
            plt.imshow(gray_image)
            plt.xticks([]), plt.yticks([])  
            plt.show()

        return gray_image

    else: 
        return 0        

def generate_qa_xiny():
    df = pd.read_csv('raw/idrid_qa_gt.csv')
    if debug ==2: print(df.head())

if __name__ == '__main__':
    main()
    # test()
