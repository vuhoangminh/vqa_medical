# -*- coding: utf-8 -*-
"""
ICIAR2018 - Grand Challenge on Breast Cancer Histology Images
https://iciar2018-challenge.grand-challenge.org/home/
"""

import os
import glob
os.environ['PATH'] = "C:/Users/minhm/Documents/GitHub/bin/openslide-win64-20171122/bin" + ";" + os.environ['PATH']
import openslide

import datasets.utils.paths_utils as path_utils
import datasets.utils.xml_utils as xml_utils
import datasets.utils.image_utils as image_utils



if __name__=='__main__':
    
    folder_name = 'C:/Users/minhm/Documents/GitHub/vqa_idrid/data/raw/breast-cancer/ICIAR2018_BACH_Challenge/WSI/' #path to the dataset folder
    img_dirs = glob.glob(os.path.join(folder_name, "*.xml"))
    store = []
    for img_dir in img_dirs:
        file_name = path_utils.get_filename_without_extension(img_dir)
        
        print('Reading scan',file_name)
        scan = openslide.OpenSlide(folder_name+file_name+'.svs')
        dims = scan.dimensions
        print('Generating thumbnail')
        
        coords,labels,length,area,pixel_spacing = xml_utils.read_xml_breast(img_dir)
        store += [[coords,labels,length,area,pixel_spacing]]
        gt = image_utils.generate_groundtruth_from_xml(folder_name+file_name+'.png', dims, coords, labels, is_debug=False)
    
    

