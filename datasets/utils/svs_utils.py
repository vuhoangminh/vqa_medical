import os
OPENSLIDE_PATH = "C:/Users/minhm/Documents/GitHub/bin/openslide-win64-20171122/bin"
if os.path.exists(OPENSLIDE_PATH):
    os.environ['PATH'] = OPENSLIDE_PATH + ";" + os.environ['PATH']
import numpy as np
from tqdm import tqdm
import openslide
from matplotlib import pyplot as plt
from scipy.misc import imsave, imresize
from openslide import open_slide
import datasets.utils.paths_utils as path_utils


def read_svs(svs_path, patch_size, is_save=False):
    scan = openslide.OpenSlide(svs_path)
    filename_without_ext = path_utils.get_filename_without_extension(svs_path)
    parent_path = path_utils.get_parent_dir(svs_path)
    npy_path = "{}/{}.npy".format(parent_path, filename_without_ext) 

    if os.path.exists(npy_path):
        print(">> loading", npy_path)
        img_np = np.load(npy_path)
    else:
        orig_w = np.int(scan.properties.get('aperio.OriginalWidth'))
        orig_h = np.int(scan.properties.get('aperio.OriginalHeight'))

        # create an array to store our image
        img_np = np.zeros((orig_w,orig_h,3),dtype=np.uint8)

        print(">> reading", svs_path)
        for r in tqdm(range(0,orig_w,patch_size[0])):
            for c in range(0, orig_h,patch_size[1]):
                if c+patch_size[1] > orig_h and r+patch_size[0]<= orig_w:
                    p = orig_h-c
                    img = np.array(scan.read_region((c,r),0,(p,patch_size[1])),dtype=np.uint8)[...,0:3]
                elif c+patch_size[1] <= orig_h and r+patch_size[0] > orig_w:
                    p = orig_w-r
                    img = np.array(scan.read_region((c,r),0,(patch_size[0],p)),dtype=np.uint8)[...,0:3]
                elif  c+patch_size[1] > orig_h and r+patch_size[0] > orig_w:
                    p = orig_h-c
                    pp = orig_w-r
                    img = np.array(scan.read_region((c,r),0,(p,pp)),dtype=np.uint8)[...,0:3]
                else:    
                    img = np.array(scan.read_region((c,r),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0:3]
                img_np[r:r+patch_size[0],c:c+patch_size[1]] = img

        scan.close
        if is_save:
            np.save(npy_path, img_np)

    return img_np