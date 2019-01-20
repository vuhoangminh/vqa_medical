import numpy as np
from PIL import Image
import datasets.utils.normalization_utils as normalization_utils

path = "C:/Users/minhm/Documents/GitHub/vqa_idrid/data/raw/breast-cancer/ICIAR2018_BACH_Challenge/Photos/Benign/b001.tif"

path = "C:/Users/minhm/Documents/GitHub/kaggle_whale/dataset/samples/"

path = "C:/Users/minhm/Documents/GitHub/vqa_idrid/data/raw/m2cai16-tool-locations/JPEGImages/"


def test(path_in):

    im = Image.open(path_in)
    im.show()

    imarray = np.array(im)

    im_norm = normalization_utils.normalize_staining(imarray)

    im_norm = Image.fromarray(im_norm)
    im_norm.show()

    im_norm = im_norm.resize((256,256), Image.ANTIALIAS)
    im_norm.show()


for img in ["v01_002075", "v02_042400_flip"]:
    path_in = path + img + ".jpg"
    test(path_in)