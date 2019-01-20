import datasets.utils.svs_utils as svs_utils
import datasets.utils.normalization_utils as normalization_utils
import datasets.utils.print_utils as print_utils
import datasets.utils.paths_utils as path_utils
import datasets.utils.xml_utils as xml_utils
import datasets.utils.image_utils as image_utils
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import glob
from scipy.misc import imsave, imresize
from scipy.misc import toimage
import os
OPENSLIDE_PATH = "C:/Users/minhm/Documents/GitHub/bin/openslide-win64-20171122/bin"
if os.path.exists(OPENSLIDE_PATH):
    os.environ['PATH'] = OPENSLIDE_PATH + ";" + os.environ['PATH']
import openslide


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
DATASETS_PHOTOS_DIR = PROJECT_DIR + \
    "/data/raw/breast-cancer/ICIAR2018_BACH_Challenge/Photos/"
DATASETS_WSI_DIR = PROJECT_DIR + \
    "/data/raw/breast-cancer/ICIAR2018_BACH_Challenge/WSI/"
PREPROCESSED_IMAGE_PHOTOS_DIR = PROJECT_DIR + \
    "/data/raw/breast-cancer/preprocessed/Photos/"
PREPROCESSED_IMAGE_WSI_DIR = PROJECT_DIR + \
    "/data/raw/breast-cancer/preprocessed/WSI/"
PREPROCESSED_IMAGE_WSI_PATCH_DIR = PREPROCESSED_IMAGE_WSI_DIR + "patch/"
PREPROCESSED_IMAGE_WSI_GT_DIR = PREPROCESSED_IMAGE_WSI_DIR + "patch_gt/"


path_utils.make_dir(PREPROCESSED_IMAGE_PHOTOS_DIR)
path_utils.make_dir(PREPROCESSED_IMAGE_WSI_DIR)
path_utils.make_dir(PREPROCESSED_IMAGE_WSI_PATCH_DIR)
path_utils.make_dir(PREPROCESSED_IMAGE_WSI_GT_DIR)


LIST_BREAST_CLASS = ["Benign", "InSitu", "Invasive", "Normal"]


def normalize(path_in, path_out=None, is_debug=False, is_save=False, is_resize=True):
    im = Image.open(path_in)
    imarray = np.array(im)
    im_norm = normalization_utils.normalize_staining(imarray)
    im_norm = Image.fromarray(im_norm)

    if is_resize:
        im_resized = im_norm.resize((256, 256), Image.ANTIALIAS)
    else:
        im_resized = im_norm

    if is_debug:
        im.show()
        im_norm.show()
        im_resized.show()

    if is_save:
        im_resized.save(path_out)


def prepare_image_model(overwrite=False):
    for breast in LIST_BREAST_CLASS:
        folder_in = DATASETS_PHOTOS_DIR + breast
        folder_out = PREPROCESSED_IMAGE_PHOTOS_DIR + breast
        path_utils.make_dir(folder_out)
        img_dirs = glob.glob(os.path.join(folder_in, "*.tif"))
        for index, path_in in enumerate(img_dirs):
            filename = path_utils.get_filename_without_extension(path_in)
            path_out = folder_out + "/{}.jpg".format(filename)
            if not os.path.exists(path_out) or overwrite:
                print(">> processing {}/{}".format(index+1, len(img_dirs)))
                normalize(path_in, path_out=path_out, is_debug=False,
                          is_save=True, is_resize=True)
            else:
                print("skip {}/{}".format(index+1, len(img_dirs)))


def process_one_svs_one_xml(filename, img_svs, gt, upsampling_factor=4, is_debug=False):
    image_shape = img_svs.shape

    patch_size_img = (256*2**upsampling_factor, 256*2**upsampling_factor, 3)

    patch_indices = image_utils.compute_patch_indices(
        image_shape=image_shape, patch_size=patch_size_img, overlap=0)

    # remove negative index
    patch_indices = [item for item in patch_indices if item[0]
                     >= 0 and item[1] >= 0 and item[2] >= 0]
    patch_indices = [item for item in patch_indices if
                     item[0] + patch_size_img[0] < image_shape[0] and
                     item[1] + patch_size_img[1] < image_shape[1]]

    gt = gt[..., np.newaxis]
    patch_size_gt = (256*2**upsampling_factor, 256*2**upsampling_factor, 1)

    # for i in tqdm(range(len(patch_indices))):
    for i in range(len(patch_indices)):
        print(">> processing {}/{}".format(i+1, len(patch_indices)))
        outname = "{}_idx-{}-{}_ps-{}-{}".format(filename, str(patch_indices[i][0]), str(
            patch_indices[i][1]), str(patch_size_img[0]), str(patch_size_img[1]))
        patch_save_dir = PREPROCESSED_IMAGE_WSI_PATCH_DIR + outname + ".png"
        gt_save_dir = PREPROCESSED_IMAGE_WSI_GT_DIR + outname + ".png"

        if not os.path.exists(patch_save_dir) or not os.path.exists(gt_save_dir):
        # if "A01_idx-12864-14928_ps-16384-16384" in gt_save_dir:

            patch_extracted = image_utils.get_patch_from_3d_data(
                img_svs, patch_shape=patch_size_img, patch_index=patch_indices[i])
            gt_extracted = image_utils.get_patch_from_3d_data(
                gt, patch_shape=patch_size_gt, patch_index=patch_indices[i])

            patch_extracted_resized = imresize(
                patch_extracted, (256, 256, 3), interp="bilinear")
            gt_extracted_resized = np.squeeze(gt_extracted, axis=2)
            gt_extracted_resized = imresize(
                gt_extracted_resized, (256, 256), interp="nearest")

            # if np.count_nonzero(gt_extracted_resized)>256*256/8:
            gt_extracted_resized = image_utils.convert_gray_to_rgb(
                gt_extracted_resized)
            # try:
            #     patch_extracted_resized = normalization_utils.normalize_staining(patch_extracted_resized)
            # except:
            #     pass

            print(">> saving images...")
            # imsave(gt_save_dir, gt_extracted_resized)
            # imsave(patch_save_dir, patch_extracted_resized)
            gt_extracted_resized = Image.fromarray(gt_extracted_resized.astype(np.uint8))
            gt_extracted_resized.save(gt_save_dir, format="PNG")
            patch_extracted_resized = Image.fromarray(patch_extracted_resized.astype(np.uint8))
            patch_extracted_resized.save(patch_save_dir, format="PNG")


            if is_debug:
                im = Image.fromarray(gt_extracted_resized)
                im.show()


def prepare_question_model(is_save=False):
    patch_size = (2000, 2000)

    xml_dirs = glob.glob(os.path.join(DATASETS_WSI_DIR, "*.xml"))

    store = []
    for xml_dir in xml_dirs:
        filename = path_utils.get_filename_without_extension(xml_dir)
        img_dir = DATASETS_WSI_DIR + filename + ".svs"
        scan = openslide.OpenSlide(img_dir)
        dims = scan.dimensions

        # read svs
        img_svs = svs_utils.read_svs(
            img_dir, patch_size=patch_size, is_save=True)

        # read xml
        coords, labels, length, area, pixel_spacing = xml_utils.read_xml_breast(
            xml_dir)
        store += [[coords, labels, length, area, pixel_spacing]]
        save_dir = DATASETS_WSI_DIR+filename+'.png'
        gt = image_utils.generate_groundtruth_from_xml(
            save_dir, dims, coords, labels, is_debug=False, is_save=is_save)

        for upsampling_factor in [4,5,6]:
            process_one_svs_one_xml(filename, img_svs, gt, upsampling_factor=upsampling_factor, is_debug=False)


def main():
    print_utils.print_section("image model")
    prepare_image_model()
    print_utils.print_section("question model")
    prepare_question_model()


if __name__ == "__main__":
    main()
