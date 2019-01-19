import numpy as np
import datasets.utils.image_utils as image_utils

image_shape = (36546, 63245, 3)
img = np.zeros(image_shape,dtype=np.uint8)
upsampling_factor = 4
patch_size = (256*2**upsampling_factor,256*2**upsampling_factor,3)

patch = image_utils.compute_patch_indices(image_shape=image_shape, patch_size=patch_size, overlap=0)

patch = [item for item in patch if item[0]>=0 and item[1]>=0 and item[2]>=0]

print(patch)

patch1 = image_utils.get_patch_from_3d_data(img, patch_shape=patch_size, patch_index=patch[0])

b = 2