
import nibabel as nib
import numpy as np

import sklearn.preprocessing as skp
import scipy.ndimage
import os
from PIL import Image , ImageOps


# normalise the Input Images.

OUT_DIR = 'normalized'

def normalize(input_image,image_name):
    # current_path = os.path.join(OUT_DIR, dir_name)
    #
    # if not os.path.exists(current_path):
    #     os.makedirs(current_path)


    image = nib.load(input_image)
    header = image.header
    affine = image.affine

    FLAIR_image = nib.load(input_image).get_fdata()
    FLAIR_image = np.array(FLAIR_image)

    FLAIR_image -= np.min(FLAIR_image)
    FLAIR_image /= np.max(FLAIR_image)

    FLAIR_image = nib.Nifti1Image(FLAIR_image, affine=affine, header=header)
    path = os.path.join(OUT_DIR,image_name )
    nib.save(FLAIR_image, path)


# file = open('your_file.txt','r')
# test_list = []
# for line in file:
#     line = line.split("\n")
#     line = line[0] + ".nii"
#     test_list.append(line)


input_dir = "E:\\abhivanth\\INPUT"

train_list = os.listdir(input_dir)



for subjects in train_list:
    path = os.path.join(input_dir,subjects)
    normalize(path,subjects)




# for subjects in os.listdir(HOME_DIR):
#     path = os.path.join(HOME_DIR, subjects)
#     normalize(path,subjects)






#
# for subjects in os.listdir(HOME_DIR):
#     path = os.path.join(HOME_DIR,subjects)
#     for folders in os.listdir(path):
#         if(subjects == 'ADNI 4'):
#           normalize(os.path.join(path,folders),folders,subjects)

