import nibabel as nib
import numpy as np
import os
import random
from sklearn.externals import joblib
from skimage import transform as tf
import torch
import scipy.ndimage as sind

# input_dir = 'in_patches'
# gt_dir = 'out_patches'

input_dir = 'in_patches_16_16_10'
gt_dir = 'out_patches_16_16_10'

def balanced_dataset(input_dir, gt_dir):

    positive_list = []
    negative_list = []

    input = os.listdir(input_dir)
    gt = os.listdir(gt_dir)
    input.sort()
    gt.sort()

    file_count = 0

    for x, y in zip(input, gt):
        file_count += 1
        print('Executing folder ', file_count)
        input_sub_dir = os.path.join(input_dir, x)
        gt_sub_dir = os.path.join(gt_dir, y)

        input_sub_dir_patches = os.listdir(input_sub_dir)
        gt_sub_dir_patches = os.listdir(gt_sub_dir)

        input_sub_dir_patches.sort()
        gt_sub_dir_patches.sort()

        for in_patch, gt_patch in zip(input_sub_dir_patches, gt_sub_dir_patches):
            input_image = os.path.join(input_sub_dir, in_patch)
            gt_image = os.path.join(gt_sub_dir, gt_patch)

            FLAIR_image_in = nib.load(input_image).get_data()
            FLAIR_image_in = np.array(FLAIR_image_in)

            FLAIR_image_gt = nib.load(gt_image).get_data()
            FLAIR_image_gt = np.array(FLAIR_image_gt)

            if FLAIR_image_gt.max() == 1.0:
                positive_list.append((FLAIR_image_in, 1.0))

                FLAIR_image_in_fliped =  FLAIR_image_in[::-1, :, :]
                positive_list.append((FLAIR_image_in_fliped, 1.0))

                FLAIR_image_in_rotated = sind.rotate(FLAIR_image_in,-15,reshape=False)
                positive_list.append((FLAIR_image_in_rotated, 1.0))


            # else:
            #     negative_list.append((FLAIR_image_in, 0.0))



    print('Executed all folders')





    return positive_list

train_data = balanced_dataset(input_dir,gt_dir)
len(train_data)
print('Saving to data file')
filename = 'positive.sav'
joblib.dump(train_data, filename)