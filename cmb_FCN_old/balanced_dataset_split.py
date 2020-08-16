import nibabel as nib
import numpy as np
import os
import random
import joblib
#from sklearn.externals import joblib
#from skimage import transform as tf
#import torch

import scipy.ndimage as sind



# To convert the Input patches and Output patches to a '.sav ' format as x,y pair
# Equal no of input and output patches

# input_dir = 'in_patches'
# gt_dir = 'out_patches'

input_dir = 'in_patches_16_16_10'
gt_dir = 'out_patches_16_16_10'


'''
This file splits random 450 images as train images and 50 images as test images.
and saves the Ids of the test images.

This file pairs groundTruth with corresponding image patch(X,Y - Pairs) and saves it as .sav file.

Augmentation of positive images patches is done.
'''



def balanced_dataset(input_dir, gt_dir):

    positive_list = []
    negative_list = []

    input = os.listdir(input_dir)
    gt = os.listdir(gt_dir)
    gt_new = []

    input_new = random.sample(input,450)

    for input_folders in input_new:
        mypath = os.path.join(gt_dir,input_folders)
        gt_new.append(mypath)




    validation_list_input = [element for element in input if element not in input_new]
    validation_list_output = [element for element in gt if element not in gt_new]

    validation_list_input.sort()

    with open('test_file_input.txt', 'w') as f:
        for item in validation_list_input:
            f.write("%s\n" % item)
    with open('test_file_output.txt', 'w') as f:
        for item in validation_list_output:
            f.write("%s\n" % item)

    input_new.sort()
    gt_new.sort()

    print(input_new.__len__(),gt_new.__len__(),validation_list_input.__len__())

    file_count = 0

    for x, y in zip(input_new, gt_new):
        file_count += 1
        print('Executing folder ', file_count)
        input_sub_dir = os.path.join(input_dir, x)
        gt_sub_dir = y

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

                FLAIR_image_in_rotated_1 = sind.rotate(FLAIR_image_in,-12,reshape=False)
                positive_list.append((FLAIR_image_in_rotated_1, 1.0))

                FLAIR_image_in_rotated_2 = sind.rotate(FLAIR_image_in, 12, reshape=False)
                positive_list.append((FLAIR_image_in_rotated_2, 1.0))

                FLAIR_image_shifted_1 = np.roll(FLAIR_image_in,10,0)
                positive_list.append((FLAIR_image_shifted_1, 1.0))

                FLAIR_image_shifted_2 = np.roll(FLAIR_image_in, -10, 0)
                positive_list.append((FLAIR_image_shifted_2, 1.0))





            else:
                negative_list.append((FLAIR_image_in, 0.0))


    print('Executed all folders')

    positive_count  = len(positive_list)
    negative_list_1 = random.sample(negative_list, positive_count)

    balanced_list = positive_list + negative_list_1
    print(len(positive_list))
    print(len(negative_list_1))


    random.shuffle(balanced_list)

    print(len(balanced_list))

    return balanced_list

train_data = balanced_dataset(input_dir,gt_dir)

print(len(train_data))
print('Saving to data file')
# filename = 'balanced_dataset.sav'
filename = 'balanced_dataset_16_16_10_augmented.sav'
joblib.dump(train_data, filename)