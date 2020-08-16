import nibabel as nib
import numpy as np
import os
import random
import joblib

import scipy.ndimage as sind

'''
This file pairs groundTruth with corresponding image patch(X,Y - Pairs) and saves it as .sav file.
'''

# To convert the Input patches and Output patches to a '.sav ' format as x,y pair
# Equal no of input and output patches

# input_dir = 'in_patches'
# gt_dir = 'out_patches'

input_dir = 'in_patches_16_16_10'
gt_dir = 'out_patches_16_16_10'

input_path_new = []
gt_path_new = []

path = 'E:\\abhivanth\\cmb_FCN\\splitfiles\\split2\\test_file.txt'


def balanced_dataset(input_path_new):
    before_augment = 0
    positive_list = []
    negative_list = []

    input_list = os.listdir(input_dir)
    gt_list = os.listdir(gt_dir)

    input = [element for element in input_list if element in input_path_new]
    gt = [element for element in gt_list if element in input_path_new]

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

                before_augment += 1

                FLAIR_image_in_fliped = FLAIR_image_in[::-1, :, :]
                positive_list.append((FLAIR_image_in_fliped, 1.0))

                FLAIR_image_in_rotated_1 = sind.rotate(FLAIR_image_in, -12, reshape=False)
                positive_list.append((FLAIR_image_in_rotated_1, 1.0))

                FLAIR_image_in_rotated_2 = sind.rotate(FLAIR_image_in, 12, reshape=False)
                positive_list.append((FLAIR_image_in_rotated_2, 1.0))

                FLAIR_image_shifted_1 = np.roll(FLAIR_image_in, 10, 0)
                positive_list.append((FLAIR_image_shifted_1, 1.0))

                FLAIR_image_shifted_2 = np.roll(FLAIR_image_in, -10, 0)
                positive_list.append((FLAIR_image_shifted_2, 1.0))

            else:
                negative_list.append((FLAIR_image_in, 0.0))

    print('Executed all folders')

    positive_count = len(positive_list)
    negative_list_1 = random.sample(negative_list, positive_count)

    balanced_list = positive_list + negative_list_1

    print(len(positive_list))
    print(len(negative_list_1))

    print('before augment', before_augment)

    random.shuffle(balanced_list)

    print(len(balanced_list))

    return balanced_list


f = open(path, "r")

for x in f:
    x = x.split("\n")

    image_name = x[0]
    image_name, ext = image_name.split(".")

    input_path_new.append(image_name)

train_data = balanced_dataset(input_path_new)
#
print(len(train_data))
print('Saving to data file')
filename = 'balanced_dataset_test.sav'
# filename = 'test_dataset_16_16_10.sav'
joblib.dump(train_data, filename)
