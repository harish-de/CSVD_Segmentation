import h5py
import os
import nibabel as nib
import numpy as np
import random


def normalize(image):

    image -= np.min(image)
    image /= np.max(image)

    return  image


cmb_path_input = 'C:\\Users\\bhuva\\PycharmProjects\\detect_cmb\\patches\\inputpatches'

cmb_path_gt  = 'C:\\Users\\bhuva\\PycharmProjects\\detect_cmb\\patches\\\output_patches'
#
hf = h5py.File('cmbdata_new.h5', 'a')

grp1 = hf.create_group('input_images')
grp2 = hf.create_group('output_images')

gt_list = os.listdir(cmb_path_gt)
input_list = os.listdir(cmb_path_input)
gt_list.sort()
input_list.sort()
xy_list = list(range(0,100))
# print(xy_list)
count = 0

shape = (16,16,10)


for input_files,gt_files in zip(input_list,gt_list):

    current_path_input = os.path.join(cmb_path_input,input_files)
    image_list_input = os.listdir(current_path_input)
    image_list_input.sort()

    current_path_output = os.path.join(cmb_path_gt, gt_files)
    image_list_output = os.listdir(current_path_output)
    image_list_output.sort()
    for input,output in zip(image_list_input,image_list_output):
        input_patch_path = os.path.join(current_path_input,input)
        output_patch_path = os.path.join(current_path_output,output)

        input_image_patch = nib.load(input_patch_path).get_fdata().astype(np.float32)
        normalize(input_image_patch)

        output_image_patch = nib.load(output_patch_path).get_fdata().astype(np.float32)
        shape = input_image_patch.shape


        grp1.create_dataset(input,shape=shape,dtype=h5py.h5t.IEEE_F32BE,data=input_image_patch,compression="gzip", compression_opts=5)
        grp2.create_dataset(output, shape=shape, dtype=h5py.h5t.IEEE_F32BE, data=output_image_patch,compression="gzip", compression_opts=5)
        count += 1

        


hf.close()
print('finished')






























