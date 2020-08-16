import os
import nibabel as nib


'''
This file separates from  'test_file.txt' to 'test_input' and 'test_gt' directories which has 
test images and groundTruths respectively
'''





Dir_gt = 'D:\\de\\12_credit\\crebral_microbleeds\\adni\\OUTPUT'
Dir_in = 'D:\\de\\12_credit\\crebral_microbleeds\\adni\\INPUT'

save_in = 'test_input'
save_gt = 'test_gt'

input_sub_dir_patches = []
gt_sub_dir_patches = []

os.chdir('D:\\de\\12_credit\\crebral_microbleeds\\cmb code\\fcn_cmb')

f = open("test_file.txt", "r")
for x in f:
    os.chdir('D:\\de\\12_credit\\crebral_microbleeds\\adni')

    x = x.split("\n")

    image_name = x[0] + ".nii"

    path_in = os.path.join(Dir_in, image_name)
    path_gt = os.path.join(Dir_gt, image_name)

    input_image = nib.load(path_in)

    gt_image = nib.load(path_gt)

    os.chdir('D:\\de\\12_credit\\crebral_microbleeds\\cmb code\\fcn_cmb')

    save_path_in = os.path.join(save_in, image_name)
    image_name = x[0] + '_gt' + ".nii"
    save_path_gt = os.path.join(save_gt, image_name)

    nib.save(input_image, save_path_in)
    nib.save(gt_image, save_path_gt)

