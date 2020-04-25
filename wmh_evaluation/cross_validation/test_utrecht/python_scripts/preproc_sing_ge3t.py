import nibabel as nib
import numpy as np
import scipy.ndimage
import os
import torch
from sklearn.externals import joblib
import shutil

dir_input = 'dir_input' #'location where preprocessed input images to be saved'
dir_output = 'dir_output' #'location where preprocessed groung truth images to be saved'

del_input = 'dir_input'
del_output = 'dir_output'
del_groundtruth_slices = 'out_dir_groudtruth_slices'
del_input_slices = 'out_dir_input_slices'

for files in os.listdir(del_input):
    os.remove(os.path.join(del_input,files))

for files in os.listdir(del_output):
    os.remove(os.path.join(del_output,files))

for files in os.listdir(del_groundtruth_slices):
    os.remove(os.path.join(del_groundtruth_slices,files))

for files in os.listdir(del_input_slices):
    os.remove(os.path.join(del_input_slices,files))



'''
PREPROCESSING WMH DATASET
'''

def preprocess_ge3t(flair_path, t1_path):

    '''
    :param flair_path: path of flair image of each subject
    :param t1_path: path of t1 image of each subject
    :param image_name: name of subject
    :return: preprocessed image - cropped or padded to shape 200*200 and gaussian normalized with two modalities
                                  T1 and FLAIR
    '''

    channel_num = 2
    start_cut = 46

    rows_standard = 200
    cols_standard = 200

    FLAIR_image = nib.load(flair_path).get_data()
    T1_image = nib.load(t1_path).get_data()


    image_rows_Dataset = np.shape(FLAIR_image)[0]
    image_cols_Dataset = np.shape(FLAIR_image)[1]
    num_selected_slice = np.shape(FLAIR_image)[2]

    FLAIR_image = np.float32(FLAIR_image)
    T1_image = np.float32(T1_image)

    # print(FLAIR_image.shape)

    brain_mask_FLAIR = np.ndarray((image_rows_Dataset, image_cols_Dataset,num_selected_slice), dtype=np.float32)
    brain_mask_T1 = np.ndarray((image_rows_Dataset, image_cols_Dataset,num_selected_slice), dtype=np.float32)
    FLAIR_image_suitable = np.ndarray((rows_standard, cols_standard, num_selected_slice), dtype=np.float32)
    T1_image_suitable = np.ndarray((rows_standard, cols_standard, num_selected_slice), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >= 70] = 1
    brain_mask_FLAIR[FLAIR_image < 70] = 0

    for iii in range(num_selected_slice):
        brain_mask_FLAIR[:, :, iii] = scipy.ndimage.morphology.binary_fill_holes(
            brain_mask_FLAIR[:, :, iii])  # fill the holes inside brain
        # ------Gaussion Normalization
    FLAIR_image -= np.mean(FLAIR_image[brain_mask_FLAIR == 1])  # Gaussion Normalization
    FLAIR_image /= np.std(FLAIR_image[brain_mask_FLAIR == 1])

    FLAIR_image_suitable[...] = np.min(FLAIR_image)
    # print(cols_standard, image_rows_Dataset)
    FLAIR_image_suitable[int(cols_standard/2-image_rows_Dataset/2):int(cols_standard/2+image_rows_Dataset/2),:, :] = \
        FLAIR_image[:, start_cut:start_cut+rows_standard, :]

    # T1 -----------------------------------------------
    brain_mask_T1[T1_image >= 30] = 1
    brain_mask_T1[T1_image < 30] = 0
    for iii in range(num_selected_slice):
        brain_mask_T1[:, :, iii] = scipy.ndimage.morphology.binary_fill_holes(
            brain_mask_T1[:, :, iii])  # fill the holes inside brain
        # ------Gaussian Normalization
    T1_image -= np.mean(T1_image[brain_mask_T1 == 1])  # Gaussian Normalization
    T1_image /= np.std(T1_image[brain_mask_T1 == 1])

    T1_image_suitable[...] = np.min(T1_image)
    T1_image_suitable[
    int(cols_standard / 2 - image_rows_Dataset / 2):int(cols_standard / 2 + image_rows_Dataset / 2), :, :] = \
        T1_image[:, start_cut:start_cut + rows_standard, :]
    # ---------------------------------------------------
    FLAIR_image_suitable = FLAIR_image_suitable[..., np.newaxis]
    T1_image_suitable = T1_image_suitable[..., np.newaxis]

    imgs_two_channels = np.concatenate((FLAIR_image_suitable, T1_image_suitable), axis=3)
    # print(np.shape(imgs_two_channels))
    imgs_two_channels = nib.Nifti1Image(imgs_two_channels, affine=None, header=None)
    return imgs_two_channels


def preprocess_seg_ge3t(seg_path):
    '''
    :param seg_path: path of labeled ground truth
    :param image_name: subject name
    :return: ground truth cropped or padded to 200*200
    '''
    channel_num = 2
    start_cut = 46

    rows_standard = 200
    cols_standard = 200

    image = nib.load(seg_path)
    affine = image.affine
    header = image.header

    SEG_image = nib.load(seg_path).get_data()

    image_rows_Dataset = np.shape(SEG_image)[0]
    image_cols_Dataset = np.shape(SEG_image)[1]
    num_selected_slice = np.shape(SEG_image)[2]

    SEG_image = np.float32(SEG_image)

    imgs_two_channels = np.ndarray((rows_standard, cols_standard, num_selected_slice, channel_num), dtype=np.float32)
    SEG_image_suitable = np.ndarray((rows_standard, cols_standard, num_selected_slice), dtype=np.float32)

    SEG_image_suitable[...] = np.min(SEG_image)
    # print(cols_standard, image_rows_Dataset)
    SEG_image_suitable[int(cols_standard/2-image_rows_Dataset/2):int(cols_standard/2+image_rows_Dataset/2),:, :] = \
        SEG_image[:, start_cut:start_cut+rows_standard, :]


    # ---------------------------------------------------
    SEG_image_suitable = SEG_image_suitable[..., np.newaxis]

    SEG_image = nib.Nifti1Image(SEG_image_suitable, affine=affine, header=header)
    return SEG_image


def utrecht_singapore_preprocessing_t1(input_image):
    rows_standard = 200
    cols_standard = 200

    image = nib.load(input_image)
    header = image.header
    affine = image.affine

    t1_image = nib.load(input_image).get_fdata()
    t1_image = np.array(t1_image)

    num_selected_slice = np.shape(t1_image)[2]  # Z direction
    image_rows_Dataset = np.shape(t1_image)[0]  # X
    image_cols_Dataset = np.shape(t1_image)[1]  # Y

    brain_mask_t1 = np.ndarray((image_rows_Dataset, image_cols_Dataset, num_selected_slice), dtype=np.float32)


    brain_mask_t1[t1_image >= 30] = 1
    brain_mask_t1[t1_image < 30] = 0

    for iii in range(num_selected_slice):
        brain_mask_t1[:, :, iii] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_t1[:, :, iii])

    t1_image = t1_image[
                  int(image_rows_Dataset / 2 - rows_standard / 2):int(image_rows_Dataset / 2 + rows_standard / 2),
                  int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2), :]

    brain_mask_FLAIR = brain_mask_t1[
                       int(image_rows_Dataset / 2 - rows_standard / 2):int(image_rows_Dataset / 2 + rows_standard / 2),
                       int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2),
                       :]

    t1_image -= np.mean(t1_image[brain_mask_FLAIR == 1])
    t1_image /= np.std(t1_image[brain_mask_FLAIR == 1])

    t1_image = nib.Nifti1Image(t1_image, affine=affine, header=header)
    # nib.save(FLAIR_image, image_name+'_tI_pp.nii.gz')
    return t1_image

def utrecht_singapore_preprocessing_flair(input_image):
    rows_standard = 200
    cols_standard = 200

    image = nib.load(input_image)
    header = image.header
    affine = image.affine

    FLAIR_image = nib.load(input_image).get_fdata()
    FLAIR_image = np.array(FLAIR_image)

    num_selected_slice = np.shape(FLAIR_image)[2]  # Z direction
    image_rows_Dataset = np.shape(FLAIR_image)[0]  # X
    image_cols_Dataset = np.shape(FLAIR_image)[1]  # Y

    brain_mask_FLAIR = np.ndarray((image_rows_Dataset, image_cols_Dataset, num_selected_slice), dtype=np.float32)


    brain_mask_FLAIR[FLAIR_image >= 70] = 1
    brain_mask_FLAIR[FLAIR_image < 70] = 0

    for iii in range(num_selected_slice):
        brain_mask_FLAIR[:, :, iii] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[:, :, iii])

    FLAIR_image = FLAIR_image[
                  int(image_rows_Dataset / 2 - rows_standard / 2):int(image_rows_Dataset / 2 + rows_standard / 2),
                  int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2), :]

    brain_mask_FLAIR = brain_mask_FLAIR[
                       int(image_rows_Dataset / 2 - rows_standard / 2):int(image_rows_Dataset / 2 + rows_standard / 2),
                       int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2),
                       :]

    FLAIR_image -= np.mean(FLAIR_image[brain_mask_FLAIR == 1])
    FLAIR_image /= np.std(FLAIR_image[brain_mask_FLAIR == 1])

    FLAIR_image = nib.Nifti1Image(FLAIR_image, affine=affine, header=header)
    # nib.save(FLAIR_image, image_name+'_seg_pp.nii.gz')
    return FLAIR_image

def utrecht_singapore_out_preprocessing(input_image):
    rows_standard = 200
    cols_standard = 200

    image = nib.load(input_image)
    header = image.header
    affine = image.affine

    SEG_image = nib.load(input_image).get_data()
    SEG_image = np.array(SEG_image)

    image_rows_Dataset = np.shape(SEG_image)[0]  # X
    image_cols_Dataset = np.shape(SEG_image)[1]  # Y

    SEG_image = SEG_image[
                  int(image_rows_Dataset / 2 - rows_standard / 2):int(image_rows_Dataset / 2 + rows_standard / 2),
                  int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2), :]

    SEG_image = nib.Nifti1Image(SEG_image, affine=affine, header=header)
    return SEG_image

GE3T_DIR = 'P:\\Clean_Datasets_Backup\\WMH\\Amsterdam_GE3T\\GE3T'

'''
takes FLAIR and TI images of each subject from GE3T hospital and preprocess the image 
a) crops or pads to 200x200
b) normalization
c) two modalities/channels
'''

train_images = os.listdir(GE3T_DIR)

for subjects in train_images:
    folder_path = os.path.join(GE3T_DIR, subjects)
    files = os.listdir(os.path.join(GE3T_DIR, subjects))
    for file in files:
        if(file == 'pre'):
            image_path = os.path.join(folder_path,file)
            imgs_two_channels = preprocess_ge3t(os.path.join(image_path, 'FLAIR.nii.gz'),
                                  os.path.join(image_path, 'T1.nii.gz'))
            nib.save(imgs_two_channels, dir_input + '\\' + subjects + '_pp.nii.gz')

'''
preprocess the ground truth of GE3T subjects
'''

for subjects in train_images:
    folder_path = os.path.join(GE3T_DIR, subjects)
    files = os.listdir(os.path.join(GE3T_DIR, subjects))
    for file in files:
        if(file == 'wmh.nii.gz'):
            SEG_image = preprocess_seg_ge3t(os.path.join(folder_path,file))
            nib.save(SEG_image, dir_output + '\\' + subjects + '_pp.nii.gz')


'''
preprocess augmented ge3t dataset
'''

GE3T_AUG_DIR = 'D:\\wmh\\ge3t_aug'

train_images = os.listdir(GE3T_AUG_DIR)

for subjects in train_images:
    folder_path = os.path.join(GE3T_AUG_DIR, subjects)

    imgs_two_channels = preprocess_ge3t(os.path.join(folder_path, 'flair.nii.gz'),
                                        os.path.join(folder_path, 't1.nii.gz'))
    nib.save(imgs_two_channels, dir_input+'\\'+subjects+'_pp.nii.gz')

    SEG_image = preprocess_seg_ge3t(os.path.join(folder_path, 'wmh.nii.gz'))
    nib.save(SEG_image, dir_output + '\\' + subjects + '_pp.nii.gz')


SINGAPORE_DIR = 'P:\\Clean_Datasets_Backup\\WMH\\Singapore\\Singapore'

'''
takes FLAIR and TI images of each subject from Singapore hospital and preprocess the image
a) crops or pads to 200x200
b) normalization
c) two modalities/channels
'''

train_images = os.listdir(SINGAPORE_DIR)

for subjects in train_images:
    folder_path = os.path.join(SINGAPORE_DIR, subjects)
    files = os.listdir(os.path.join(SINGAPORE_DIR, subjects))
    for file in files:
        if(file == 'pre'):
            image_path = os.path.join(folder_path,file)
            flairImage = utrecht_singapore_preprocessing_flair(os.path.join(image_path, 'FLAIR.nii.gz'))
            flairImage = flairImage.get_fdata()[..., np.newaxis]
            T1Image = utrecht_singapore_preprocessing_t1(os.path.join(image_path, 'T1.nii.gz'))
            T1Image = T1Image.get_fdata()[..., np.newaxis]

            imgs_two_channels = np.concatenate((flairImage, T1Image), axis=3)
            imgs_two_channels = nib.Nifti1Image(imgs_two_channels, affine=None, header=None)
            nib.save(imgs_two_channels, dir_input+'\\'+subjects+'_pp.nii.gz')

'''
preprocess the ground truth of Singapore subjects
'''

for subjects in train_images:
    folder_path = os.path.join(SINGAPORE_DIR, subjects)
    files = os.listdir(os.path.join(SINGAPORE_DIR, subjects))
    for file in files:
        if(file == 'wmh.nii.gz'):
            SEG_image = utrecht_singapore_out_preprocessing(os.path.join(folder_path,file))
            nib.save(SEG_image, dir_output + '\\' + subjects + '_pp.nii.gz')


'''
preprocess augmented singapore dataset
'''

SINGAPORE_AUG_DIR = 'D:\\wmh\\singapore_aug'

train_images = os.listdir(SINGAPORE_AUG_DIR)

for subjects in train_images:
    folder_path = os.path.join(SINGAPORE_AUG_DIR, subjects)
    # files = os.listdir(os.path.join(SINGAPORE_DIR, subjects))
    flairImage = utrecht_singapore_preprocessing_flair(os.path.join(folder_path, 'flair.nii.gz'))
    flairImage = flairImage.get_fdata()[..., np.newaxis]
    T1Image = utrecht_singapore_preprocessing_t1(os.path.join(folder_path, 't1.nii.gz'))
    T1Image = T1Image.get_fdata()[..., np.newaxis]

    imgs_two_channels = np.concatenate((flairImage, T1Image), axis=3)
    imgs_two_channels = nib.Nifti1Image(imgs_two_channels, affine=None, header=None)
    nib.save(imgs_two_channels, dir_input+'\\'+subjects+'_pp.nii.gz')

    SEG_image = utrecht_singapore_out_preprocessing(os.path.join(folder_path, 'wmh.nii.gz'))
    nib.save(SEG_image, dir_output + '\\' + subjects + '_pp.nii.gz')
'''
PREPARE AXIAL SLICES
'''

'''
PREPARE AXIAL SLICES
'''

out_dir_input_slices = 'out_dir_input_slices' #'location of slices to be saved'

for sub in os.listdir(dir_input):
    img = nib.load(os.path.join(dir_input,sub)).get_data()

    sub = sub.split('.')[0]

    for i in range(img.shape[2]):
        slice = img[:,:,i,:]        # the image is in x,y,z,c format
        slice = nib.Nifti1Image(slice, affine=None, header=None)
        nib.save(slice, os.path.join(out_dir_input_slices,f'{sub}_{i:02}.nii.gz'))


out_dir_groudtruth_slices = 'out_dir_groudtruth_slices' #'location of slices to be saved'

for sub in os.listdir(dir_output):
    img = nib.load(os.path.join(dir_output,sub)).get_data()

    sub = sub.split('.')[0]

    for i in range(img.shape[2]):
        slice = img[:,:,i]
        slice = nib.Nifti1Image(slice, affine=None, header=None)
        nib.save(slice, os.path.join(out_dir_groudtruth_slices,f'{sub}_{i:02}.nii.gz'))

'''
CREATE x,y PAIR for training and validation
'''

data = []

for imgs in os.listdir(out_dir_input_slices):
    img = nib.load(os.path.join(out_dir_input_slices,imgs)).get_data()
    out_img = nib.load(os.path.join(out_dir_groudtruth_slices,imgs)).get_data()

    img = np.array(img)
    out_img = np.array(out_img)
    out_img = np.squeeze(out_img)

    img = np.transpose(img, (2, 0, 1))

    img = torch.from_numpy(img).type(torch.FloatTensor)
    out_img = torch.from_numpy(out_img).type(torch.FloatTensor)

    data.append([img,out_img])

from sklearn.externals import joblib
filename = 'data_sing_ge3t.sav'
joblib.dump(data, filename)





