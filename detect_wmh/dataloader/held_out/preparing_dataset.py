import os
import random
import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
from skimage import transform as tf


# import tensorflow as tf
# from keras.preprocessing.image import apply_affine_transform

class preprocess():

    '''
    PREPROCESSING WMH DATASET
    '''

    data_2Dslices = []

    def preprocess_ge3t(self, flair_path, t1_path):

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

        brain_mask_FLAIR = np.ndarray((image_rows_Dataset, image_cols_Dataset, num_selected_slice), dtype=np.float32)
        brain_mask_T1 = np.ndarray((image_rows_Dataset, image_cols_Dataset, num_selected_slice), dtype=np.float32)
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

        FLAIR_image_suitable[
        int(cols_standard / 2 - image_rows_Dataset / 2):int(cols_standard / 2 + image_rows_Dataset / 2), :, :] = \
            FLAIR_image[:, start_cut:start_cut + rows_standard, :]

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

        imgs_two_channels = nib.Nifti1Image(imgs_two_channels, affine=None, header=None)
        return imgs_two_channels

    def preprocess_seg_ge3t(self, seg_path):
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

        imgs_two_channels = np.ndarray((rows_standard, cols_standard, num_selected_slice, channel_num),
                                       dtype=np.float32)
        SEG_image_suitable = np.ndarray((rows_standard, cols_standard, num_selected_slice), dtype=np.float32)

        SEG_image_suitable[...] = np.min(SEG_image)

        SEG_image_suitable[
        int(cols_standard / 2 - image_rows_Dataset / 2):int(cols_standard / 2 + image_rows_Dataset / 2), :, :] = \
            SEG_image[:, start_cut:start_cut + rows_standard, :]

        # ---------------------------------------------------
        SEG_image_suitable = SEG_image_suitable[..., np.newaxis]

        SEG_image = nib.Nifti1Image(SEG_image_suitable, affine=affine, header=header)
        return SEG_image

    def utrecht_singapore_preprocessing_t1(self, input_image):
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
                           int(image_rows_Dataset / 2 - rows_standard / 2):int(
                               image_rows_Dataset / 2 + rows_standard / 2),
                           int(image_cols_Dataset / 2 - cols_standard / 2):int(
                               image_cols_Dataset / 2 + cols_standard / 2),
                           :]

        t1_image -= np.mean(t1_image[brain_mask_FLAIR == 1])
        t1_image /= np.std(t1_image[brain_mask_FLAIR == 1])

        t1_image = nib.Nifti1Image(t1_image, affine=affine, header=header)

        return t1_image

    def utrecht_singapore_preprocessing_flair(self, input_image):
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
                      int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2),
                      :]

        brain_mask_FLAIR = brain_mask_FLAIR[
                           int(image_rows_Dataset / 2 - rows_standard / 2):int(
                               image_rows_Dataset / 2 + rows_standard / 2),
                           int(image_cols_Dataset / 2 - cols_standard / 2):int(
                               image_cols_Dataset / 2 + cols_standard / 2),
                           :]

        FLAIR_image -= np.mean(FLAIR_image[brain_mask_FLAIR == 1])
        FLAIR_image /= np.std(FLAIR_image[brain_mask_FLAIR == 1])

        FLAIR_image = nib.Nifti1Image(FLAIR_image, affine=affine, header=header)
        return FLAIR_image

    def utrecht_singapore_out_preprocessing(self, input_image):
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


    def rotate(self, x, y):
        angle = random.randint(-5, 5)
        x_np = np.array(x)
        y_np = np.array(y)
        # x_rot = scipy.ndimage.rotate(x_np,angle)
        # y_rot = scipy.ndimage.rotate(y_np,angle)

        x_rot = tf.rotate(x_np, angle)
        y_rot = tf.rotate(y_np, angle)
        input_slice = torch.from_numpy(x_rot).type(torch.FloatTensor)
        output_slice = torch.from_numpy(y_rot).type(torch.FloatTensor)
        self.data_2Dslices.append([input_slice, output_slice])

    def shear(self, x, y):
        angle = random.randint(-8, 8)
        affine_shear = tf.AffineTransform(shear=angle)
        x_shear = tf.warp(x,inverse_map=affine_shear)
        y_shear = tf.warp(y,inverse_map=affine_shear)
        scipy.ndimage.affine_transform()
        input_slice = torch.from_numpy(x_shear).type(torch.FloatTensor)
        output_slice = torch.from_numpy(y_shear).type(torch.FloatTensor)
        self.data_2Dslices.append([input_slice, output_slice])

    def getAxialSlices(self, input_img, output_img, aug):

        input_img = input_img.get_data()
        output_img = output_img.get_data()

        for i in range(input_img.shape[2]):
            input_slice = input_img[:, :, i, :]  # the image is in x,y,z,c format
            input_slice = np.array(input_slice)
            input_slice = np.transpose(input_slice, (2, 0, 1))

            output_slice = output_img[:, :, i]
            output_slice = np.array(output_slice)
            output_slice = np.squeeze(output_slice)

            if(aug):
                for iter in range(0,1):
                    self.rotate(input_slice,output_slice)
                    # self.shear(input_slice,output_slice)

            input_slice = torch.from_numpy(input_slice).type(torch.FloatTensor)
            output_slice = torch.from_numpy(output_slice).type(torch.FloatTensor)

            self.data_2Dslices.append([input_slice,output_slice])

        return self.data_2Dslices

    def create_dataset(self, directory, num_training_images, aug=False):
        ge3t_dir = directory + '\\Amsterdam_GE3T\\GE3T'
        sing_dir = directory + '\\Singapore\\Singapore'
        utre_dir = directory + '\\Utrecht\\Utrecht'

        test_imgs = []
    
        train_images = random.sample(os.listdir(ge3t_dir), num_training_images)
        for subjects in os.listdir(ge3t_dir):
            if subjects not in train_images:
                test_imgs.append(os.path.join(ge3t_dir,subjects))
    
        for subjects in train_images:
            folder_path = os.path.join(ge3t_dir, subjects)
            files = os.listdir(os.path.join(ge3t_dir, subjects))
            for file in files:
                if (file == 'pre'):
                    image_path = os.path.join(folder_path, file)
                    imgs_two_channels = self.preprocess_ge3t(os.path.join(image_path, 'FLAIR.nii.gz'),
                                                        os.path.join(image_path, 'T1.nii.gz'))
                if (file == 'wmh.nii.gz'):
                    SEG_image = self.preprocess_seg_ge3t(os.path.join(folder_path, file))

            self.getAxialSlices(imgs_two_channels,SEG_image, aug)

        train_images = random.sample(os.listdir(sing_dir), num_training_images)
        for subjects in os.listdir(sing_dir):
            if subjects not in train_images:
                test_imgs.append(os.path.join(sing_dir,subjects))
    
        for subjects in train_images:
            folder_path = os.path.join(sing_dir, subjects)
            files = os.listdir(os.path.join(sing_dir, subjects))
            for file in files:
                if (file == 'pre'):
                    image_path = os.path.join(folder_path, file)
                    flairImage = self.utrecht_singapore_preprocessing_flair(os.path.join(image_path, 'FLAIR.nii.gz'))
                    flairImage = flairImage.get_fdata()[..., np.newaxis]
                    T1Image = self.utrecht_singapore_preprocessing_t1(os.path.join(image_path, 'T1.nii.gz'))
                    T1Image = T1Image.get_fdata()[..., np.newaxis]
    
                    imgs_two_channels = np.concatenate((flairImage, T1Image), axis=3)
                    imgs_two_channels = nib.Nifti1Image(imgs_two_channels, affine=None, header=None)
    
                if (file == 'wmh.nii.gz'):
                    SEG_image = self.utrecht_singapore_out_preprocessing(os.path.join(folder_path, file))

            self.getAxialSlices(imgs_two_channels, SEG_image, aug)
    
        train_images = random.sample(os.listdir(utre_dir), num_training_images)
        for subjects in os.listdir(utre_dir):
            if subjects not in train_images:
                test_imgs.append(os.path.join(utre_dir,subjects))
    
        for subjects in train_images:
            folder_path = os.path.join(utre_dir, subjects)
            files = os.listdir(os.path.join(utre_dir, subjects))
            for file in files:
                if (file == 'pre'):
                    image_path = os.path.join(folder_path, file)
                    flairImage = self.utrecht_singapore_preprocessing_flair(os.path.join(image_path, 'FLAIR.nii.gz'))
                    flairImage = flairImage.get_fdata()[..., np.newaxis]
                    T1Image = self.utrecht_singapore_preprocessing_t1(os.path.join(image_path, 'T1.nii.gz'))
                    T1Image = T1Image.get_fdata()[..., np.newaxis]
    
                    imgs_two_channels = np.concatenate((flairImage, T1Image), axis=3)
                    imgs_two_channels = nib.Nifti1Image(imgs_two_channels, affine=None, header=None)
    
                if (file == 'wmh.nii.gz'):
                    SEG_image = self.utrecht_singapore_out_preprocessing(os.path.join(folder_path, file))

            self.getAxialSlices(imgs_two_channels, SEG_image, aug)
    
        return self.data_2Dslices,test_imgs
    
    
    
    
    
