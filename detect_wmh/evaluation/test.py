import scipy.ndimage
import dataloader.dataloader as test_dataloader
import numpy as np
import torch
import model.unet_model as unet_model
from torch.autograd import Variable
import nibabel as nib
from tqdm import tqdm
import evaluation
import os
import evaluation.metric_results as metrics

#flair, t1 as input
# preprocess
# take slices
# dataloader without splits
# test model
# concatente predicted outputs

class test_data():
    dsc_list = []
    avd_list = []
    recall_list = []
    f1_list = []
    evaluation_matrix = []

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
        # imgs_two_channels = nib.Nifti1Image(imgs_two_channels, affine=None, header=None)
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

        imgs_two_channels = np.ndarray((rows_standard, cols_standard, num_selected_slice, channel_num), dtype=np.float32)
        SEG_image_suitable = np.ndarray((rows_standard, cols_standard, num_selected_slice), dtype=np.float32)

        SEG_image_suitable[...] = np.min(SEG_image)
        # print(cols_standard, image_rows_Dataset)
        SEG_image_suitable[int(cols_standard/2-image_rows_Dataset/2):int(cols_standard/2+image_rows_Dataset/2),:, :] = \
            SEG_image[:, start_cut:start_cut+rows_standard, :]


        # ---------------------------------------------------
        SEG_image_suitable = SEG_image_suitable[..., np.newaxis]

        # SEG_image = nib.Nifti1Image(SEG_image_suitable, affine=affine, header=header)
        return SEG_image_suitable


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
                           int(image_rows_Dataset / 2 - rows_standard / 2):int(image_rows_Dataset / 2 + rows_standard / 2),
                           int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2),
                           :]

        t1_image -= np.mean(t1_image[brain_mask_FLAIR == 1])
        t1_image /= np.std(t1_image[brain_mask_FLAIR == 1])

        t1_image = nib.Nifti1Image(t1_image, affine=affine, header=header)
        # nib.save(FLAIR_image, image_name+'_tI_pp.nii.gz')
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

        # SEG_image = nib.Nifti1Image(SEG_image, affine=affine, header=header)
        return SEG_image

    def test_evaluation(self, test_subjects, checkpoint):

        dsc_list = []
        avd_list = []
        recall_list = []
        f1_list = []
        evaluation_matrix = []

        evaluation_matrix.append(['subject', 'DSC', 'AVD', 'Recall', 'F1'])

        for test_subject in test_subjects:
            # test_subject = #'P:\\Clean_Datasets_Backup\\WMH\\Utrecht\\Utrecht\\49'
            test_subject = test_subject.rstrip()
            test_subject.replace('\n', '')
            flair_path = test_subject + '\\pre\\FLAIR.nii.gz'
            t1_path = test_subject + '\\pre\\T1.nii.gz'

            image = nib.load(flair_path)
            affine = image.affine
            header = image.header

            check_img_dim = nib.load(flair_path).get_data()
            if ((np.shape(check_img_dim)[0] > 200) & (np.shape(check_img_dim)[0] > 200)):
                t1_image = self.utrecht_singapore_preprocessing_t1(t1_path)
                flair_image = self.utrecht_singapore_preprocessing_flair(flair_path)
                flair_image = flair_image.get_fdata()[..., np.newaxis]
                t1_image = t1_image.get_fdata()[..., np.newaxis]
                imgs_two_channels = np.concatenate((flair_image, t1_image), axis=3)

                out_path = test_subject + '\\wmh.nii.gz'
                seg_image = nib.load(out_path)
                out_array = self.utrecht_singapore_out_preprocessing(out_path)
                out_image = nib.Nifti1Image(out_array, affine=seg_image.affine, header=seg_image.header)

            else:
                imgs_two_channels = self.preprocess_ge3t(flair_path, t1_path)
                flair_image = np.ndarray(
                    (np.shape(imgs_two_channels)[0], np.shape(imgs_two_channels)[1], np.shape(imgs_two_channels)[2]),
                    dtype=np.float32)

                out_path = test_subject + '\\wmh.nii.gz'
                seg_image = nib.load(out_path)
                out_array = self.preprocess_seg_ge3t(out_path)
                out_image = nib.Nifti1Image(out_array, affine=seg_image.affine, header=seg_image.header)

            slices = []

            for i in range(imgs_two_channels.shape[2]):
                slice = imgs_two_channels[:, :, i, :]  # the image is in x,y,z,c format
                slice = np.transpose(slice, (2, 0, 1))
                slice = torch.from_numpy(slice).type(torch.FloatTensor)
                slices.append(slice)

            dataloader = test_dataloader.wmh_dataloader().create_test_dset(slices)

            path = checkpoint
            device = torch.device('cuda')
            model = unet_model.CleanU_Net()
            model.to(device)
            model.eval()
            state = torch.load(path)

            # load params
            model.load_state_dict(state['state_dict'])

            # predicted_image = np.array(flair_image.shape[0],flair_image.shape[1],flair_image.shape[2])

            predicted_array = np.ndarray((np.shape(flair_image)[0], np.shape(flair_image)[1], np.shape(flair_image)[2]),
                                         dtype=np.float32)

            pbar = tqdm(enumerate(dataloader), total=len(dataloader))

            z_count = 0

            for index, images in pbar:
                images = Variable(images.cuda())
                predicted = model(images)

                for i in range(0, len(images)):
                    image_to_save = np.squeeze(predicted[i].cpu().data.numpy())
                    predicted_array[:, :, z_count] = image_to_save
                    # predicted_image = np.concatenate(predicted_image,image_to_save)
                    z_count += 1

            predicted_output = nib.Nifti1Image(predicted_array, affine=affine, header=header)
            subject_name = test_subject.split('\\')

            nib.save(predicted_output, subject_name[-1] + '_predicted.nii')
            nib.save(out_image, subject_name[-1] + '_actual.nii')

            testImage, resultImage = evaluation.metric_results.getImages(subject_name[-1] + '_actual.nii',
                                                          subject_name[-1] + '_predicted.nii')
            dsc = evaluation.metric_results.getDSC(testImage, resultImage)
            avd = evaluation.metric_results.getAVD(testImage, resultImage)
            recall, f1 = evaluation.metric_results.getLesionDetection(testImage, resultImage)

            # os.remove(subject_name[-1] + '_predicted.nii')
            # os.remove(subject_name[-1] + '_actual.nii')

            print(subject_name)

            evaluation_matrix.append([subject_name, dsc, avd, recall, f1])

            dsc_list.append(dsc)
            avd_list.append(avd)
            recall_list.append(recall)
            f1_list.append(f1)

        print('Average Dice score for held-out test set', (sum(dsc_list) / len(dsc_list)))
        print('Average volume differenc score for held-out test set', sum(avd_list) / len(avd_list))
        print('Average Recall score for held-out test set', sum(recall_list) / len(recall_list))
        print('Average F1 score for held-out test set', sum(f1_list) / len(f1_list))

        file_name = 'evaluation_heldout.txt'

        evaluation_matrix.append(
            ['avg', (sum(dsc_list) / len(dsc_list)), sum(avd_list) / len(avd_list), sum(recall_list) / len(recall_list),
             sum(f1_list) / len(f1_list)])

        with open(file_name, 'w') as f:
            for item in evaluation_matrix:
                f.write("%s\n" % item)