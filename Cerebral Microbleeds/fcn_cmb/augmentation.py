import numpy as np
import nibabel as nib



def utrecht_out_preprocessing(input_image):
    rows_standard = 200
    cols_standard = 200

    image = nib.load(input_image)
    header = image.header
    affine = image.affine

    FLAIR_image = nib.load(input_image).get_data()
    FLAIR_image = np.array(FLAIR_image)

    image_rows_Dataset = np.shape(FLAIR_image)[0]  # X
    image_cols_Dataset = np.shape(FLAIR_image)[1]  # Y

    FLAIR_image = FLAIR_image[
                  int(image_rows_Dataset / 2 - rows_standard / 2):int(image_rows_Dataset / 2 + rows_standard / 2),
                  int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2), :]

    FLAIR_image = nib.Nifti1Image(FLAIR_image, affine=affine, header=header)
    nib.save(FLAIR_image,  '_seg_pp.nii.gz')

def utrecht_preprocessing_flair(input_image,image_name):
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


    FLAIR_image = FLAIR_image[
                  int(image_rows_Dataset / 2 - rows_standard / 2):int(image_rows_Dataset / 2 + rows_standard / 2),
                  int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2), :]

    # brain_mask_FLAIR = brain_mask_FLAIR[
    #                    int(image_rows_Dataset / 2 - rows_standard / 2):int(image_rows_Dataset / 2 + rows_standard / 2),
    #                    int(image_cols_Dataset / 2 - cols_standard / 2):int(image_cols_Dataset / 2 + cols_standard / 2),
    #                    :]

    # FLAIR_image -= np.mean(FLAIR_image[brain_mask_FLAIR == 1])
    # FLAIR_image /= np.std(FLAIR_image[brain_mask_FLAIR == 1])

    FLAIR_image = nib.Nifti1Image(FLAIR_image, affine=affine, header=header)
    nib.save(FLAIR_image, image_name+'_seg_pp.nii.gz')
    return FLAIR_image

path = 'D:\\de\\12_credit\\crebral_microbleeds\\viz\\wmh.nii.gz'

# utrecht_preprocessing_flair(path,'cropped.nii.gz')
utrecht_out_preprocessing(path)