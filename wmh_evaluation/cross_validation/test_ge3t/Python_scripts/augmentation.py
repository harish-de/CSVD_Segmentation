import nibabel as nib
import numpy as np
import os
import scipy.ndimage as sind


utrecht_dir = 'P:\\Clean_Datasets_Backup\\WMH\\Utrecht\\Utrecht'
singapore_dir = 'P:\\Clean_Datasets_Backup\\WMH\\Singapore\\Singapore'
ge3t_dir = 'P:\\Clean_Datasets_Backup\\WMH\\Amsterdam_GE3T\\GE3T'

HOME_DIR = [utrecht_dir, singapore_dir, ge3t_dir]

def rotate(hospital_name, subject_id, path, angle, img_typ):

    #angle = random.randint(-15,15)
    image = nib.load(path)

    header = image.header
    affine = image.affine

    image = nib.load(path).get_fdata()

    image = np.array(image)

    image = sind.rotate(image, angle, reshape=False)

    image = nib.Nifti1Image(image, affine=affine, header=header)

    if angle == -12:
        folder_name = str(subject_id) + '_rot1'
    else:
        folder_name = str(subject_id) + '_rot2'

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if img_typ == 't1':
        nib.save(image,hospital_name + folder_name+'\\t1.nii.gz')
    elif img_typ == 'flair':
        nib.save(image, hospital_name + folder_name+'\\flair.nii.gz')
    else:
        nib.save(image, hospital_name + folder_name+'\\wmh.nii.gz')

for no_of_aug in range(0,2):

    if no_of_aug == 0:
        angle = -12
    else:
        angle = 12


    for home in HOME_DIR:

        if ('Utrecht' in home):
            main_folder = 'utrecht\\'
        elif ('Singapore' in home):
            main_folder = 'singapore\\'
        else:
            main_folder = 'ge3t\\'

        for subjects in os.listdir(home):
            folder_path = os.path.join(home, subjects)
            files = os.listdir(os.path.join(home, subjects))
            for file in files:
                if (file == 'pre'):
                    image_path = os.path.join(folder_path, file)
                    images = os.listdir(os.path.join(folder_path, file))
                    for image in images:
                        if image == 'T1.nii.gz':
                            rotate(main_folder, subjects, (os.path.join(image_path, image)), angle, 't1')
                        elif image == 'FLAIR.nii.gz':
                            rotate(main_folder, subjects, (os.path.join(image_path, image)), angle, 'flair')

                elif(file == 'wmh.nii.gz'):
                    rotate(main_folder, subjects, (os.path.join(folder_path,file)), angle, 'seg')