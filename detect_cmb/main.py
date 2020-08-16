from ops import dicom_to_nifti
from ops import generateGroundTruth
from ops import normalize_images
from dataloader import dataloader
from dataloader import generate_patches
from dataloader import balanced_dataset
from executor import screening_stage

def execute(inputIsDicom):
    if (inputIsDicom == 'y'):
        # dicom_dir = input('Enter the dicom file location')
        # nifti_dir = input('Enter the nifti file location')
        # csv_file = input('Enter the path of csv file containing ground truth coordinates')

        nifti_dir = 'E:\\abhivanth\\cmb'
        csv_file = 'F:\Dataset-CMB\ADNI-CMBs\meta\MAYOADIRL_MRI_MCH_08_15_19.csv'

        # dicom_to_nifti.prepareNiftiFromDicom().createNifti(dicom_dir, nifti_dir)
        # generateGroundTruth.generateGT().createGT(nifti_dir, csv_file)

        train, val, test = dataloader.splitDataset().get_split(nifti_dir)

        train_images = normalize_images.normalization().call_normalize(train)
        train_patches = generate_patches.patch_generation().create_3dpatches(train_images)

        val_images = normalize_images.normalization().call_normalize(val)
        val_patches = generate_patches.patch_generation().create_3dpatches(val_images)

        test_images = normalize_images.normalization().call_normalize(test)
        test_patches = generate_patches.patch_generation().create_3dpatches(test_images)

        train_balanced = balanced_dataset.balanced_dataset(train_patches)
        valid_balanced = balanced_dataset.balanced_dataset(val_patches)

        screening_stage.call_screening1().train_ss1(train_balanced,valid_balanced)






execute('y')

# if __name__ == '__main__':
    # inputIsDicom = input('Do you want to generate nifti files from dicom [y/n]')



