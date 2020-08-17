from ops import dicom_to_nifti
from ops import generateGroundTruth
from ops import normalize_images
from ops import extract_mimics
from ops import score_map
from dataloader import dataloader
from dataloader import generate_patches_screening
from dataloader import balanced_dataset
from executor import screening_stage,screening_stage2

if __name__ == '__main__':
    choice = input('Which step to execute \n'
                         '1. Create Nifti input and groundtruth images from Dicom files  \n'
                         '2. Execute screening stage-1 \n'
                         '3. Execute screening stage-2 \n'
                         '4. Get score map')

    if(choice == '1'):
        dicom_dir = input('Enter the dicom file location')
        nifti_dir = input('Enter the file location to save Nifti files')
        csv_file = input('Enter the path of meta data csv file containing ground truth coordinates')

        dicom_to_nifti.prepareNiftiFromDicom().createNifti(dicom_dir, nifti_dir)
        generateGroundTruth.generateGT().createGT(nifti_dir, csv_file)

    elif(choice == '2'):
        nifti_dir = input('Enter the file location containing input and ground truth')

        #splitting dataset
        train, val, test = dataloader.splitDataset().get_split(nifti_dir)

        #normalizing training dataset & batch generation
        train_images = normalize_images.normalization().call_normalize(train)
        train_patches = generate_patches_screening.patch_generation().create_3dpatches(train_images)

        #normalizing validation dataset & batch generation
        val_images = normalize_images.normalization().call_normalize(val)
        val_patches = generate_patches_screening.patch_generation().create_3dpatches(val_images)

        #balanced dataset
        train_balanced = balanced_dataset.balanced_dataset(train_patches)
        valid_balanced = balanced_dataset.balanced_dataset(val_patches)

        #screening stage
        screening_stage.call_screening1().train_ss1(train_balanced, valid_balanced)

    elif(choice == '3'):
        nifti_dir = input('Enter the file location containing input and ground truth')
        checkpoint_ss1 = input('Enter the checkpoint location from screening stage-1')

        #splitting dataset
        train, val, test = dataloader.splitDataset().get_split(nifti_dir)

        #normalizing training dataset & batch generation
        train_images = normalize_images.normalization().call_normalize(train)
        train_patches = generate_patches_screening.patch_generation().create_3dpatches(train_images)

        #balanced dataset
        train_balanced = balanced_dataset.balanced_dataset(train_patches)

        #extract mimics
        complete_data = extract_mimics.prepare_datset_with_mimics(train_balanced, checkpoint_ss1)

        #screening stage2
        screening_stage2.train_ss2().call_train_ss2(complete_data, checkpoint_ss1)


    elif(choice == '4'):
        testImgPath = input('Enter the test image location')
        checkpoint_ss2 = input('Enter the checkpoint location from screening stage-2')

        score_map.scoreMap().call_scoreMap(testImgPath,checkpoint_ss2)

    else:
        print('Please enter a valid choice 1,2,3 or 4')




