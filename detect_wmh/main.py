#this is the main file
# step1: call preprocessing.create_dataset(home_dir, 17) -> creates h5 file
# step2: call train.train_data() -> this in turn calls dataloader.create_train_val_dset() -> change the hardcoded joblib file to h5 file
# step3: call utils.plot_losses() -> this shows a graph of training vs validation loss
# step4: call test.test()

from dataloader.held_out import preparing_dataset
from dataloader.cross_validation import preparing_dataset as cv_dataset
from executor.held_out_dataset import train

# from utils import plot_losses
from evaluation import test

def held_out_dataset(file_location,num_images):
    preprocessed_data, test_images = preparing_dataset.preprocess().create_dataset(file_location, num_images,True)
    with open('test_list.txt', 'w') as f:
        for item in test_images:
            f.write("%s\n" % item)

    train.learnUnet().train_data(preprocessed_data)
    test.test_data().test_evaluation(test_images, 'checkpoints\\checkpoint_99.pth.tar')

def cross_validation(file_location, test_clinic_data):
    preprocessed_data, test_images = cv_dataset.preprocess().create_dataset(file_location, test_clinic_data)
    with open('test_list.txt', 'w') as f:
        for item in test_images:
            f.write("%s\n" % item)

    train.learnUnet().train_data(preprocessed_data)
    test.test_data().test_evaluation(test_images, 'checkpoints\\checkpoint_99.pth.tar')


if __name__ == '__main__':

    file_location = input('Enter the WMH file location')

    test_case = input('Which test case to execute \n'
          '1 for Held-out dataset \n'
          '2 for Cross validation')

    if(test_case == '1'):
        num_images = input('Enter the number of images to be trained per clinical dataset')
        num_images = int(num_images)
        train_data(file_location,num_images)
    elif(test_case == '2'):
        test_clinic_data = input('Enter the clinic data to be tested based on other clinical data - GE3T, SING, or UTRE')
        cross_validation(file_location, test_clinic_data)
    else:
        print('Please select a proper test case 1, or 2')

