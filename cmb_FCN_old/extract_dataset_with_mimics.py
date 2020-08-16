import torch
import model_screening
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
import random
import joblib
from random import shuffle
import scipy.ndimage as sind

'''
This file is the second sub step of the Screening Stage, which with the help of the trained screening model 
splits the data-set into false-positive,positive, and negatives with percentages mentioned in the paper.


'''


class FullTrainningDataset(torch.utils.data.Dataset):
    '''
    Performs indexing on whole dataset to split them as train & validation datasets
    '''

    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(FullTrainningDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i + self.offset]


def trainTestSplit(dataset, val_share):
    '''
    :param dataset: Complete dataset in X,y pair after formatting & augmenting
    :param val_share: Validation dataset size
    :return: Train and test datasets
    '''
    val_offset = int(len(dataset) * (1 - val_share))
    return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, val_offset,
                                                                              len(dataset) - val_offset)


def remove_percentage(list_a, percentage):
    shuffle(list_a)
    count = int(len(list_a) * percentage)
    if not count: return []  # edge case, no elements removed
    list_a[-count:], list_b = [], list_a[-count:]
    return list_b


def create_dset_complete():
    ## SET VALIDATION SET SIZE & BATCH SIZE
    validationRatio = 0
    batch_size = 100

    # dset_train = joblib.load('balanced_dataset_16_16_10_augmented.sav')

    dset_train = joblib.load('balanced_dataset_test.sav')
    # dset_train = joblib.load('data_pos_fp_neg_new_1.sav')

    train_ds, val_ds = trainTestSplit(dset_train, validationRatio)

    ## USE THESE FOR TRAINING & EVALUATING MODEL
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader


def prepare_datset_with_mimics(dataloader):
    # path = 'checkpoints_screening_stage1\\checkpoint_188.pth.tar'
    path = 'checkpoint_screeningStage2_corrected\\checkpoint_53.pth.tar'
    # path ='checkpoints_screening_stage2\\checkpoint_223.pth.tar'
    device = torch.device('cpu')
    model = model_screening.CNN()
    model.to(device)
    model.eval()
    state = torch.load(path, map_location='cpu')

    # load params
    model.load_state_dict(state['state_dict'])

    false_positive = []  # 28.85%
    positive = []  # 23.63 %
    negative = []  # 47.52 %
    false_negative = []

    # set a progress bar
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    total = 0

    for i, (images, labels) in pbar:
        images = Variable(images)

        images = images.unsqueeze(dim=1)

        outputs = model(images)

        labels = np.squeeze(labels)
        labels = Variable(labels)
        labels = labels.long()

        outputs = np.squeeze(outputs)
        outputs = torch.nn.functional.softmax(outputs)

        for x in range(0, len(images)):
            total += 1
            max_op = max(outputs[x])
            if (max_op == outputs[x][0]):
                pred = 0.0
            else:
                pred = 1.0

            if (labels[x] == 0.0) & (pred == 1.0):
                false_positive.append([images[x], 0.0])

                # current_image = images[x]

                # convert tensor to numpy

                # current_image = np.asarray(current_image)

                # FLAIR_image_in_fliped =  (np.flip(current_image,axis=3)).copy()
                # FLAIR_image_in_fliped = torch.from_numpy(FLAIR_image_in_fliped)
                # false_positive.append([FLAIR_image_in_fliped, 0.0])
                #
                # FLAIR_image_shifted_1 = torch.from_numpy(np.roll(current_image, 10, 0)).float()
                # false_positive.append([FLAIR_image_shifted_1, 0.0])
                #
                # FLAIR_image_shifted_2 = torch.from_numpy(np.roll(current_image, -10, 0)).float()
                # false_positive.append([FLAIR_image_shifted_2, 0.0])

            if (labels[x] == 0.0) & (pred == 0.0):
                negative.append([images[x], 0.0])

            if (labels[x] == 1.0) & (pred == 1.0):
                positive.append([images[x], 1.0])

            if (labels[x] == 1.0) & (pred == 0.0):
                false_negative.append([images[x], 1.0])

    # new_negative_list = random.sample(negative, len(positive))
    # false_positive = random.sample(false_positive, len(positive))

    print(positive.__len__(), 'positive')
    print(negative.__len__(), 'negative')
    print(false_positive.__len__(), 'false_positive')
    print(false_negative.__len__(), 'false_negative')

    random.shuffle(negative)
    random.shuffle(positive)
    random.shuffle(false_positive)

    new_false_positive = remove_percentage(false_positive,29)

    # new_false_positive = false_positive
    # new_positive = random.sample(positive,875)
    # new_negative_list = random.sample(negative,1750)

    new_negative_list = remove_percentage(negative, 0.47)


    new_positive = remove_percentage(positive, 0.24)



    print('after removal')

    print(new_false_positive.__len__(), 'false_positive')
    print(new_negative_list.__len__(), 'negative')
    print(new_positive.__len__(), 'positive')

    # positive_count = len(false_positive)
    # positive_list = random.sample(positive, positive_count)

    complete_dataset_stage2 = new_false_positive + new_negative_list + new_positive

    random.shuffle(complete_dataset_stage2)

    return complete_dataset_stage2


dataloader = create_dset_complete()

complete_dataset_stage2 = prepare_datset_with_mimics(dataloader)
filename = 'data_pos_fp_neg.sav'
joblib.dump(complete_dataset_stage2, filename)
