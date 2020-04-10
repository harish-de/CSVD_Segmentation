import torch
import model_discrimination
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
import random
from sklearn.externals import joblib


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

def create_dset_complete():
    ## SET VALIDATION SET SIZE & BATCH SIZE
    validationRatio = 0
    batch_size = 64

    dset_train = joblib.load('complete_dataset_16_16_10.sav')

    train_ds, val_ds = trainTestSplit(dset_train, validationRatio)

    ## USE THESE FOR TRAINING & EVALUATING MODEL
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader

def prepare_datset_with_mimics(dataloader):
    path = 'checkpoints_discrimination_stage1\\checkpoint_75.pth.tar'
    device = torch.device('cpu')
    model = model_discrimination.classifier()
    model.to(device)
    model.eval()
    state = torch.load(path)

    # load params
    model.load_state_dict(state['state_dict'])

    false_positive = []
    positive = []
    negative = []
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

        for x in range(0,len(images)):
            total += 1
            max_op = max(outputs[x])
            if(max_op == outputs[x][0]):
                pred = 0.0
            else:
                pred = 1.0

            if (labels[x] == 0.0) & (pred==1.0):
                false_positive.append([images[x],0.0])


            if (labels[x] == 0.0) & (pred==0.0):
                negative.append([images[x],0.0])

            if (labels[x] == 1.0) & (pred==1.0):
                positive.append([images[x],1.0])


    new_negative_list = random.sample(negative, len(positive))
    false_positive = random.sample(false_positive, len(positive))
    complete_dataset_stage2 = new_negative_list + positive + false_positive
    # #
    return complete_dataset_stage2

dataloader = create_dset_complete()
# test(dataloader)
complete_dataset_stage2 = prepare_datset_with_mimics(dataloader)
filename = 'data_pos_neg_fp.sav'
joblib.dump(complete_dataset_stage2, filename)