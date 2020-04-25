import torch
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

'''
set validation ratio & batch size
'''
validationRatio = 0.30
batch_size = 16

def trainTestSplit(dataset, val_share=validationRatio):
    '''
    Split the datasets into train and validation sets based on validation ratio
    '''
    val_offset = int(len(dataset) * (1 - val_share))
    return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, val_offset,
                                                                              len(dataset) - val_offset)
def create_train_val_dset():
    '''
    load the x,y preprocessed dataset
    :return: Dataloaders
    '''
    dset_train = joblib.load('data_tensor.sav')
    train_ds, val_ds = trainTestSplit(dset_train)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader

def create_test_dset(test_data):
    ds = FullTrainningDataset(test_data,0,len(test_data))
    test_loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    return test_loader
