import torch
import joblib
import json
import configs
import os

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


class wmh_dataloader():

    def trainTestSplit(self, dataset,val_share):
        '''
        Split the datasets into train and validation sets based on validation ratio
        '''
        val_offset = int(len(dataset) * (1 - val_share))
        return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, val_offset,
                                                                                  len(dataset) - val_offset)
    def create_train_val_dset(self, preprocessed_data):
        '''
        load the x,y preprocessed dataset
        :return: Dataloaders
        '''

        '''
        set validation ratio & batch size
        '''
        # dir_path = os.path.dirname(__file__)
        # config_path = os.path.relpath('..\\configs\\config.json', dir_path)
        # with open(config_path) as json_file:
        #     data = json.load(json_file)
        #     loader = data['data_loader']
        #     validationRatio = loader['validation_set']
        #     batch_size = loader['batch_size']
        validationRatio = 0.3
        batch_size = 16

        # dset_train = joblib.load('data_tensor.sav')
        dset_train = preprocessed_data
        train_ds, val_ds = self.trainTestSplit(dset_train, validationRatio)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        return train_loader, val_loader

    def create_test_dset(self, test_data):
        ds = FullTrainningDataset(test_data,0,len(test_data))
        test_loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        return test_loader
