import torch

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

class convert2dataloader():

    batch_size = 128

    def trainTestSplit(self, dataset, val_share):
        '''
        :param dataset: Complete dataset in X,y pair after formatting & augmenting
        :param val_share: Validation dataset size
        :return: Train and test datasets
        '''
        val_offset = int(len(dataset) * (1 - val_share))

        return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, val_offset,
                                                                                  len(dataset) - val_offset)
    def create_dset_screening_stage1(self,dset):

        ## USE THESE FOR TRAINING & EVALUATING MODEL
        dloader = torch.utils.data.DataLoader(dset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return dloader

