import torch
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from model import screening
import torch.nn as nn


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


class cmb_dataloader():

    def trainTestSplit(self, dataset,val_share):
        '''
        Split the datasets into train and validation sets based on validation ratio
        '''
        val_offset = int(len(dataset) * (1 - val_share))
        return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, val_offset,
                                                                                  len(dataset) - val_offset)
    def create_train_val_dset(self, complete_data):
        '''
        load the x,y preprocessed dataset
        :return: Dataloaders
        '''

        '''
        set validation ratio & batch size
        '''
        validationRatio = 0.3
        batch_size = 16

        dset_train = complete_data
        train_ds, val_ds = self.trainTestSplit(dset_train, validationRatio)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        return train_loader, val_loader


class AverageMeter(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class train_ss2():

    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 700

    def save_checkpoint(self, state, count):
        """
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
        :param state:
        :param is_best:
        :param filename:
        :return:
        """
        filename = 'checkpoints\\ss_2\\checkpoint_' + str(count) + '.pth.tar'
        torch.save(state, filename)

    def train(self, train_loader, model, epoch, num_epochs,checkpoint_ss1):
        if(epoch == 0):
            model = screening.screening()
            device = torch.device('cuda')
            model.to(device)
            checkpoint = torch.load(checkpoint_ss1,map_location='cuda')
            model.load_state_dict(checkpoint['state_dict'])

        device = torch.device('cuda')
        model.to(device)
        model.train()
        losses = AverageMeter()

        optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in pbar:
            images = images.float()
            images = Variable(images.cuda())

            # images = images.unsqueeze(dim=1)

            # compute output
            optimizer.zero_grad()
            if epoch == 0:
                images = nn.init.normal_(images, mean=0, std=0.01)

            outputs = model(images)

            labels = np.squeeze(labels)
            labels = Variable(labels.cuda())
            labels = labels.long()

            outputs = np.squeeze(outputs)
            outputs = torch.nn.functional.softmax(outputs)

            loss = torch.nn.functional.cross_entropy(outputs, labels)
            losses.update(loss.data, images.size(0))

            loss.backward()
            optimizer.step()

            pbar.set_description('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.4f/ %.4f(avg) '
                                 % (epoch + 1, num_epochs, losses.val, losses.avg))
        return losses.avg


    def validate(self, val_loader, epoch_index):
        path = 'checkpoints\\ss_2\\checkpoint_' + str(epoch_index) + '.pth.tar'
        device = torch.device('cuda')
        model = screening.screening()
        model.to(device)
        model.eval()
        state = torch.load(path)

        # load params
        model.load_state_dict(state['state_dict'])

        losses = AverageMeter()

        optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

        # set a progress bar
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))

        for i, (images, labels) in pbar:
            images = images.float()
            images = Variable(images.cuda())

            # images = images.unsqueeze(dim=1)

            # compute output
            optimizer.zero_grad()

            outputs = model(images)

            labels = np.squeeze(labels)
            labels = Variable(labels.cuda())
            labels = labels.long()

            outputs = np.squeeze(outputs)
            outputs = torch.nn.functional.softmax(outputs)

            loss = torch.nn.functional.cross_entropy(outputs, labels)
            losses.update(loss.data, images.size(0))

            loss.backward()
            optimizer.step()

            pbar.set_description('[validate] - BATCH LOSS: %.4f/ %.4f(avg) '
                                 % (losses.val, losses.avg))

        return losses.avg

    def call_train_ss2(self, complete_data, checkpoint_ss1):

        train_loader, val_loader = cmb_dataloader().create_train_val_dset(complete_data)

        model = screening.screening().cuda()

        optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

        best_loss = 0

        plt_loss = []
        plt_loss.append(['Current epoch', 'Train loss', 'Validation loss'])

        for epoch in range(0, self.num_epochs):

            # train for one epoch
            curr_loss = self.train(train_loader, model, epoch, self.num_epochs,checkpoint_ss1)
            curr_loss = curr_loss.item()

            if ((epoch + 1) % 1 == 0):
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_loss,
                    'optimizer': optimizer.state_dict(),
                }, epoch + 1)

                val_loss = self.validate(val_loader, epoch + 1)
                val_loss = val_loss.item()

                print(epoch + 1, curr_loss, val_loss)
                plt_loss.append([epoch + 1, curr_loss, val_loss])

        with open('screening_stage2.txt', 'w') as f:
            for item in plt_loss:
                f.write("%s\n" % item)