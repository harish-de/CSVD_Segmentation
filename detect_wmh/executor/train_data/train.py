from tqdm import tqdm
from torch.autograd import Variable
import executor.loss as loss
from model import unet_model
from dataloader import dataloader

import json
import torch
import os
import torch.optim as optim

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


class learnUnet():
    # dir_path = os.path.dirname(__file__)
    # config_path = "C:\\Users\\haris\\PycharmProjects\\detect_wmh\\executor\\configs\\config.json"
    #
    # with open(config_path) as json_file:
    #     data = json.load(json_file)
    #     parameters = data['train']
    #     learning_rate = parameters['learning_rate']
    #     num_epochs = parameters['num_epochs']

    learning_rate = 0.0002
    num_epochs = 100

    def save_checkpoint(self, state, count, filename='checkpoints\\checkpoint.pth.tar'):
        """
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
        :param state:
        :param is_best:
        :param filename:
        :return:
        """
        filename = 'checkpoints\\checkpoint_' + str(count) + '.pth.tar'
        torch.save(state, filename)

    def train_stage(self, train_loader, model, epoch, num_epochs):
        model.train()
        losses = AverageMeter()

        optimizer = optim.Adam(model.parameters(),
                               lr= self.learning_rate)

        '''
        Progress bar
        '''
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in pbar:
            # Convert torch tensor to Variable
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            # compute output
            optimizer.zero_grad()
            outputs = model(images)

            labels = labels.unsqueeze(dim=1)

            # measure loss
            losss = loss.loss().dice_coef_loss(outputs, labels)
            # losses.AverageMeter().update(losss.data, images.size(0))
            losses.update(losss.data, images.size(0))

            # compute gradient and do SGD step
            losss.backward()
            optimizer.step()

        # return avg loss over the epoch
        # return losses.AverageMeter().get_avg()
        return losses.avg

    def validation(self, val_loader, epoch_index):
        path = 'checkpoints\\checkpoint_' + str(epoch_index) + '.pth.tar'
        device = torch.device('cuda')
        model = unet_model.CleanU_Net()
        model.to(device)
        model.eval()
        state = torch.load(path)

        # load params
        model.load_state_dict(state['state_dict'])

        losses = AverageMeter()

        optimizer = optim.Adam(model.parameters(),
                               lr= self.learning_rate)

        # set a progress bar
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))

        for i, (images, labels) in pbar:
            # Convert torch tensor to Variable
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            # compute output
            optimizer.zero_grad()
            outputs = model(images)

            labels = labels.unsqueeze(dim=1)

            # measure loss
            losss = loss.loss().dice_coef_loss(outputs, labels)
            # losses.AverageMeter().update(losss.data, images.size(0))
            losses.update(losss.data, images.size(0))

            # compute gradient and do SGD step
            losss.backward()
            optimizer.step()

        # return losses.AverageMeter().get_avg()
        return losses.avg

    def train_data(self, preprocessed_data):
        train_loader, val_loader = dataloader.wmh_dataloader().create_train_val_dset(preprocessed_data)

        print('Fetched data')

        model = unet_model.CleanU_Net().cuda()

        best_loss = 0

        # run the training loop
        # num_epochs = 200

        plt_loss = []
        plt_loss.append(['Current_epoch','Train_loss','Validation_loss'])

        for epoch in range(0, self.num_epochs):
            # train for one epoch
            curr_loss = self.train_stage(train_loader, model, epoch, self.num_epochs)
            curr_loss = curr_loss.item()

            if ((epoch + 1) % 1 == 0):
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_loss,
                }, epoch + 1)

                val_loss = self.validation(val_loader, epoch + 1)
                val_loss = val_loss.item()

                print(epoch + 1, curr_loss, val_loss)
                plt_loss.append([epoch + 1,curr_loss,val_loss])

        with open('plot_loss.txt', 'w') as f:
            for item in plt_loss:
                f.write("%s\n" % item)


