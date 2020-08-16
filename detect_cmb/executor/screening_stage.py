import torch
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from model import screening
from dataloader import create_dataloader

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


class call_screening1():

    criterion = torch.nn.CrossEntropyLoss()

    def save_checkpoint(self,state, count, filename='checkpoints\\checkpoint.pth.tar'):
        """
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
        :param state:
        :param is_best:
        :param filename:
        :return:
        """
        filename = 'checkpoint_screeningStage1_corrected\\checkpoint_' + str(count) + '.pth.tar'
        torch.save(state, filename)

    def train(self, train_loader, model, epoch, num_epochs):

        device = torch.device('cuda')
        model.to(device)
        model.train()
        losses = AverageMeter()

        optimizer = optim.SGD(model.parameters(),lr=0.03,momentum=0.9)

        # optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in pbar:

            images = images.float()
            images = Variable(images.cuda())

            images = images.unsqueeze(dim=1)

            if epoch == 0 :
                images = nn.init.normal_(images, mean=0, std=0.01)

            # compute output
            optimizer.zero_grad()

            outputs = model(images)

            labels = np.squeeze(labels)
            labels = Variable(labels.cuda())
            labels = labels.long()

            outputs = np.squeeze(outputs)
            outputs = torch.nn.functional.softmax(outputs)

            loss = torch.nn.functional.cross_entropy(outputs,labels)
            losses.update(loss.data, images.size(0))

            loss.backward()
            optimizer.step()

            pbar.set_description('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.4f/ %.4f(avg) '
                                 % (epoch + 1, num_epochs, losses.val, losses.avg))

        return losses.avg

    def validate(self, val_loader, epoch_index):
        path = 'checkpoint_screeningStage1_corrected\\checkpoint_' + str(epoch_index) + '.pth.tar'
        device = torch.device('cuda')
        model = screening.screening()
        model.to(device)
        model.eval()
        state = torch.load(path)

        # load params
        model.load_state_dict(state['state_dict'])

        losses = AverageMeter()

        optimizer = optim.SGD(model.parameters(),lr=0.03,momentum=0.9)

        # optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)

        # set a progress bar
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))


        for i, (images, labels) in pbar:
            images = Variable(images.cuda())

            images = images.unsqueeze(dim=1)

            if epoch_index == 1:
                images = nn.init.normal_(images, mean=0, std=0.01)

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

    ### MAIN PROGRAM STARTS HERE ###
    def train_ss1(self,train_balanced,valid_balanced):
        train_loader = create_dataloader.convert2dataloader().create_dset_screening_stage1(train_balanced)

        val_loader = create_dataloader.convert2dataloader().create_dset_screening_stage1(valid_balanced)


        model = screening.screening().cuda()

        optimizer = optim.SGD(model.parameters(),lr=0.03,momentum=0.9)

        best_loss = 0

        num_epochs = 200

        plt_loss = []
        plt_loss.append(['Current epoch', 'Train loss', 'Validation loss'])


        for epoch in range(0, num_epochs):

            # train for one epoch
            curr_loss = self.train(train_loader, model, epoch, num_epochs)
            curr_loss = curr_loss.item()

            if ((epoch + 1) % 1 == 0):
                self.save_checkpoint({
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_loss,
                    'optimizer': optimizer.state_dict(),
                }, epoch+1)

                val_loss = self.validate(val_loader, epoch + 1)
                val_loss = val_loss.item()

                print(epoch + 1, curr_loss, val_loss)
                plt_loss.append([epoch + 1, curr_loss, val_loss])

        with open('screening_stage1_balanced', 'w') as f:
            for item in plt_loss:
                f.write("%s\n" % item)