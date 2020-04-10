import torch
from sklearn.externals import joblib
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
import model_screening

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
def create_dset_screening_stage1():
    ## SET VALIDATION SET SIZE & BATCH SIZE
    validationRatio = 0.30
    # TestRatio = 0.10
    batch_size = 100

    # dset_train = joblib.load('balanced_dataset.sav')
    dset_train = joblib.load('positive.sav')



    train_ds, val_ds = trainTestSplit(dset_train,validationRatio)

    ## USE THESE FOR TRAINING & EVALUATING MODEL
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    # test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader,val_loader


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


def save_checkpoint(state, count, filename='checkpoints\\checkpoint.pth.tar'):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = 'checkpoints_screening_stage1\\checkpoint_' + str(count) + '.pth.tar'
    torch.save(state, filename)

criterion = torch.nn.CrossEntropyLoss()

def train(train_loader, model, epoch, num_epochs):
    model.train()
    losses = AverageMeter()

    # optimizer = optim.SGD(model.parameters(),lr=0.03,momentum=0.9)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (images, labels) in pbar:
        images = Variable(images)

        images = images.unsqueeze(dim=1)

        # compute output
        optimizer.zero_grad()
        outputs = model(images)

        labels = np.squeeze(labels)
        labels = Variable(labels)
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

def validate(val_loader, epoch_index):
    path = 'checkpoints_screening_stage1\\checkpoint_' + str(epoch_index) + '.pth.tar'
    # device = torch.device
    model = model_screening.CNN()
    # model.to(device)
    model.eval()
    state = torch.load(path)

    # load params
    model.load_state_dict(state['state_dict'])


    losses = AverageMeter()

    # optimizer = optim.SGD(model.parameters(),lr=0.03,momentum=0.9)

    optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)

    # set a progress bar
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))


    for i, (images, labels) in pbar:
        images = Variable(images)

        images = images.unsqueeze(dim=1)

        # compute output
        optimizer.zero_grad()
        outputs = model(images)

        labels = np.squeeze(labels)
        labels = Variable(labels)
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

train_loader, val_loader   = create_dset_screening_stage1()

model = model_screening.CNN()

# optimizer = optim.SGD(model.parameters(),lr=0.03,momentum=0.9)

optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-06, weight_decay=0)

best_loss = 0

# run the training loop
num_epochs = 100

plt_loss = []
plt_loss.append(['Current epoch', 'Train loss', 'Validation loss'])


for epoch in range(0, num_epochs):
    # train for one epoch
    curr_loss = train(train_loader, model, epoch, num_epochs)
    curr_loss = curr_loss.item()

    if ((epoch + 1) % 1 == 0):
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_loss,
            'optimizer': optimizer.state_dict(),
        }, epoch + 1)

        val_loss = validate(val_loader, epoch + 1)
        val_loss = val_loss.item()



        print(epoch + 1, curr_loss, val_loss)
        plt_loss.append([epoch + 1, curr_loss, val_loss])

with open('screening_stage1_plot_loss.txt', 'w') as f:
    for item in plt_loss:
        f.write("%s\n" % item)