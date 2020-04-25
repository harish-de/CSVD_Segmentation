from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import torch
import unet_model
import cross_validation_utr_sin

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


smooth=1.

def dice_coef_for_training(y_pred, y_true):
    '''
    :param y_pred: predicted output
    :param y_true: actual ground truth
    :return: Dice Similarity coefficient score
    '''
    y_true_f = y_true.contiguous().view(-1)
    y_pred_f = y_pred.contiguous().view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_coef_loss(y_pred,y_true):
    '''
    :return: return dice loss score
    '''
    return 1.-dice_coef_for_training(y_pred, y_true )


def save_checkpoint(state, count, filename='checkpoints\\checkpoint.pth.tar'):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = 'checkpoints_utr_sin\\checkpoint_' + str(count) + '.pth.tar'
    torch.save(state, filename)


def train(train_loader, model, epoch, num_epochs):
    model.train()
    losses = AverageMeter()

    optimizer = optim.Adam(model.parameters(),
                          lr=2e-4)

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
        losss = dice_coef_loss(outputs, labels)
        losses.update(losss.data, images.size(0))

        # compute gradient and do SGD step
        losss.backward()
        optimizer.step()


    # return avg loss over the epoch
    return losses.avg

def validation(val_loader, epoch_index):
    path = 'checkpoints_utr_sin\\checkpoint_' + str(epoch_index) + '.pth.tar'
    device = torch.device('cuda')
    model = unet_model.CleanU_Net()
    model.to(device)
    model.eval()
    state = torch.load(path)

    # load params
    model.load_state_dict(state['state_dict'])


    losses = AverageMeter()

    optimizer = optim.Adam(model.parameters(),
                          lr=2e-4)

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
        losss = dice_coef_loss(outputs, labels)
        losses.update(losss.data, images.size(0))

        # compute gradient and do SGD step
        losss.backward()
        optimizer.step()

    return losses.avg

train_loader, val_loader = cross_validation_utr_sin.create_train_val_dset()

print('Fetched data')

model = unet_model.CleanU_Net().cuda()

optimizer = optim.Adam(model.parameters(),
                      lr=2e-4)

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

        val_loss = validation(val_loader, epoch + 1)
        val_loss = val_loss.item()

        print(epoch + 1, curr_loss, val_loss)
        plt_loss.append([epoch + 1, curr_loss, val_loss])

with open('plot_loss_utr_sin.txt', 'w') as f:
    for item in plt_loss:
        f.write("%s\n" % item)