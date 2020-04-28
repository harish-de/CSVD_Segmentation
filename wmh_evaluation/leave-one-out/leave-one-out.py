import torch
import evaluation

'''
DATALOADER - BEGINS HERE
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

'''
set validation ratio & batch size
'''
validationRatio = 0.30
batch_size = 16

def create_train_val_dset(train_data):
    '''
    load the x,y preprocessed dataset
    :return: Dataloaders
    '''
    val_offset = int(len(train_data) * (1 - validationRatio))
    train_ds = FullTrainningDataset(train_data, 0, val_offset)
    val_ds = FullTrainningDataset(train_data, val_offset,len(train_data) - val_offset)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader

def create_test_dset(test_data):
    ds = FullTrainningDataset(test_data,0,len(test_data))
    test_loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    return test_loader

'''
DATALOADER - ENDS HERE
'''

'''
TRAIN NETWORK BEGINS HERE
'''

from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import torch
import unet_model
import cross_validation

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


def save_checkpoint(state, count, leaveoneout_idx):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    checkpoint_filename = 'checkpoints_' + str(leaveoneout_idx)
    if(os.path.exists(checkpoint_filename) == False):
        os.mkdir(checkpoint_filename)

    filename = checkpoint_filename+'\\checkpoint_' + str(count) + '.pth.tar'
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

def validation(val_loader, epoch_index,leaveoneout_idx):
    path = 'checkpoints_' + str(leaveoneout_idx) + '\\checkpoint_' + str(epoch_index) + '.pth.tar'
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

'''
TRAIN NETWORK ENDS HERE
'''


'''
MAIN IMPLEMENTATION BEGINS HERE
'''

from preprocessing_leaveoneout import get_slices
import os
import nibabel as nib
import numpy as np

get_slices()

'''
# CREATE x,y PAIR for training and validation
# '''
data = []
for imgs in os.listdir('out_dir_input_slices'):
    img = nib.load(os.path.join('out_dir_input_slices', imgs)).get_data()
    out_img = nib.load(os.path.join('out_dir_groudtruth_slices', imgs)).get_data()

    img = np.array(img)
    out_img = np.array(out_img)
    out_img = np.squeeze(out_img)

    img = np.transpose(img, (2, 0, 1))

    img = torch.from_numpy(img).type(torch.FloatTensor)
    out_img = torch.from_numpy(out_img).type(torch.FloatTensor)

    data.append([img, out_img])

slices = data.copy()

'''
INITIALIZE VARIABLES
'''
start_offset = 0
end_offset = 48
x = 200
y = 200
z = 48
evaluation_matrix = []
dsc_list = []
avd_list = []
recall_list = []
f1_list = []

out_images = os.listdir('dir_output')
out_images.sort()
evaluation_matrix.append(['subject','DSC','AVD','Recall','F1'])


'''
Go to loop on each subject
current subject is treated as test subject
remaining as training data
'''
for i in range(0,60):
    '''
    data is saved in order of singapore, utrecht and ge3t
    hence first 40 subjects will have 48 slices in axial
    and rest has 83
    '''
    if((i>0 and i<40)):
        start_offset += 48
        end_offset += 48
        z = 48

    if(i>=40):
        start_offset += 83
        end_offset += 83
        z = 83

    slices_copy = slices.copy()
    del slices_copy[start_offset:end_offset]

    train_data = slices_copy.copy()
    test_data = slices[start_offset:end_offset]

    '''
    start training
    num_epochs = 50, this can be changed below
    '''
    train_loader, val_loader = create_train_val_dset(train_data=train_data)

    model = unet_model.CleanU_Net().cuda()

    optimizer = optim.Adam(model.parameters(),
                          lr=2e-4)

    best_loss = 0

    # run the training loop
    num_epochs = 50

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
            }, epoch + 1, i)

            val_loss = validation(val_loader, epoch + 1,i)
            val_loss = val_loss.item()

            print(epoch + 1, curr_loss, val_loss)
            plt_loss.append([epoch + 1, curr_loss, val_loss])


    with open('plot_loss_' + str(i) + '.txt', 'w') as f:
        for item in plt_loss:
            f.write("%s\n" % item)

    '''
    training is done
    test begins
    converting test data to Tensor
    '''
    test_slices = []

    for index_td in range(0,len(test_data)):
        test_d = test_data.__getitem__(index_td)
        test_d = np.array(test_d.__getitem__(0))
        # test = np.transpose(test_d, (2, 0, 1))
        test_slice = torch.from_numpy(test_d).type(torch.FloatTensor)
        test_slices.append(test_slice)


    test_loader = create_test_dset(test_slices)

    path = 'checkpoints_' + str(i) + '\\checkpoint_' + str(2) + '.pth.tar'
    device = torch.device('cuda')
    model = unet_model.CleanU_Net()
    model.to(device)
    model.eval()
    state = torch.load(path)

    # load params
    model.load_state_dict(state['state_dict'])

    predicted_array = np.ndarray((x, y, z),dtype=np.float32)

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))

    z_count = 0

    for index, images in pbar:
        images = Variable(images.cuda())
        predicted = model(images)

        for img_idx in range(0, len(images)):
            image_to_save = np.squeeze(predicted[img_idx].cpu().data.numpy())
            predicted_array[:, :, z_count] = image_to_save
            # predicted_image = np.concatenate(predicted_image,image_to_save)
            z_count += 1

    predicted_output = nib.Nifti1Image(predicted_array, affine=None)
    # subject_name = test_subject.split('\\')

    nib.save(predicted_output,'predicted.nii')

    '''
    computes accuracy metrics on predicted and actual output
    this is done for each test subject in leave one out fashion
    and finally aggregate score is computed
    '''
    testImage, resultImage = evaluation.getImages(os.path.join('dir_output',out_images[i]), 'predicted.nii')
    dsc = evaluation.getDSC(testImage, resultImage)
    avd = evaluation.getAVD(testImage, resultImage)
    recall, f1 = evaluation.getLesionDetection(testImage, resultImage)

    os.remove('predicted.nii')

    evaluation_matrix.append([out_images[i].split('_')[0],dsc,avd,recall,f1])

    dsc_list.append(dsc)
    avd_list.append(avd)
    recall_list.append(recall)
    f1_list.append(f1)

'''
the average score is saved in this file for all test subjects
'''
file_name = 'evaluation_leavoneout.txt'

evaluation_matrix.append(
    ['avg', (sum(dsc_list) / len(dsc_list)), sum(avd_list) / len(avd_list), sum(recall_list) / len(recall_list),
     sum(f1_list) / len(f1_list)])

with open(file_name, 'w') as f:
    for item in evaluation_matrix:
        f.write("%s\n" % item)





















































