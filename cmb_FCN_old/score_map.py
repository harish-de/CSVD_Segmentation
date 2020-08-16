import nibabel as nib
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import os
import model_screening
import skimage.feature
import scipy.ndimage


'''
This file inputs a whole volume to the screening model and the non maximum supression is done in the resulting tensor
and score mapping to original coordinates is done
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


def call_screening_model(test_screening_loader):
    os.chdir('D:\\de\\12_credit\\crebral_microbleeds\\cmb code\\cmb_FCN')
    path_scr = 'D:\\de\\12_credit\\crebral_microbleeds\\cmb code\\cmb_FCN\\checkpoint_390.pth.tar'
    # device = torch.device('cpu')
    model_scr = model_screening.CNN()
    # model_scr.to(device)
    model_scr.eval()
    state = torch.load(path_scr,map_location='cpu')

    # load params
    model_scr.load_state_dict(state['state_dict'])


    pbar = tqdm(enumerate(test_screening_loader), total=len(test_screening_loader))

    count_cmb = 0
    count_candidate = 0

    for index, patch in pbar:
        patch = Variable(patch)

        pred_candidate_score = model_scr(patch)

        pred_candidate_score = torch.nn.functional.softmax(pred_candidate_score)



    return (pred_candidate_score)


def normalize(input_image):



    image = nib.load(input_image)
    header = image.header
    affine = image.affine

    FLAIR_image = nib.load(input_image).get_fdata().astype(np.float32).squeeze()
    FLAIR_image = np.array(FLAIR_image)

    FLAIR_image -= np.min(FLAIR_image)
    FLAIR_image /= np.max(FLAIR_image)


    return FLAIR_image

input_path = 'D:\\de\\12_credit\\crebral_microbleeds\\adni\\INPUT\\0210_20161103.nii'
input_image = normalize(input_path)

Score_map = np.zeros((input_image.shape))

print(input_image.shape)

#
# input_path = 'normalised.nii'
# input_image = nib.load(input_path).get_data().astype(np.float32).squeeze()

input_image_torch = torch.from_numpy(input_image).type(torch.FloatTensor)



input_image_torch = input_image_torch.unsqueeze(0)
input_image_torch = input_image_torch.unsqueeze(0)



print(input_image_torch.size())

input_image_torch = input_image_torch[None]

pred_map = call_screening_model(input_image_torch)





pred_map = pred_map.squeeze(0)

pred_map = pred_map.detach().numpy()


mat_supressed = pred_map[1, :, :, :]

print((mat_supressed.shape))
print(np.min(mat_supressed))
print((np.max(mat_supressed)))





footprint = np.ones((11,11,9))

mat_supressed = skimage.feature.peak_local_max(mat_supressed,footprint=footprint,indices=False,threshold_abs=0.35)
# mat_supressed = scipy.ndimage.maximum_filter(mat_supressed,size=(11,11,9))

mat_supressed = mat_supressed.astype(int)

mat_supressed = np.asarray(mat_supressed,dtype=np.float32)

D = 2

C = 6

thresholded = []

x,y,z  = mat_supressed.shape



print(x,y,z)

count = 0
for i in range(0,x):
    for j in range(0,y):
        for k in range(0,z):
           if(mat_supressed[i][j][k] == 1):
               X_cord = i*D + C
               Y_cord = j*D + C
               Z_cord = k*2 + 2
               Score_map[X_cord][Y_cord][Z_cord] = 1

Score_map = nib.Nifti1Image(Score_map, affine=None, header=None)
nib.save(Score_map, 'image_name_max_1.nii')





























