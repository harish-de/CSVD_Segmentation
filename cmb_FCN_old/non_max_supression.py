import nibabel as nib
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import os
import itk
import model_screening
import torchvision.ops as tops
import skimage.feature
import model_screening_copy

import scipy.ndimage




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
    # os.chdir('D:\\de\\12_credit\\crebral_microbleeds\\cmb code\\fcn_cmb')
    path_scr = 'checkpoints_screening_stage2\\checkpoint_169.pth.tar'
    # path_scr = 'checkpoint_screeningStage2_corrected\\checkpoint_53.pth.tar'
    # device = torch.device('cpu')
    model_fcn = model_screening.CNN()
    # model_scr.to(device)
    model_fcn.eval()

    state = torch.load(path_scr)
    # load params
    model_fcn.load_state_dict(state['state_dict'], strict=False)






    pbar = tqdm(enumerate(test_screening_loader), total=len(test_screening_loader))

    count_cmb = 0
    count_candidate = 0

    for index, patch in pbar:
        patch = Variable(patch)

        pred_candidate_score = model_fcn(patch)

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
#
input_path = 'E:\\abhivanth\\normalized_input\\2155_20170920.nii'

output_path = 'E:\\abhivanth\\OUTPUT\\2155_20170920.nii'

input_image = nib.load(input_path).get_fdata().astype(np.float32)

output_image = nib.load(output_path).get_fdata().astype(np.float32)

Scoremap =  np.zeros(input_image.shape)

count = np.nonzero(output_image)

list_nonzero = []

for x in range(len(count[0])):

    i = count[0][x]
    j = count[1][x]
    k = count[2][x]

    Scoremap[i][j][k] = 2




# input_image = normalize(input_path)


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

result = np.transpose(pred_map,(1,2,3,0))





# mat_supressed = np.zeros((121,121,18),dtype=np.float32)

mat_supressed = result[:, :, :, 1]

print((mat_supressed.shape))


# mat_supressed = mat_supressed <= 0.4
#
# mat_supressed = mat_supressed.astype(int)

# mat_supressed = scipy.ndimage.maximum_filter(mat_supressed,size=(11,11,9))

print(np.min(mat_supressed))
print(np.max(mat_supressed))


footprint = np.ones((11,11,9))

mat_supressed = skimage.feature.peak_local_max(mat_supressed,footprint=footprint,indices=False,threshold_abs=0.98)

# mat_supressed = scipy.ndimage.maximum_filter(mat_supressed,size=(11,11,9))

mat_supressed = np.asarray(mat_supressed,dtype=float)

print(np.min(mat_supressed))
print(np.max(mat_supressed))


# mat_supressed = np.asarray(mat_supressed,dtype=np.float32)
#
# mat_supressed = nib.Nifti1Image(mat_supressed, affine=None, header=None)
# nib.save(mat_supressed, 'image_name_max_1.nii')







D = 2

C = 6



x,y,z  = mat_supressed.shape
patches = []

count = 0
for i in range(0,x):
    for j in range(0,y):
        for k in range(0,z):
            if(mat_supressed[i][j][k] == 1):

                  x_cord = D*i + C
                  y_cord = D*j + C
                  z_cord = k*2 + 3
                  Scoremap[x_cord][y_cord][z_cord] = 1
                  count += 1
                  rect = np.copy(input_image[(x_cord - 10):(x_cord + 10), (y_cord - 10):(y_cord + 10), (z_cord - 8):(z_cord + 8)])
                  rect = nib.Nifti1Image(rect, affine=None, header=None)
                  nib.save(rect, 'seg_patches\\image_name_max' + str(i) + '.nii')
                  patches.append([rect,1.0])



print(count)

# print(patches)


# mat_supressed[i][j][k] = (D * (mat_supressed[i][j][k].item()) ) + C







# footprint = np.ones((11,11,6,1))

# result = scipy.ndimage.maximum_filter(result,footprint=footprint)



# print(mat_supressed.shape)
#
Scoremap = nib.Nifti1Image(Scoremap, affine=None, header=None)
nib.save(Scoremap, 'image_name_max_1.nii')
#
#



























