import nibabel as nib
import torch
import numpy as np
import model_discrimination
from tqdm import tqdm
from torch.autograd import Variable
import os
import itk
import model_screening

def cut_3d_image(index_x, index_y, index_z, size_x, size_y, size_z, x_stride, y_stride, z_stride, img, image):
    stop = False
    x_offset = img.shape[0] - x_shape
    y_offset = img.shape[1] - y_shape
    z_offset = img.shape[2] - z_shape

    patches = []

    while (stop == False):

        if (index_x <= x_offset and index_y <= y_offset and index_z <= z_offset):
            cropper = itk.ExtractImageFilter.New(Input=image)
            cropper.SetDirectionCollapseToIdentity()
            extraction_region = cropper.GetExtractionRegion()

            size = extraction_region.GetSize()
            size[0] = int(size_x)
            size[1] = int(size_y)
            size[2] = int(size_z)

            index = extraction_region.GetIndex()
            index[0] = int(index_x)
            index[1] = int(index_y)
            index[2] = int(index_z)

            extraction_region.SetSize(size)
            extraction_region.SetIndex(index)
            cropper.SetExtractionRegion(extraction_region)
            patch = itk.GetArrayFromImage(cropper)
            patch = np.expand_dims(patch,axis=0)

            patch = np.transpose(patch,(0,3,2,1))

            patch = torch.from_numpy(patch).type(torch.FloatTensor)
            patches.append(patch)

        # now x,y,z index are 0
        # will cut one layer after another (top to bottom)
        # keeping y = 0, for every increase of x by 16 pos, iterate z-axis to 100 in installments of 10
        # i.e when z becomes 100, increment x by 16 and put back z as 0
        # if x reaches 512, then set x = 0, z = 0 but increment y by 16 pos

        if index_x <= x_offset:
            if index_z < z_offset:
                index_z += z_stride

            else:
                index_x += x_stride
                index_z = 0

        else:

            if index_x > x_offset:
                if index_y > y_offset:
                    stop = True

                else:
                    index_x = 0

            if index_y < y_offset:
                index_y += 16
                index_x = 0
                index_z = 0

            else:
                stop = True

    return patches


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
    path_scr = 'checkpoints_screening_stage1\\checkpoint_20.pth.tar'
    device = torch.device('cuda')
    model_scr = model_screening.CNN()
    model_scr.to(device)
    model_scr.eval()
    state = torch.load(path_scr)

    # load params
    model_scr.load_state_dict(state['state_dict'])

    path_discrim = 'checkpoints_discrimination_stage2\\checkpoint_30.pth.tar'
    device = torch.device('cuda')
    model_discrim = model_discrimination.classifier()
    model_discrim.to(device)
    model_discrim.eval()
    state = torch.load(path_discrim)

    # load params
    model_discrim.load_state_dict(state['state_dict'])


    # set a progress bar
    pbar = tqdm(enumerate(test_screening_loader), total=len(test_screening_loader))

    count_cmb = 0
    count_candidate = 0

    for index, patch in pbar:
        patch = Variable(patch.cuda())

        pred_candidate_score = model_scr(patch)
        pred_candidate_score = torch.nn.functional.softmax(pred_candidate_score)

        pred_candidate_score = np.squeeze(pred_candidate_score)

        # print(pred_candidate_score[0].item(),pred_candidate_score[1].item())

        if (pred_candidate_score[1].item() > 0.60):

            count_candidate += 1

            pred_cmb_score = model_discrim(patch)
            pred_cmb_score = torch.nn.functional.softmax(pred_cmb_score)
            pred_cmb_score = np.squeeze(pred_cmb_score)
            if(pred_cmb_score[1].item() > pred_cmb_score[0].item()):
                count_cmb += 1

    print(count_candidate, count_cmb)


# input_path = 'P:\\CMB\\NORM_INPUT\\4710_20120510.nii.gz'
input_path = 'P:\\CMB\\NORM_INPUT\\0069_20140128.nii.gz'
input_image = nib.load(input_path).get_data().astype(np.float32).squeeze()

# input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
reader = itk.imread(input_path, itk.F)
reader.Update()
image = reader

x_shape = 16
y_shape = 16
z_shape = 10

x_stride = x_shape
y_stride = y_shape
z_stride = z_shape

index_x = 0
index_y = 0
index_z = 0

size_x = x_shape
size_y = y_shape
size_z = z_shape

patches = cut_3d_image(index_x, index_y, index_z, size_x, size_y, size_z, x_stride, y_stride,
                 z_stride, input_image, image)


ds = FullTrainningDataset(patches, 0, len(patches))
test_screening_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

call_screening_model(test_screening_loader)








