import numpy as np
import nibabel as nib
import torch
from model import screening
from tqdm import tqdm
from torch.autograd import Variable
import skimage.feature as feature

class scoreMap():

    def call_screening_model(self, test_dataloader, checkpoint_ss2):
        model_fcn = screening.screening().cuda()
        model_fcn.eval()

        state = torch.load(checkpoint_ss2)
        # load params
        model_fcn.load_state_dict(state['state_dict'], strict=False)

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

        for index, patch in pbar:
            patch = patch.float()
            patch = Variable(patch.cuda())

            pred_candidate_score = model_fcn(patch)

            pred_candidate_score = torch.nn.functional.softmax(pred_candidate_score)

        return pred_candidate_score

    def call_scoreMap(self, test_img_path, checkpoint_ss2):
        input_image = nib.load(test_img_path).get_fdata().astype(np.float32)
        Scoremap = np.zeros(input_image.shape)

        input_image_torch = torch.from_numpy(input_image).type(torch.FloatTensor)

        input_image_torch = input_image_torch.unsqueeze(0)
        input_image_torch = input_image_torch.unsqueeze(0)

        print(input_image_torch.size())

        input_image_torch = input_image_torch[None]

        pred_map = self.call_screening_model(input_image_torch,checkpoint_ss2)

        pred_map = pred_map.squeeze(0)

        pred_map = pred_map.cpu().detach().numpy()

        result = np.transpose(pred_map, (1, 2, 3, 0))

        mat_supressed = result[:, :, :, 1]

        footprint = np.ones((11, 11, 9))

        mat_supressed = feature.peak_local_max(mat_supressed, footprint=footprint, indices=False,
                                                       threshold_abs=0.98)

        mat_supressed = np.asarray(mat_supressed,dtype=float)

        D = 2

        C = 6

        x, y, z = mat_supressed.shape
        patches = []

        count = 0
        for i in range(0, x):
            for j in range(0, y):
                for k in range(0, z):
                    if (mat_supressed[i][j][k] == 1):
                        x_cord = D * i + C
                        y_cord = D * j + C
                        z_cord = k * 2 + 3
                        Scoremap[x_cord][y_cord][z_cord] = 1
                        count += 1
                        rect = np.copy(input_image[(x_cord - 10):(x_cord + 10), (y_cord - 10):(y_cord + 10),
                                       (z_cord - 8):(z_cord + 8)])
                        rect = nib.Nifti1Image(rect, affine=None, header=None)
                        patches.append([rect, 1.0])

        Scoremap = nib.Nifti1Image(Scoremap, affine=None, header=None)
        return Scoremap
