import nibabel as nib
import numpy as np
import scipy.ndimage
import os
from PIL import Image , ImageOps

class normalization():
        normalized_images = []

        def normalize(self,input_path,gt_path):
            image = nib.load(input_path).get_fdata()
            image = np.array(image)

            image -= np.mean(image)
            image /= np.std(image)

            gt = nib.load(gt_path).get_fdata()
            gt = np.array(gt)

            self.normalized_images.append([image,gt])

        def call_normalize(self, paths):
            for gt_images in paths:
                input_image = gt_images.replace('groundtruth\\','')
                input_image = input_image.replace('_gt','')
                try:
                    self.normalize(input_image,gt_images)
                except:
                    print('Invalid filename is excluded')
            return self.normalized_images