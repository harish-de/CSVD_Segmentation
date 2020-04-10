import scipy.ndimage as sind
import numpy as np
import nibabel as nib
from PIL import Image



path = 'D:\\de\\12_credit\\crebral_microbleeds\\viz\\wmh.nii.gz'
path2 = 'D:\\de\\12_credit\\crebral_microbleeds\\viz\\49_predicted.nii'


image = nib.load(path)

header = image.header
affine = image.affine

image_actual = nib.load(path2).get_data()

image_actual = np.asarray(image_actual)

image_actual = nib.Nifti1Image(image_actual,affine=affine,header=header)

nib.save(image_actual,'seg_predicted.nii.gz')