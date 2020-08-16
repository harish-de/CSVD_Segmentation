import  numpy as np
import nibabel as nib

output_path = 'E:\\abhivanth\\OUTPUT\\2155_20170920.nii'

output_image = nib.load(output_path).get_fdata().astype(np.float32)


count = np.nonzero(output_image)

list = []

for x in range(len(count[0])):

    i = count[0][x]
    j = count[1][x]
    k = count[2][x]

    list.append([i,j,k])



