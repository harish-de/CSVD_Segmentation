import scipy.io as sci
import os
import pandas as pd
import re
import nibabel as nib
import numpy as np
import numpy.linalg as npl

# DIR = '/Users/lokesh/Desktop/CSVD/cmb-3dcnn-data/ground_truth/'
#
# for file in os.listdir(DIR):
#
#     input = os.path.join(DIR, file)
#     x = sci.loadmat(input)
#     cen_data = x['cen']
#     gt_num_data = x['gt_num']
#     print(file)
#     print(cen_data)
#     print(gt_num_data)


os.chdir('D:\\de\\12_credit\\crebral_microbleeds\\adni\\groundtruth')
DIR = 'D:\\de\\12_credit\\crebral_microbleeds\\adni\\groundtruth'

df = pd.read_csv('D:\\de\\12_credit\\crebral_microbleeds\\adni\\groundtruth\\MAYOADIRL_MRI_MCH_08_15_19.csv', delimiter=",")
df = df[df['TYPE']=='MCH']
df_subject = pd.DataFrame(df.iloc[:, 1].copy())
df_scan_date = pd.DataFrame(df.iloc[:, 4].copy())
df_coordinates = pd.DataFrame(df.iloc[:, 17].copy())

# list_temp = []
#
# for i,j,k in df_subject, df_scan_date, df_coordinates:
#     list_temp = list.append(i + ',' + j + ','+ k)

#new_df = pd.concat([df_subject, df_scan_date, df_coordinates])
# new_df = pd.merge(df_subject, df_scan_date, on=df_subject, how='outer')
#new_df = df_subject.merge(df_scan_date, on=df_subject.index).merge(df_coordinates, on=df_subject.index)
#new_df = [df_subject, df_scan_date, df_coordinates]
new_df = df_subject.join(df_scan_date, how ='outer')
new_df = new_df.join(df_coordinates, how = 'outer')


#new_df.to_csv('/Volumes/Seagate Backup Plus Drive/Dataset-CMB/ADNI-CMBs/meta/final_output.csv', sep=',', index=False)

output_list = []
gt_list = []


for folder in os.listdir('.'):
    if folder == 'ADNI 4':
        target = os.path.join(DIR, folder)
        os.chdir(target)

        for file in os.listdir('.'):
            if (file != 'groundtruth') and (file !='groundtruth_1'):
                file = file[:-4]
                subject = file[:-9]
                subject = int(subject)
                scan_date = file[-8:]
                scan_date = int(scan_date)
                np_image = nib.load(os.path.join(target, str(file + '.nii'))).get_data().astype(np.float32).squeeze()
                build_gt = np.zeros(np_image.shape)

                for i in new_df[new_df.columns[0]]:
                    if subject == i:
                        new_df_temp = new_df[new_df['RID']==i]
                        for j in new_df[new_df_temp.columns[1]]:
                            if scan_date == j:
                                new_df_temp = new_df_temp[new_df_temp['SCANDATE'] == j]
                                count = 0


                                for k in new_df_temp[new_df_temp.columns[2]]:

                                    #my_output = (str(i) + ',' + str(j) + ',' + str(k))
                                    my_output = (str(k))

                                    x, y, z, a = re.split('\s+', my_output)
                                    img = nib.load(os.path.join(target, str(file + '.nii')))
                                    x1, y1, z1 = nib.affines.apply_affine(npl.inv(img.affine), ((int(float(x)), int(float(y)), int(float(z)))))
                                    print(x1, y1, z1)

                                    build_gt[int(x1), int(y1), int(z1)] = 1.0

                                    #output_list.append(my_output.split(","))
                                    output_list.append(my_output)

                                    gt_list.append(str(x1) +','+ str(y1) +','+ str(z1))

                                break
                        break
            new_image = nib.Nifti1Image(build_gt, affine=np.eye(4))
            nib.save(new_image, os.path.join(target, 'groundtruth', str(file + '_gt' + '.nii')))
            print(os.path.join(target, 'groundtruth', str(file + '_gt' + '.nii')))

        # np_image = nib.load(os.path.join(target, str(file + '.nii'))).get_data().astype(np.float32).squeeze()
        # build_gt = np.zeros(np_image.shape)
        # new_image = nib.Nifti1Image(build_gt, affine=np.eye(4))
        # nib.save(new_image, os.path.join(target, 'groundtruth', str(file + '_gt' + '.nii')))
        # print(os.path.join(target, 'groundtruth', str(file + '_gt' + '.nii')))

# list_df = pd.DataFrame(output_list)
# list_df_gt = pd.DataFrame(gt_list)
# list_df.to_csv('/Volumes/Seagate Backup Plus Drive/Dataset-CMB/ADNI-CMBs/meta/final_latest_output.csv', sep=',', index=False)
# list_df_gt.to_csv('/Volumes/Seagate Backup Plus Drive/Dataset-CMB/ADNI-CMBs/meta/gt_list.csv', sep=',', index=False)