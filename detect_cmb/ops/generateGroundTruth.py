import scipy.io as sci
import os
import pandas as pd
import re
import nibabel as nib
import numpy as np
import numpy.linalg as npl

class generateGT():
    def createGT(self,subjects_path,csvFile):

        os.chdir(subjects_path)
        DIR = subjects_path

        df = pd.read_csv(csvFile, delimiter=",")
        df = df[df['TYPE']=='MCH']
        df_subject = pd.DataFrame(df.iloc[:, 1].copy())
        df_scan_date = pd.DataFrame(df.iloc[:, 4].copy())
        df_coordinates = pd.DataFrame(df.iloc[:, 17].copy())

        new_df = df_subject.join(df_scan_date, how ='outer')
        new_df = new_df.join(df_coordinates, how = 'outer')

        output_list = []
        gt_list = []

        for folder in os.listdir('.'):
            target = os.path.join(DIR, folder)
            os.chdir(target)
            if(os.path.exists(target+'\\groundtruth') == False):
                os.mkdir(target+'\\groundtruth')

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
                if(build_gt.max() == 1.0):
                    new_image = nib.Nifti1Image(build_gt, affine=np.eye(4))
                    nib.save(new_image, os.path.join(target, 'groundtruth', str(file + '_gt' + '.nii')))

