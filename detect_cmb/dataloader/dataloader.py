import numpy as np
import nibabel as nib
import os
import random

'''
count cmbs
'''
class splitDataset():

    def get_split(self,input_dir):

        total_groundtruths_paths = []
        sum = 0

        for files in os.listdir(input_dir):
            gt_home = os.path.join(os.path.join(input_dir,files),'groundtruth')
            for gt_images in os.listdir(gt_home):
                gt = nib.load(os.path.join(gt_home,gt_images)).get_fdata()
                gt = np.array(gt)
                count = gt.nonzero()
                count_cmb = len(count[0])
                total_groundtruths_paths.append([os.path.join(gt_home,gt_images),count_cmb])
                sum += count_cmb

        val_size = 0.14
        test_size = 0.2
        train_size = 1 - (val_size+test_size)

        train = int(train_size*sum)
        val = int(val_size*sum)

        random.shuffle(total_groundtruths_paths)

        train_data = []
        val_data = []
        test_data = []

        sum_train = 0
        sum_val = 0

        for index, (path, cmb_count) in enumerate(total_groundtruths_paths):
            if(sum_train <= train):
                train_data.append(path)
                sum_train += cmb_count
            elif(sum_val <= val):
                val_data.append(path)
                sum_val += cmb_count
            else:
                test_data.append(path)

        # print(len(train_data),len(test_data),len(val_data))

        with open('test_file.txt', 'w') as f:
            for path in test_data:
                f.write("%s\n" % str(path))

        return train_data,val_data,test_data
