import numpy as np
import nibabel as nib
import os
import random

'''
count cmbs
'''
path = 'E:\\abhivanth\\OUTPUT'

path_list = os.listdir(path)

random.shuffle(path_list)


def splitCountCmb(path_list):
    sum = 0

    subject_list = []

    for subjects in path_list:

        new_path = os.path.join(path, subjects)

        image = nib.load(new_path).get_fdata()

        image = np.asarray(image)

        z = image.nonzero()

        count = len(z[0])

        sum += count

        subject_list.append(subjects)

        if (sum >= 1000):

            break

        else:
            continue

    return subject_list


test_list = splitCountCmb(path_list)

print(test_list.__len__())

with open('test_file.txt', 'w') as f:
    for item in test_list:
        f.write("%s\n" % item)

temp_list = [element for element in path_list if element not in test_list]



val_list = splitCountCmb(temp_list)

print(val_list.__len__())

with open('val_file.txt', 'w') as f:
    for item in val_list:
        f.write("%s\n" % item)

train_list= [element for element in path_list if element not in test_list]

train_list = [element for element in train_list if element not in val_list]

print(train_list.__len__())

with open('train_file.txt', 'w') as f:
    for item in train_list:
        f.write("%s\n" % item)





