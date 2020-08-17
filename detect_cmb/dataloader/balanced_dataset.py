import numpy as np
import random
import itk
import torch

import scipy.ndimage as sind

'''
This file pairs groundTruth with corresponding image patch(X,Y - Pairs) and saves it as .sav file.
'''

# To convert the Input patches and Output patches to a '.sav ' format as x,y pair
# Equal no of input and output patches

def balanced_dataset(subjects):

    positive_list = []
    negative_list = []

    for count, (x, y) in enumerate(subjects):
        x = itk.GetArrayFromImage(x)
        y = itk.GetArrayFromImage(y)

        x = np.array(x)
        y = np.array(y)

        if y.max() == 1.0:

            # input = torch.from_numpy(x).type(torch.FloatTensor)
            positive_list.append((x, 1.0))

        else:
            negative_list.append((x, 0.0))

    positive_count = len(positive_list)
    negative_list_1 = random.sample(negative_list, positive_count)

    balanced_list = positive_list + negative_list_1

    random.shuffle(balanced_list)
    return balanced_list

