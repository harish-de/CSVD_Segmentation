import h5py
import numpy as np
import random

'''
count cmbs
'''


class splitDataset():

    def get_split(self,file):

        total_groundtruths_paths = []
        sum = 0
        xy_count = 0

        hf = h5py.File(str(file), 'r', )

        input_data = (hf.get('input_images'))
        output_data = (hf.get('output_images'))

        for x, y in zip(input_data.keys(), output_data.keys()):
            input_image = np.array(input_data.get(x)).astype(np.float32)
            gt = np.array(output_data.get(y))
            count = gt.nonzero()
            count_cmb = len(count[0])
            total_groundtruths_paths.append([input_image,gt, count_cmb])
            sum += count_cmb
            xy_count += 1
            if xy_count > 10000:
                break

        val_size = 0.14
        test_size = 0.2
        train_size = 1 - (val_size + test_size)

        train = int(train_size * sum)
        val = int(val_size * sum)

        random.shuffle(total_groundtruths_paths)

        train_data = []
        val_data = []
        test_data = []

        sum_train = 0
        sum_val = 0

        for index, (x, y, cmb_count) in enumerate(total_groundtruths_paths):
            if (sum_train <= train):
                train_data.append([x,y])
                sum_train += cmb_count
            elif (sum_val <= val):
                val_data.append([x,y])
                sum_val += cmb_count
            else:
                test_data.append([x,y])

        print(len(train_data),len(test_data),len(val_data))

        with open('test_file.txt', 'w') as f:
            for path in test_data:
                f.write("%s\n" % str(path))

        return train_data, val_data, test_data

















