import torch
from model import screening
from dataloader import create_dataloader
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
import random


def remove_percentage(list_a, percentage):
    random.shuffle(list_a)
    count = int(len(list_a) * percentage)
    if not count: return []  # edge case, no elements removed
    list_a[-count:], list_b = [], list_a[-count:]
    return list_b

def prepare_datset_with_mimics(train,checkpoint):

    dataloader = create_dataloader.convert2dataloader().create_dset_screening_stage1(train)

    device = torch.device('cuda')
    model = screening.screening()
    model.to(device)
    model.eval()
    state = torch.load(checkpoint, map_location='cuda')

    # load params
    model.load_state_dict(state['state_dict'])

    false_positive = []  # 28.85%
    positive = []  # 23.63 %
    negative = []  # 47.52 %
    false_negative = []

    # set a progress bar
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    total = 0

    for i, (images, labels) in pbar:
        images = images.float()
        images = Variable(images.cuda())

        images = images.unsqueeze(dim=1)

        outputs = model(images)

        labels = np.squeeze(labels)
        labels = Variable(labels.cuda())
        labels = labels.long()

        outputs = np.squeeze(outputs)
        outputs = torch.nn.functional.softmax(outputs)

        for x in range(0, len(images)):
            total += 1
            max_op = max(outputs[x])
            if (max_op == outputs[x][0]):
                pred = 0.0
            else:
                pred = 1.0

            if (labels[x] == 0.0) & (pred == 1.0):
                false_positive.append([images[x], 0.0])

            if (labels[x] == 0.0) & (pred == 0.0):
                negative.append([images[x], 0.0])

            if (labels[x] == 1.0) & (pred == 1.0):
                positive.append([images[x], 1.0])

            if (labels[x] == 1.0) & (pred == 0.0):
                false_negative.append([images[x], 1.0])

    random.shuffle(negative)
    random.shuffle(positive)
    random.shuffle(false_positive)

    new_false_positive = remove_percentage(false_positive,29)

    new_negative_list = remove_percentage(negative, 0.47)

    new_positive = remove_percentage(positive, 0.24)

    complete_dataset_stage2 = new_false_positive + new_negative_list + new_positive

    random.shuffle(complete_dataset_stage2)

    return complete_dataset_stage2
