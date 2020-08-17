import torch
from model import screening
from dataloader import create_dataloader
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
import random
import itk

def prepare_datset_with_mimics(subjects, checkpoint):
    data = []
    for count, (x, y) in enumerate(subjects):
        x = itk.GetArrayFromImage(x)
        y = itk.GetArrayFromImage(y)

        x = np.array(x)
        y = np.array(y)

        if y.max() == 1.0:
            data.append([x,1.0])
        else:
            data.append([x,0.0])

    dataloader = create_dataloader.convert2dataloader().create_dset_screening_stage1(data)

    device = torch.device('cuda')
    model = screening.screening()
    model.to(device)
    model.eval()
    state = torch.load(checkpoint, map_location='cuda')

    # load params
    model.load_state_dict(state['state_dict'])

    false_positive = []
    positive = []

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

            if (labels[x] == 1.0) & (pred == 1.0):
                positive.append([images[x], 1.0])

    random.shuffle(positive)
    random.shuffle(false_positive)

    complete_dataset_discrimination = positive + false_positive

    random.shuffle(complete_dataset_discrimination)

    return complete_dataset_discrimination
