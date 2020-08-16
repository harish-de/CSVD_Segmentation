import torch
from model import screening

def prepare_datset_with_mimics(dataloader,checkpoint):

    device = torch.device('cuda')
    model = screening.CNN()
    model.to(device)
    model.eval()
    state = torch.load(path, map_location='cpu')

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
        images = Variable(images)

        images = images.unsqueeze(dim=1)

        outputs = model(images)

        labels = np.squeeze(labels)
        labels = Variable(labels)
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

                # current_image = images[x]

                # convert tensor to numpy

                # current_image = np.asarray(current_image)

                # FLAIR_image_in_fliped =  (np.flip(current_image,axis=3)).copy()
                # FLAIR_image_in_fliped = torch.from_numpy(FLAIR_image_in_fliped)
                # false_positive.append([FLAIR_image_in_fliped, 0.0])
                #
                # FLAIR_image_shifted_1 = torch.from_numpy(np.roll(current_image, 10, 0)).float()
                # false_positive.append([FLAIR_image_shifted_1, 0.0])
                #
                # FLAIR_image_shifted_2 = torch.from_numpy(np.roll(current_image, -10, 0)).float()
                # false_positive.append([FLAIR_image_shifted_2, 0.0])

            if (labels[x] == 0.0) & (pred == 0.0):
                negative.append([images[x], 0.0])

            if (labels[x] == 1.0) & (pred == 1.0):
                positive.append([images[x], 1.0])

            if (labels[x] == 1.0) & (pred == 0.0):
                false_negative.append([images[x], 1.0])

    # new_negative_list = random.sample(negative, len(positive))
    # false_positive = random.sample(false_positive, len(positive))

    print(positive.__len__(), 'positive')
    print(negative.__len__(), 'negative')
    print(false_positive.__len__(), 'false_positive')
    print(false_negative.__len__(), 'false_negative')

    random.shuffle(negative)
    random.shuffle(positive)
    random.shuffle(false_positive)

    new_false_positive = remove_percentage(false_positive,29)

    # new_false_positive = false_positive
    # new_positive = random.sample(positive,875)
    # new_negative_list = random.sample(negative,1750)

    new_negative_list = remove_percentage(negative, 0.47)


    new_positive = remove_percentage(positive, 0.24)



    print('after removal')

    print(new_false_positive.__len__(), 'false_positive')
    print(new_negative_list.__len__(), 'negative')
    print(new_positive.__len__(), 'positive')

    # positive_count = len(false_positive)
    # positive_list = random.sample(positive, positive_count)

    complete_dataset_stage2 = new_false_positive + new_negative_list + new_positive

    random.shuffle(complete_dataset_stage2)

    return complete_dataset_stage2


dataloader = create_dset_complete()

complete_dataset_stage2 = prepare_datset_with_mimics(dataloader)
filename = 'data_pos_fp_neg.sav'
joblib.dump(complete_dataset_stage2, filename)
