from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
# import torchvision
from torchvision import transforms, datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import copy
from Custom import models, evaluation, figure, model_sample


# My_Folder,


# train the model
def train_model(model, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_result = [0.0, 0.0]
    rec_loss = []
    rec_acc = []
    rec_prc = []
    rec_rec = []
    rec_f1 = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            class_len = len(class_names)
            # Table for evaluation
            table = [[0 for i in range(class_len)] for i in range(class_len)]

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                # Update table
                for i in range(inputs.size(0)):
                    table[labels.data[i]][preds[i]] += 1

            [epoch_loss, epoch_acc, epoch_prc, epoch_rec, epoch_f1] = evaluation.evaluation(
                datasets_sizes, running_loss, phase, table)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_result[1]:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_result = [epoch_loss, epoch_acc, epoch_prc, epoch_rec, epoch_f1]

            # recording
            rec_loss += [epoch_loss]
            rec_acc += [epoch_acc]
            rec_prc += [epoch_prc]
            rec_rec += [epoch_rec]
            rec_f1 += [epoch_f1]

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_result[0]))
    for it in range(len(class_names)):
        print('{}: Precision = {:.4f}, Recall = {:.4f}, F1 measure = {:.4f}'.format(
            class_names[it], best_result[2][it], best_result[3][it], best_result[4][it]))

    loss_train = rec_loss[::2]
    loss_val = rec_loss[1::2]
    acc_train = rec_acc[::2]
    acc_val = rec_acc[1::2]
    prc_train = rec_prc[::2]
    prc_val = rec_prc[1::2]
    rec_train = rec_rec[::2]
    rec_val = rec_rec[1::2]
    f1_train = rec_f1[::2]
    f1_val = rec_f1[1::2]

    # plot figure
    figure.plot_figure('train', loss_train, acc_train, prc_train, rec_train, f1_train, result_dir, class_names)
    figure.plot_figure('val', loss_val, acc_val, prc_val, rec_val, f1_val, result_dir, class_names)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# main function
if __name__ == '__main__':
    plt.ion()  # interactive mode on

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = 'dataset-resized'
    # data_dir = 'dataset-large'
    # data_dir = 'dataset-debug'

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    # image_datasets = {x: My_Folder.My_Dataset(os.path.join(data_dir, x), '.jpg',
    #                                           data_transforms[x])
    #                   for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    use_gpu = torch.cuda.is_available()

    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 6)

    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, [55, 75], gamma=0.5, last_epoch=-1)

    '''
    Result recording
    '''
    result_dir = 'result/parameter'
    # folder
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # image sample display

    model_ft = train_model(model_ft, optimizer_ft, exp_lr_scheduler, num_epochs=120)

    model_sample.visualization(model_ft, dataloaders, use_gpu, class_names, result_dir, pic_num=10)