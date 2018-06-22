from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import numpy.matlib
import torchvision
import my_model
import my_dataset
import my_loss
from my_transform import MyTransform
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import time
from datetime import datetime  
import os
import copy



# train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, img_name = data
                # get bounding box ground truth
                xml_name = []
                xml_gt = []
                size_gt = []
                for name in img_name:
                    xml_name.append(name.split('.')[0] + '.xml')
                for file in xml_name:
                    lower_prefix = file.rsplit('_', 1)[0]
                    file_path = os.path.join('E:\\Dataset\\ILSVRC2012_Task3\\annotations',
                                       phase, lower_prefix, file) 
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    temp = []
                    for i in range(4):
                        temp.append(root[5][4][i].text)
                        
                    xml_gt.append(temp)
                    size_gt.append([root[3][0].text, root[3][1].text])

                # normalize bounding box data to [0, 1]
                xml_gt_array = np.asarray(xml_gt, dtype=np.float32)
                size_gt_array = np.asarray(size_gt, dtype=np.float32)
                size_gt_array = np.matlib.repmat(size_gt_array, 1, 2)
                bbox_gt_array_normal = np.divide(xml_gt_array, size_gt_array)
                bbox_gt_array_normal -= 0.5
                bbox_gt_array_normal *= 2
                bbox_gt_normal = torch.from_numpy(bbox_gt_array_normal)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    bbox_gt_normal = Variable(bbox_gt_normal.cuda())                    
                else:
                    inputs, labels, bbox_gt_normal = Variable(inputs), Variable(labels), Variable(bbox_gt_normal)
                    # inputs, labels = Variable(inputs), Variable(labels)
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs_pair = model(inputs)
                # batch_size * num_cls
                outputs = outputs_pair[0]
                # batch_size * 4
                outputs_bbox = outputs_pair[1]


                outputs_bbox_denormal = (outputs_bbox/2)+0.5 
                test = np.multiply(outputs_bbox_denormal.cpu().data.numpy(), size_gt_array)
                print('bbox_output_large\n',test)
                print('gt_bbox_large\n', xml_gt_array)

                print('bbox_output\n', outputs_bbox)
                print('bbox__gt\n', bbox_gt_normal)

                _, preds = torch.max(outputs.data, 1)

                # calculate loss(crossentropy loss for class score, L2 loss for bounding box)
                loss_class = criterion(outputs, labels)
                loss_bbox = my_loss.L2_loss(outputs_bbox.data, bbox_gt_normal.data)

                loss = loss_bbox + loss_class
                # loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# main function
if __name__ == '__main__':
    plt.ion()   # interactive mode

    # load data
    data_dir = 'E:\\Dataset\\ILSVRC2012_Task3\\images'
    # data_dir = 'C:\\Users\\Zhiyang\\Desktop\\Semantic-segmentation\\classdification\\dataset_catdog_10\\images'
    image_datasets = {x: my_dataset.My_Dataset(os.path.join(data_dir, x), '.jpeg', 
                                         MyTransform.data_transforms[x])
                       for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    use_gpu = torch.cuda.is_available()

    # Fine-tune the fully connected layer
    model_ft = my_model.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 6) 

    if use_gpu:
        model_ft = model_ft.cuda()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train the model
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

    # filename = 'best_model_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pt'
    filename = 'best_model.pt'
    torch.save(model_ft, filename)




