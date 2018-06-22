import torch
from torch.autograd import Variable
import torchvision

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import xml.etree.ElementTree as ET
import numpy as np
import numpy.matlib
import os
import my_dataset
from my_transform import MyTransform


# show the image
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# visualize model
if __name__ == '__main__':

    data_dir = 'dataset-catdog'

    image_datasets = {x: my_dataset.My_Dataset(os.path.join(data_dir, x), '.jpg', 
                                         MyTransform.data_transforms[x])
                       for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    use_gpu = torch.cuda.is_available()
    num_images = 8
    model = torch.load('best_model.pt')
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        if i > 1:
            break

        # get the inputs
        inputs, labels, img_name = data
        # get bounding box ground truth
        xml_name = []
        xml_gt = []
        size_gt = []
        for name in img_name:
            xml_name.append(name.split('.')[0] + '.xml')
        for file in xml_name:
            lower_prefix = file.split('_')[0].lower()
            file_path = os.path.join('C:\\Users\\Zhiyang\\Desktop\\Semantic-segmentation\\classdification\\dataset-catdog\\annotation',
                                'val', lower_prefix, file) 
            tree = ET.parse(file_path)
            root = tree.getroot()
            temp = []
            for para in root[5][4].itertext():
                temp.append(para)
            xml_gt.append(temp)
            # for para in root[3].itertext():
            size_gt.append([root[3][0].text, root[3][1].text])

        xml_gt_array = np.asarray(xml_gt, dtype=np.float32)
        size_gt_array = np.asarray(size_gt, dtype=np.float32)
        size_gt_array = np.matlib.repmat(size_gt_array, 1, 2)
        # bbox_gt_array_normal = np.multiply(xml_gt_array, size_gt_array)

        # bbox_gt_normal = torch.from_numpy(bbox_gt_array_normal)


        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # forward
        outputs_pair = model(inputs)
        # batch_size * num_cls
        outputs = outputs_pair[0]
        # batch_size * 4
        outputs_bbox = outputs_pair[1]
        # print(outputs_bbox)
        _, preds = torch.max(outputs.data, 1)

        outputs_bbox_unnormal = np.multiply(outputs_bbox.cpu().data.numpy(), size_gt_array)
        
        print('bbox_gt', xml_gt_array)
        print('bbox_output', outputs_bbox_unnormal)
        
        for j in range(inputs.size()[0]):
            images_so_far += 1
            print('imge', images_so_far)
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                # return
    show()
    # plt.pause(10)  # pause a bit so that plots are updated
            
    model.train(mode=was_training)