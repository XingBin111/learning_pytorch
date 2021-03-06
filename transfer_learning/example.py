# coding = utf-8
#__author__ = 'xingb001'
#__date__ = '18-11-15'

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

print(os.getcwd())
# os.chdir('./transfer_learning')
plt.ion()

data_transforms = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

# ImageFolder(root,transform=None,target_transform=None,loader=
# default_loader)
# root : 在指定的root路径下面寻找图片
# transform: 对PIL Image进行转换操作,transform 输入是loader读取图片返回的对象
# target_transform :对label进行变换
# loader: 指定加载图片的函数，默认操作是读取PIL image对象
# 从ImageFolder中返回的图片已经经过transformer处理过了, next(iter(ImageFolder))

data_dir = './hymenoptera_data'
image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(
            data_dir,
            x),
        data_transforms[x]) for x in [
        'train',
        'val']}
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=4,
        shuffle=True,
        num_workers=4) for x in [
            'train',
        'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device('cuda:0')

# 可视化一些图片


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = inp * std + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


inputs, labels = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in labels])

# training the model


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # 先深拷贝一份当前模型的参数，后面迭代过程中若遇到更优模型则替换
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward, track history if only in train
                with torch.set_grad_enabled(phase == 'train'):      # 判断是否进行要求导
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in trianing phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistic
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct.double() / dataset_sizes[phase]

            print(
                '{} Loss: {:.4f} Acc:{:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed //
            60,
            time_elapsed %
            60))

    # load best nodel weights
    model.load_state_dict(best_model_wts)
    return model

# visualizing the model predictions


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# # tune allparamters
# model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)
#
# model_ft = model_ft.to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# # Observe that all paramters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#
# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#
# # train and evaluate
# model_ft = train_model(
#     model_ft,
#     criterion,
#     optimizer_ft,
#     exp_lr_scheduler,
#     num_epochs=25)
#
# Here, freeze all the network except the final layer. We need to set
# requires_grad == False to freeze the parameters so that the gradients
# are not computer in backward()
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only paramters of final layer are being optimized
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# train and evaluate
model_ft = train_model(
    model_conv,
    criterion,
    optimizer_conv,
    exp_lr_scheduler,
    num_epochs=25)

visualize_model(model_conv)
plt.ioff()
plt.imshow()