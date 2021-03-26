# License: BSD
# Author: Sasank Chilamkurthy  # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, transforms
# from torchvision import datasets, models, transforms
from metric.inception_3 import inception_v3
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from torchsummary import summary


# Data augmentation and normalization for training
# Just normalization for validation
train_data_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        # transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ])                                                          # [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

train_dir = ''  # 训练
train_image_datasets = datasets.ImageFolder(train_dir, train_data_transforms)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=32, shuffle=True, num_workers=4)
class_names = train_image_datasets.classes
train_sizes = len(train_image_datasets)
print('train images number:', train_sizes)

val_data_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        # transforms.RandomResizedCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    ])
val_dir = ''
val_image_datasets = datasets.ImageFolder(val_dir, val_data_transforms)
val_dataloaders = torch.utils.data.DataLoader(val_image_datasets, batch_size=32, shuffle=True, num_workers=4)
val_sizes = len(val_image_datasets)
print('val images number:', val_sizes)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated


# Get a batch of training data
# inputs, classes = next(iter(train_dataloaders))
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        start_t = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloaders = train_dataloaders
                dataset_sizes = train_sizes
            else:
                model.eval()   # Set model to evaluate mode
                dataloaders = val_dataloaders
                dataset_sizes = val_sizes

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders:  # dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # outputs[0]是logits, outputs[1]是aux_logits
                    # print(outputs)
                    if phase == 'train':
                        logits = outputs[0]
                    else:
                        logits = outputs[0]  # outputs

                    _, preds = torch.max(logits, 1)  # outputs[0]
                    loss = criterion(logits, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                batch_correct = torch.sum(preds == labels.data)
                running_corrects += torch.sum(preds == labels.data)  # 计算预测对的类别
                # print('{} Loss: {:.4f} Acc: {:.4f}'. format(phase, loss.item(), batch_correct.double() / 32.))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects.double() / dataset_sizes
            print('{} pred correct number: {:.1f}'.format(phase, running_corrects.double().item()))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print('{} Loss: {:.4f} Acc: {:.4f}, Time: {:.4f}'.
                  format(phase, epoch_loss, epoch_acc, time.time() - start_t))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training  # True
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft = inception_v3(pretrained=True, transform_input=False)

# 冻结某些层训练
set_grad = False
for dix, parameter in enumerate(model_ft.named_parameters()):  # 与model_ft.parameters()的参数相同 # model_ft.named_modules()
    name = parameter[0]
    param = parameter[1]
    print(dix, name)
    if name == 'fc.weight':  # 该层之后的所有层都可训练 263  7a 7b #  Mixed_7c.branch1x1.conv.weight; fc.weight
        set_grad = True
    if set_grad:
        param.requires_grad = True
    else:
        param.requires_grad = False
   # print('-' * 10)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))  # model_ft.fc是加载的inception v3模型中的层，fc相当与名，这里对该层进行了重新替换

model_ft = model_ft.to(device)
# print(model_ft)
print('-' * 30)
summary(model_ft, (3, 299, 299))

# 训练模型
criterion = nn.CrossEntropyLoss()  # 使用nn.CrossEntropyLoss会自动加上Softmax层。因此在定义网络时无需在最后加softmax激活
#                                    label是非one-hot的编码，输入是真实的类别，然后函数内部再转换成one-hot
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)  # model_ft.fc.parameters()
# optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=0.01, alpha=0.9)
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 15 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
torch.save(model_ft.state_dict(), 'birds_inception3_fc_sgd.pth')  # 只保存模型参数  birds_inception3_all.pth

