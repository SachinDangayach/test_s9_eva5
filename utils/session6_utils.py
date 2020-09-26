# Utility module 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

import matplotlib.pyplot as plt

def show_sample_images(data_loader, classes, mean=.5, std=.5, num_of_images = 10, is_norm = True):
    """ Display images from a given batch of images """
    smpl = iter(data_loader)
    im,lb = next(smpl)
    plt.figure(figsize=(20,20))
    if num_of_images > im.size()[0]:
        num = im.size()[0]
        print(f'Can display max {im.size()[0]} images')
    else:
        num = num_of_images
        print(f'Displaying {num_of_images} images')
    for i in range(num):
        if is_norm:
            img  = im[i].squeeze().permute(1,2,0)*std+mean
        plt.subplot(10,10,i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(classes[lb[i]],fontsize=15)

def valid_accuracy_loss_plots(train_losses, train_acc, test_losses, test_acc):
    """Plot validation and accuracy curves"""
    plt.style.use('ggplot')

    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def show_misclassified_images(model,classes, test_loader, num_of_images = 10):
    """ Display missclassified images """
    imgs = []
    labels = []
    preds = []

    model = model.to('cpu')

    for img, target in test_loader:
      imgs.append( img )
      labels.append( target )
      preds.append( torch.argmax(model(img), dim=1) )

    imgs = torch.cat(imgs, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)

    matches = preds.eq(labels)

    plt.figure(figsize=(20,20))
    i = 0
    num = 0
    while num <= num_of_images:
        if not matches[i]:
            img  = imgs[i].permute(1,2,0)*.25+.5
        else:
          i += 1
          continue
        plt.subplot(10,10,num+1)
        plt.tight_layout()
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Actual: {classes[labels[i]]} \n Predicted: {classes[preds[i]]}' ,fontsize=10)
        i += 1
        num += 1
        if i >= imgs.shape[0]:
            break
