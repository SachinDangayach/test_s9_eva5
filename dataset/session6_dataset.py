# Module for Extract Load and Transform
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_transforms():
    """get the train and test transform"""
    train_transforms = transforms.Compose([
                                           transforms.RandomRotation((-5.0, 5.0), fill=(1,)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                           # Note the difference between (0.1307) and (0.1307,) as this is one channel image, we have added one tuple for mean and std each
                                           ])
    # Test Phase transformations
    test_transforms = transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                           ])

    return(train_transform,test_transform)

def get_datasets(train_transform,test_transform):
    """Extract and transform the data"""
    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    return(train,test)

    def get_dataloaders(train_set,test_set):
        """ Dataloader Arguments & Test/Train Dataloaders - Load part of ETL"""
    SEED = 1

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return(train_loader,test_loader)
