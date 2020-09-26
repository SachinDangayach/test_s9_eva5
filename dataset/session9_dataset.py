# Module for Extract Load and Transform CIFAR10 data
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import albumentations as A
import albumentations.pytorch as AP
from   torch.utils.data import Dataset

class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, rimages, labels, transform=None):
        self.rimages = rimages
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.rimages)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.rimages[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

# Calculate Ture Mean and STD
def cifar10_mean_std():
    """Return the true mean of entire test and train dataset"""
    # simple transform
    simple_transforms = transforms.Compose([
                                           transforms.ToTensor(),
                                           ])
    exp_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=simple_transforms)
    exp_test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=simple_transforms)

    exp_tr_data = exp_train.data # train set
    exp_ts_data = exp_test.data # test set

    exp_data = np.concatenate((exp_tr_data,exp_ts_data),axis=0) # contatenate entire data

    exp_data = np.transpose(exp_data,(3,1,2,0)) # reshape to (60000, 32, 32, 3)

    norm_mean = (np.mean(exp_data[0])/255, np.mean(exp_data[1])/255, np.mean(exp_data[2])/255)
    norm_std   = (np.std(exp_data[0])/255, np.std(exp_data[1])/255, np.std(exp_data[2])/255)

    return(tuple(map(lambda x: np.round(x,2), norm_mean)), tuple(map(lambda x: np.round(x,2), norm_std)))

def get_album_transforms(norm_mean,norm_std):
    """get the train and test transform by albumentations"""
    album_train_transform = A.Compose([   A.HorizontalFlip(p=.2),
                                          A.VerticalFlip(p=.2),
                                          A.Rotate(limit=15,p=0.5),
                                          A.Normalize(
                                             mean=[0.49, 0.48, 0.45],
                                              std=[0.25, 0.24, 0.26], ),
                                          AP.transforms.ToTensor()
                                        ])

    album_test_transform = A.Compose([   A.Normalize(
                                             mean=[0.49, 0.48, 0.45],
                                              std=[0.25, 0.24, 0.26], ),
                                          AP.transforms.ToTensor()
                                        ])
    return(album_train_transform,album_test_transform)

def get_datasets():
    """Extract and transform the data"""
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,download=True)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,download=True)
    return(train_set,test_set)

def trasnform_datasets(train_set, test_set, train_transform, test_transform):
    """Transform the data"""
    train_set = AlbumentationsDataset(
                                    rimages= train_set.data,
                                    labels=train_set.targets,
                                    transform=train_transform,
                                    )

    test_set = AlbumentationsDataset(
                                    rimages= test_set.data,
                                    labels=test_set.targets,
                                    transform=test_transform,
                                    )
    return(train_set,test_set)

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
    # dataloader arguments
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64, num_workers=1)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
    # test dataloader
    test_loader  = torch.utils.data.DataLoader(test_set, **dataloader_args)
    return(train_loader,test_loader)
