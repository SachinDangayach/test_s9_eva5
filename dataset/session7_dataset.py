# Module for Extract Load and Transform CIFAR10 data
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

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

def get_transforms(norm_mean,norm_std):
    """get the train and test transform"""
    print(norm_mean,norm_std)
    train_transform = transforms.Compose([transforms.RandomRotation(10) , transforms.RandomHorizontalFlip(0.20),
                                      transforms.ToTensor(),transforms.Normalize(norm_mean, norm_std)])
    test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm_mean, norm_std)])
    return(train_transform,test_transform)

def get_datasets(train_transform,test_transform):
    """Extract and transform the data"""
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=test_transform)
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
