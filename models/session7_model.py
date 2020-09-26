# Module to define model architecture for CIFAR10 data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""Class to define Model architecture for classification of cifar10_mean_std"""
class Net(nn.Module):
    def __init__(self, dropout_value = 0.1):
        super(Net, self).__init__()
        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value), # output_size = 32, RF = 3

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 32, RF = 5

        # TRANSITION BLOCK 1
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=1, bias=False), # output_size = 34, RF = 5
            nn.MaxPool2d(2, 2, 1)
        ) # output_size = 18, RF = 6

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value), # output_size = 18, RF = 10

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 18, RF = 14

        # TRANSITION BLOCK 2
        self.transblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=1, bias=False), # output_size = 20, RF = 14
            nn.MaxPool2d(2, 2, 1)
        ) # output_size = 11, RF = 16

        # CONVOLUTION BLOCK 3 -> DWS and Dialted Conv
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, groups=64, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value), # output_size = 11, 24

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value), # output_size = 11, 24

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)
        ) # output_size = 9, RF = 40

        # TRANSITION BLOCK 3
        self.transblock3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), padding=1, bias=False), # output_size = 11, RF = 40
            nn.MaxPool2d(2, 2, 1) # output_size = 6, RF = 44
        )

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value), # output_size = 6, RF = 60

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)
        ) # output_size = 6, RF = 76

        # OUTPUT BLOCK
        self.opblock= nn.Sequential(
            nn.AvgPool2d(kernel_size=6), # output_size = 1

            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) # output_size = 1, RF = 116


    def forward(self, x):
        x = self.convblock1(x)
        x = self.transblock1(x)
        x = self.convblock2(x)
        x = self.transblock2(x)
        x = self.convblock3(x)
        x = self.transblock3(x)
        x = self.convblock4(x)
        x = self.opblock(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
