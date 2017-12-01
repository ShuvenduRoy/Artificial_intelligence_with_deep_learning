import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# load dataset
batch_size = 100

# MNIST Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='C:\datasets\CIFAR10',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.CIFAR10(root='C:\datasets\CIFAR10',
                           train=False,
                           transform=transforms.ToTensor())

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Hyper Parameters
num_epochs = 5
learning_rate = 0.001

# image transformation
transform = transforms.Compose([
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



# define the model
class (nn.Module):
    def __init__(self ):
        super(, self).__init__()
        

    def forward(self, ):
        


# creating instance of model
model = 

# Loss and optimizer
criterion = 
optimizer = torch.optim.(model.parameters(), lr=learning_rate)

# train the model
for epoch in range(num_epochs):
    for i,  in enumerate(train_loader):
        inputs = 
        targets = 
    
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
    
        # backward
        loss = criterion(outputs, targets)
        loss.backward()
    
        # optimize
        optimizer.step()
    
        if (i+1) % 100 == 0:
            print ("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, 80, i+1, 500, loss.data[0]))

# save the model
torch.save(model.state_dict(), 'model.pkl')
