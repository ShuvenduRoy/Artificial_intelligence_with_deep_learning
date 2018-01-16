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
train_dataset = dsets.MNIST(root='C:\datasets\mnist',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='C:\datasets\mnist',
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
input_size = 784
output_size = 10
num_epochs = 5
learning_rate = 0.001


# define the model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


# creating instance of model
model = LogisticRegression(input_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # convert input into torch varaible
        inputs = Variable(images.view(-1, 28 * 28))
        targets = Variable(labels)

        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)

        # backward
        loss = criterion(outputs, targets)
        loss.backward()

        # optimize
        optimizer.step()

        if (i) % 100 == 0:
            print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' 
                   % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# test the model
correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()


print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))


# save the model
torch.save(model.state_dict(), 'model.pkl')
