import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import numpy as np
import scipy.io as io
import scipy.misc as scm
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import math

BATCH_SIZE = 100
EPOCHS = 2
mnist_data = io.loadmat('mnist-data/mnist_data.mat')
training_data = mnist_data['training_data']
tr_data = torch.from_numpy(training_data)
training_label = mnist_data['training_labels']
tr_label = torch.from_numpy(training_label)
test_data = mnist_data['test_data']
t_data = torch.from_numpy(test_data)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return F.log_softmax(X, dim=1)


class Net(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, loss_fn, train_loader, epoch):
    correct = 0
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X_batch.float())
        y_batch = y_batch.squeeze(1)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        # Total correct predictions
        predicted = torch.max(output.data, 1)[1].float()
        # print(torch.max(output.data, 1))
        correct += (predicted == y_batch.float()).sum()
        # print(correct)

        if batch_idx % 50 == 0:
            print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / \
                len(train_loader), loss.item(), float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))


def run_train(model, optimizer, loss_fn, train_loader, num_epochs):
    for epoch in range(num_epochs):
        train(model, optimizer, loss_fn, train_loader, epoch)

def mlp_optimize():
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())  # torch.optim.SGD(model.parameters(), lr=0.01)

    X_train, X_test, y_train, y_test = train_test_split(training_data, training_label, test_size=0.15)
    print(y_test.shape)

    torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
    torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # data type is long

    # create feature and targets tensor for test set.
    torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
    torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # data type is long

    # Pytorch train and test sets
    train_tens = torchdata.TensorDataset(torch_X_train, torch_y_train)
    test_tens = torchdata.TensorDataset(torch_X_test, torch_y_test)

    # data loader
    train_loader = torchdata.DataLoader(train_tens, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torchdata.DataLoader(test_tens, batch_size=BATCH_SIZE, shuffle=False)
    run_train(model, optimizer, loss_fn, train_loader, EPOCHS)
    return model, test_loader

def conv_optimize():
    model = CNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())  # torch.optim.SGD(model.parameters(), lr=0.01)

    X_train, X_test, y_train, y_test = train_test_split(training_data, training_label, test_size=0.15)
    print(y_test.shape)

    torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor).view(-1, 1,28,28).float()
    torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # data type is long

    # create feature and targets tensor for test set.
    torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor).view(-1, 1,28,28).float()
    torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # data type is long

    # Pytorch train and test sets
    train_tens = torchdata.TensorDataset(torch_X_train, torch_y_train)
    test_tens = torchdata.TensorDataset(torch_X_test, torch_y_test)

    # data loader
    train_loader = torchdata.DataLoader(train_tens, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torchdata.DataLoader(test_tens, batch_size=BATCH_SIZE, shuffle=False)
    run_train(model, optimizer, loss_fn, train_loader, EPOCHS)
    return model, test_loader

def evaluate(model, test_loader):
    correct = 0
    for test_imgs, test_labels in test_loader:
        test_imgs = test_imgs.float()
        test_labels = test_labels.squeeze(1)
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        correct += (predicted == test_labels).sum()

    print("Test accuracy:{:.3f}% ".format( float(correct *100) / (len(test_loader)*BATCH_SIZE)))

if __name__ == '__main__':
    model, test_loader = conv_optimize()
    evaluate(model, test_loader)
