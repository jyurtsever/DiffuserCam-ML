
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import pandas as pd
import numpy as np
import scipy.io as io
import os
import random
import scipy.misc as scm
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import sys
from reconNet import *
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset
BATCH_SIZE = 32








def train(model, optimizer, loss_fn, train_loader, epoch):
    for batch_idx, item in enumerate(train_loader):
        X_batch, Y_batch = item['image'], item['label']
        optimizer.zero_grad()
        # print(X_batch.shape, "okkkkkk")
        output = model(X_batch)
        loss = loss_fn(output, Y_batch)
        loss.backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / \
                len(train_loader), loss.item()))
    o = output.detach().numpy()
    o = o.reshape((128, 128, 3, 32))
    plt.imshow(o[:, :, :, 0])
    plt.show()

def run_train(model, optimizer, loss_fn, train_loader, num_epochs):
    for epoch in range(num_epochs):
        train(model, optimizer, loss_fn, train_loader, epoch)


def unet_optimize():
    train_set = DiffuserDataset(csv_path, rec_dir, gt_dir)
    train_loader = torchdata.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = False)
    model = UNet512512((3, 128, 128))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    run_train(model, optimizer, loss_fn, train_loader, 1)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    csv_path = data_dir + 'filenames.csv'
    gt_dir = data_dir + 'gt'
    rec_dir = data_dir + 'recon'
    model = unet_optimize()
    # evaluate(model)