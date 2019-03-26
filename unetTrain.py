
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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
import argparse
from reconNet import *
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset
# BATCH_SIZE = 200




def torch_to_im(i, torch_mat):
    dest_im = np.zeros((128,128,3))
    dest_im[:, :, 0]= torch_mat[i,0,:,:]
    dest_im[:, :, 1]= torch_mat[i,1,:,:]
    dest_im[:, :, 2]= torch_mat[i,2,:,:]
    dest_im /= np.max(dest_im)



def train(model, optimizer, loss_fn, train_loader, epoch):
    for batch_idx, item in enumerate(train_loader):
        X_batch, Y_batch = item['image'], item['label']
        optimizer.zero_grad()
        # print(X_batch.shape, "okkkkkk")
        # plt.imshow(Y_batch[1, 1, :,:].numpy())
        # plt.show()
        # plt.imshow(X_batch[1, 1, :, :].numpy())
        # plt.show()
        output = model(X_batch)
        loss = loss_fn(output, Y_batch)
        loss.backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / \
                len(train_loader), loss.item()))

def evaluate(model, loss_fn, test_loader):
    output = None
    i = 0
    with torch.no_grad():
        for batch_idx, item in enumerate(test_loader):
            X_batch, Y_batch = item['image'], item['label']
            # print(X_batch.shape, "okkkkkk")
            output = model(X_batch)
            loss = loss_fn(output, Y_batch)
            # loss.backward()

            print('[{}/{} ({:.0f}%)] \t Test Loss: {:.6f}'.format(
                batch_idx*len(X_batch), len(test_loader.dataset), 100.*batch_idx / \
                    len(test_loader), loss.item()))
            out = output.cpu().detach().numpy()
            gt = Y_batch.cpu().numpy()
            recon = X_batch.cpu().numpy()
            for j in range(out.shape[0]):
                curr_out = torch_to_im(j, out)
                curr_gt = torch_to_im(j, gt)
                curr_recon = torch_to_im(j, gt)
                im_name = test_filenames[i]
                out_name = args.save_test_results + '/out/' + im_name
                gt_name = args.save_test_results + '/gt/' + im_name
                recon_name = args.save_test_results + '/recon/' + im_name
                scm.misc.imsave(out_name, curr_out)
                scm.misc.imsave(gt_name, curr_gt)
                scm.misc.imsave(recon_name, curr_recon)
                i += 1
        # save_dict = {'pred': out, 'gt': gt, 'recon': recon}
        # io.savemat('test.mat', save_dict)

def run_train(model, optimizer, loss_fn, train_loader, num_epochs):
    for epoch in range(num_epochs):
        train(model, optimizer, loss_fn, train_loader, epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, args.save_test_results + "/saved_network.pt")

def unet_optimize(args):
    transformations = transforms.Compose([transforms.ToTensor()])
                                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    #std=[0.229, 0.224, 0.225])])
    train_set = DiffuserDataset(csv_path_train, rec_dir, gt_dir, transform=transformations, use_gpu=use_gpu)
    test_set = DiffuserDataset(csv_path_test, rec_dir, gt_dir, transform=transformations, use_gpu=use_gpu)
    # train_set = datasets.ImageFolder(csv_path_train, transform = transformations)
    # test_set = datasets.ImageFolder(csv_path_test, transform=transformations)
    train_loader = torchdata.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = torchdata.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)

    model = UNet512512((3, 128, 128))
    if use_gpu:
        model = model.cuda()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    run_train(model, optimizer, loss_fn, train_loader, int(args.num_epochs))
    evaluate(model, loss_fn, test_loader)

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--recon_dir",  # name on the CLI - drop the `--` for positional/required parameters
        type=str,
    )
    CLI.add_argument(
        "--gt_dir",  # name on the CLI - drop the `--` for positional/required parameters
        type=str,
    )

    CLI.add_argument(
        "--num_epochs",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default= 5,  # default if nothing is provided
    )

    CLI.add_argument(
        "--batch_size",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default=4,  # default if nothing is provided
    )

    CLI.add_argument(
        "--save_test_results",  # name on the CLI - drop the `--` for positional/required parameters
        type=str,
    )
    args = CLI.parse_args()


    # data_dir = args.data_dir
    csv_path_test = args.gt_dir + 'test_names.csv'
    test_filenames = pd.read_csv(csv_path_test)
    csv_path_train = args.gt_dir + 'train_names.csv'
    gt_dir = args.gt_dir
    rec_dir = args.recon_dir
    BATCH_SIZE = int(args.batch_size)
    model = unet_optimize(sys.argv)
