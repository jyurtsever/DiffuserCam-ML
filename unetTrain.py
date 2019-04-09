
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import PerceptualSimilarity as ps
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
from save_model_utils import *
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset




def torch_to_im(i, torch_mat):
    dest_im = np.zeros((270,480,3))
    dest_im[:, :, 0]= torch_mat[i,0,:,:]
    dest_im[:, :, 1]= torch_mat[i,1,:,:]
    dest_im[:, :, 2]= torch_mat[i,2,:,:]
    dest_im /= np.max(dest_im)
    return dest_im



def train(model, optimizer, loss_fn, train_loader, epoch):
    # for batch_idx, item in enumerate(train_loader):
    #     X_batch, Y_batch = item['image'], item['label']
    #     optimizer.zero_grad()
    #     # print(X_batch.shape, "okkkkkk")
    #     # plt.imshow(Y_batch[1, 1, :,:].numpy())
    #     # plt.show()
    #     # plt.imshow(X_batch[1, 1, :, :].numpy())
    #     # plt.show()
    #     output = model(X_batch)
    #     loss = loss_fn(output, Y_batch)
    #     loss.sum().backward()#loss.backward()
    #     optimizer.step()
    #     if batch_idx % 20 == 0:
    #         print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / \
    #             len(train_loader), loss.sum().item()))

def evaluate(model, loss_fn, test_loader):
    output = None
    i = 0
    with torch.no_grad():
        try:
            for batch_idx, item in enumerate(test_loader):
                    X_batch, Y_batch = item['image'], item['label']
                    # print(X_batch.shape, "okkkkkk")
                    output = model(X_batch)
                    loss = loss_fn(output, Y_batch)
                    # loss.backward()

                    print('[{}/{} ({:.0f}%)] \t Test Loss: {:.6f}'.format(
                        batch_idx*len(X_batch), len(test_loader.dataset), 100.*batch_idx / \
                            len(test_loader), loss.sum().item()))
                    out = output.cpu().detach().numpy()
                    # gt = Y_batch.cpu().numpy()
                    recon = X_batch.cpu().numpy()
                    for j in range(out.shape[0]):
                        curr_out = torch_to_im(j, out)
                        # curr_gt = torch_to_im(j, gt)
                        curr_recon = torch_to_im(j, recon)
                        im_name = test_filenames.iloc[i, 0]
                        out_name = args.save_test_results + '/out/' + im_name
                        # gt_name = args.save_test_results + '/gt/' + im_name
                        recon_name = args.save_test_results + '/recon/' + im_name
                        scm.imsave(out_name, curr_out)
                        # scm.imsave(gt_name, curr_gt)
                        scm.imsave(recon_name, curr_recon)
                        i += 1
        except AttributeError as e:
            print("stopped at batch_idx: ", batch_idx)
    print("Running Kristinas's code")
    save_model_summary(model, test_loader, args)
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
    if args.net == 'UNet512':
        model = UNet512512((3, 128, 128))
    elif args.net == 'UNet256':
        model = UNet256256((3, 128, 128))
    else:
        raise IOError('Unrecognized net')
    if use_gpu:
        model = model.cuda()
    loss_fn = ps.PerceptualLoss().forward #nn.MSELoss()
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

    # CLI.add_argument(
    #     "--save_test_results",  # name on the CLI - drop the `--` for positional/required parameters
    #     type=str,
    # )

    # CLI.add_argument(
    #     "--filename",  # name on the CLI - drop the `--` for positional/required parameters
    #     type=str,
    # )

    CLI.add_argument(
        "--n_iters",  # name on the CLI - drop the `--` for positional/required parameters
        type=str,
    )

    CLI.add_argument(
        "--dset_size",
        type=str,
        default='big',
    )

    CLI.add_argument(
        "--gpu",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default=-1,  # default if nothing is provided
    )

    CLI.add_argument(
        "--net",
        type=str,
        default='UNet512'
    )

    args = CLI.parse_args()

    dir_name = 'net_' + args.net + '_ADMM_' + args.n_iters + '_dset_size_' + args.dset_size + '_loss_' + args.loss_fn
    save_path = '../saved_models/' + dir_name + '/'
    os.mkdir(save_path); os.mkdir(save_path + 'gt/'); os.mkdir(save_path + 'out/'); os.mkdir(save_path + 'recon/')
    # use_gpu = args.gpu != -1
    # if use_gpu:
    #     print("CURRENT DEVICE: ", torch.cuda.current_device(), "num_devices: ", torch.cuda.device_count())
    # data_dir = args.data_dir
    if args.dset_size == 'big':
        csv_path_test = '../saved_models/dataset_12_12.csv'
        test_filenames = pd.read_csv(csv_path_test)
        csv_path_train = '../saved_models/dataset_12_12_test.csv'
    elif args.dset_size == 'short':
        csv_path_test = '../saved_models/dataset_12_12_short.csv'
        test_filenames = pd.read_csv(csv_path_test)
        csv_path_train = '../saved_models/dataset_12_12_test_small.csv'
    else:
        raise IOError("Dataset size not recognized")
    gt_dir = args.gt_dir
    rec_dir = args.recon_dir
    BATCH_SIZE = int(args.batch_size)
    model = unet_optimize(args)
