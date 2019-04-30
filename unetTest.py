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



def evaluate(model, test_loader):
    print("Running Kristinas's code")
    model.load_state_dict(torch.load(save_path + 'saved_network.pt'))
    save_model_summary(model, test_loader, args)

def unet_optimize(args):
    transformations = transforms.Compose([transforms.ToTensor()])
                                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    #std=[0.229, 0.224, 0.225])])
    train_set = DiffuserDataset(csv_path_train, rec_dir, gt_dir, num_data=int(args.dset_size), transform=transformations, use_gpu=use_gpu)
    test_set = DiffuserDataset(csv_path_test, rec_dir, gt_dir, transform=transformations, use_gpu=use_gpu)
    # train_set = datasets.ImageFolder(csv_path_train, transform = transformations)
    ## test_set = datasets.ImageFolder(csv_path_test, transform=transformations)
    train_loader = torchdata.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = torchdata.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)
    if args.net == 'UNet512':
        model = UNet512512((3, 128, 128))
    elif args.net == 'UNet256':
        model = UNet256256((3, 128, 128))
        print('Using unet256')
    elif args.net == 'UNet128':
        model = UNet128128((3, 128, 128))
        print('Using UNet128')
    elif args.net == 'ResNet':
        model = reconResNet()
        print('ResNet')
    elif args.net == 'ResNet2':
        model = reconResNet2()
        print('ResNet')
    elif args.net == 'ResNet3':
        model = reconResNet3()
        print('ResNet')
    elif args.net == 'ResNet4':
        model = reconResNet4()
        print('ResNet')
    else:
        raise IOError('ERROR: Unrecognized net')
    if use_gpu:
        model = model.cuda()

    if args.loss_fn == 'lpips':
        loss_fn = ps.PerceptualLoss().forward #nn.MSELoss()
    elif args.loss_fn == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss_fn == 'both':
        loss_lpips = ps.PerceptualLoss().forward
        loss_mse = nn.MSELoss()
        loss_fn = lambda output, Y_batch: 10*loss_mse(output, Y_batch) + loss_lpips(output, Y_batch).sum()
    else:
        raise IOError('ERROR: Unrecognized loss')
    evaluate(model, test_loader)

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
        "--loss_fn",  # name on the CLI - drop the `--` for positional/required parameters
        type=str,
        default='lpips'
    )

    CLI.add_argument(
        "--n_iters",  # name on the CLI - drop the `--` for positional/required parameters
        type=str,
    )

    CLI.add_argument(
        "--dset_size",
        type=str,
        default=None,
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
    CLI.add_argument(
        "--save_path",
        type=str,
        default='../saved_models_4_4/'
    )

    args = CLI.parse_args()
    if args.dset_size:
        dir_name = 'net_' + args.net + '_ADMM_' + args.n_iters + '_dset_size_' + args.dset_size + '_loss_' + args.loss_fn
    else:
        dir_name = 'net_' + args.net + '_ADMM_' + args.n_iters + '_dset_size_' + 'all' + '_loss_' + args.loss_fn
    save_path = args.save_path + dir_name + '/'
    #os.mkdir(save_path); os.mkdir(save_path + 'gt/'); os.mkdir(save_path + 'out/'); os.mkdir(save_path + 'recon/')
    if args.save_path == '../saved_models/':
        if args.dset_size != 'short':
            csv_path_test = '../saved_models/dataset_12_12_test.csv'
            test_filenames = pd.read_csv(csv_path_test)
            csv_path_train = '../saved_models/dataset_12_12.csv'
        elif args.dset_size == 'short':
            csv_path_test = '../saved_models/dataset_12_12_test_short.csv'
            test_filenames = pd.read_csv(csv_path_test)
            csv_path_train = '../saved_models/dataset_12_12_short.csv'
        else:
            raise IOError("Dataset size not recognized")

    elif args.save_path == '../saved_models_4_4/':
        csv_path_train = '../saved_models_4_4/train_names.csv'
        csv_path_test = '../saved_models_4_4/test_names.csv'

    else:
        raise IOError("save path does not exist")

    test_filenames = pd.read_csv(csv_path_test)

    gt_dir = args.gt_dir
    rec_dir = args.recon_dir
    BATCH_SIZE = int(args.batch_size)
    model = unet_optimize(args)
