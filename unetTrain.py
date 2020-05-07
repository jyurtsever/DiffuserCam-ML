
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
import imageio
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
        if args.loss_fn == 'lpips':
            loss.sum().backward()  # loss.backward()
        elif args.loss_fn == 'mse' or args.loss_fn == 'both':
            loss.backward()

        optimizer.step()
        if batch_idx % 20 == 0:
            print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / \
                len(train_loader), loss.sum().item()))

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
                    if args.loss_fn == 'lpips':
                        l = loss.sum()  # loss.backward()
                    elif args.loss_fn == 'mse' or args.loss_fn == 'both':
                        l = loss
                    print('[{}/{} ({:.0f}%)] \t Test Loss: {:.6f}'.format(
                        batch_idx*len(X_batch), len(test_loader.dataset), 100.*batch_idx / \
                            len(test_loader), l))
                    if batch_idx < 7:
                        out = output.cpu().detach().numpy()
                        # gt = Y_batch.cpu().numpy()
                        recon = X_batch.cpu().numpy()
                        for j in range(out.shape[0]):
                            curr_out = torch_to_im(j, out)
                            # curr_gt = torch_to_im(j, gt)
                            curr_recon = torch_to_im(j, recon)
                            im_name = test_filenames.iloc[i, 0]
                            out_name = save_path + '/out/' + im_name
                            # gt_name = save_path + '/gt/' + im_name
                            recon_name = save_path + '/recon/' + im_name
                            imageio.imwrite(out_name, curr_out)
                            # scm.imsave(gt_name, curr_gt)
                            imageio.imwrite(recon_name, curr_recon)
                            i += 1
        except AttributeError as e:
            print(e)
            print("stopped at batch_idx: ", batch_idx)
    print("Running Kristinas's code")
    save_model_summary(model, test_loader, args)
        # save_dict = {'pred': out, 'gt': gt, 'recon': recon}
        # io.savemat('test.mat', save_dict)

def run_train(model, optimizer, loss_fn, train_loader, num_epochs):
    epoch = 0
    for epoch in range(num_epochs):
        train(model, optimizer, loss_fn, train_loader, epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, save_path + "/saved_network.pt")

def unet_optimize(args):
    transformations = transforms.Compose([transforms.ToTensor()])
                                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    #std=[0.229, 0.224, 0.225])])
    if args.dset_size != "big":
        train_set = DiffuserDataset(csv_path_train, rec_dir, gt_dir, num_data=int(args.dset_size), transform=transformations, use_gpu=use_gpu)
    else:
        train_set = DiffuserDataset(csv_path_train, rec_dir, gt_dir, transform=transformations, use_gpu=use_gpu)

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

    if args.resume:
        model.load_state_dict(torch.load(args.resume)['model_state_dict'])

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        "--resume",
        type=str,
        default=None
    )

    CLI.add_argument(
        "--save_path",
        type=str,
        default='../saved_models_4_4/'
    )

    CLI.add_argument(
        "--lr",
        type=float,
        default= 0.001
    )

    CLI.add_argument(
        "--dir_name",
        type=str,
        default=None
    )

    CLI.add_argument(
        "--weight_decay",
        type=float,
        default=0.0
    )


    args = CLI.parse_args()
    if args.dir_name:
       dir_name = args.dir_name

    elif args.dset_size:
        dir_name = 'net_' + args.net + '_ADMM_' + args.n_iters + '_dset_size_' + args.dset_size + '_loss_' + args.loss_fn
    
    else:
        dir_name = 'net_' + args.net + '_ADMM_' + args.n_iters + '_dset_size_' + 'all' + '_loss_' + args.loss_fn
    save_path = args.save_path + dir_name + '/'

    if not args.resume:
        os.mkdir(save_path); os.mkdir(save_path + 'gt/'); os.mkdir(save_path + 'out/'); os.mkdir(save_path + 'recon/')
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
