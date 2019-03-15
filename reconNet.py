import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
import pandas as pd
import numpy as np
import scipy.io as io
import imageio as imio
import os
import random
import scipy.misc as scm
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2

from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset



class DiffuserDataset(Dataset):
   """Diffuser dataset."""

   def __init__(self, csv_file, data_dir, label_dir, transform=None, use_gpu = False):
       """
       Args:
           csv_file (string): Path to the csv file with annotations.
           data_dir (string): Directory with all the Diffuser images.
           label_dir (string): Directory with all the natural images.
           transform (callable, optional): Optional transform to be applied
               on a sample.
       """
       self.csv_contents = pd.read_csv(csv_file)
       self.data_dir = data_dir
       self.label_dir = label_dir
       self.transform = transform
       self.use_gpu = use_gpu

   def __len__(self):
       return len(self.csv_contents)

   def __getitem__(self, idx):
       def initialize_img(path):
           img = cv2.imread(path, -1).astype(np.float32)/512
           if len(img.shape) > 2 and img.shape[2] == 4:
               img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
           # plt.imshow(img)
           # plt.show()
           return img

       img_name = self.csv_contents.iloc[idx,0]

       image = initialize_img(os.path.join(self.data_dir, img_name))
       label = initialize_img(os.path.join(self.label_dir, img_name))

       if self.transform:
           image = self.transform(image)
           label = self.transform(label)

       if self.use_gpu:
           image = image.cuda()
           label = label.cuda()

       sample = {'image': image, 'label': label}

       return sample

class FLIP(object):
   """Convert ndarrays in sample to Tensors."""

   def __call__(self, sample):
       image, label = sample['image'], sample['label']

       n = random.randint(0,2)

       if n==0:
           image_new = image.copy()
           label_new = label.copy()
       elif n==1:
           image_new = np.flipud(image.copy())
           label_new = np.flipud(label.copy())
       elif n==2:
           image_new = np.fliplr(image.copy())
           label_new = np.fliplr(label.copy())
       return {'image': image_new,
               'label': label_new}

class FLIP(object):
   """Convert ndarrays in sample to Tensors."""

   def __call__(self, sample):
       image, label = sample['image'], sample['label']

       n = random.randint(0,2)

       if n==0:
           image_new = image.copy()
           label_new = label.copy()
       elif n==1:
           image_new = np.flipud(image.copy())
           label_new = np.flipud(label.copy())
       elif n==2:
           image_new = np.fliplr(image.copy())
           label_new = np.fliplr(label.copy())
       return {'image': image_new,
               'label': label_new}

class ROT90(object):
   """Convert ndarrays in sample to Tensors."""

   def __call__(self, sample):
       image, label = sample['image'], sample['label']

       n = random.randint(0,2)

       if n==0:
           image_new = image.copy()
           label_new = label.copy()
       elif n==1:
           image_new = np.rot90(image.copy(),axes = [1,0])
           label_new = np.rot90(label.copy(),axes = [1,0])
       elif n==2:
           image_new = np.rot90(image.copy(),axes = [0,1])
           label_new = np.rot90(label.copy(),axes = [0,1])
       return {'image': image_new,
               'label': label_new}


class SingleColor(object):
   """Convert ndarrays in sample to Tensors."""

   def __call__(self, sample):
       image, label = sample['image'], sample['label']

       n = random.randint(0,2)
       # swap color axis because
       # numpy image: H x W x C
       # torch image: C X H X W

       image_new = np.empty([512, 512, 1])
       label_new = np.empty([512, 512, 1])
       image_new[:,:,0] = image[:,:,n]
       label_new[:,:,0] = label[:,:,n]

       return {'image': image_new,
               'label': label_new}

class ToTensor(object):
   """Convert ndarrays in sample to Tensors."""

   def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image.copy()).type(torch.FloatTensor),
                   'label': torch.from_numpy(label.copy()).type(torch.FloatTensor)}


BN_EPS = 1e-4


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x =  F.interpolate(     x, size=(height, width), mode='bilinear')# F.upsample(x, size=(height, width), mode='bilinear')
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x

    # 32x32


class UNet512512(nn.Module):
    def __init__(self, in_shape):
        super(UNet512512, self).__init__()
        channels, height, width = in_shape

        self.down1 = StackEncoder(3, 24, kernel_size=3)  # 256
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16

        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, 3, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))

    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        down5, out = self.down5(out)

        out = self.center(out)
        out = self.up5(out, down5)
        out = self.up4(out, down4)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        # print('yooo', out.shape)
        return out


