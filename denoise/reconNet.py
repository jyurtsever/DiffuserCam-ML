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

   def __init__(self, csv_file, data_dir, label_dir, num_data=None, transform=None, use_gpu = False, flipud_gt=False):
       """
       Args:
           csv_file (string): Path to the csv file with annotations.
           data_dir (string): Directory with all the Diffuser images.
           label_dir (string): Directory with all the natural images.
           transform (callable, optional): Optional transform to be applied
               on a sample.
       """
       if num_data:
           self.csv_contents = pd.read_csv(csv_file, nrows=num_data)
       else:
           self.csv_contents = pd.read_csv(csv_file)
       self.data_dir = data_dir
       self.label_dir = label_dir
       self.transform = transform
       self.use_gpu = use_gpu
       self.flipud_gt = flipud_gt

   def __len__(self):
       return len(self.csv_contents)

   def __getitem__(self, idx):
       def initialize_img(path, flip=False):
           #print(path)
           if flip:
               img = np.flipud(cv2.imread(path, -1)).astype(np.float32) / 512
           else:
               img = cv2.imread(path, -1).astype(np.float32)/512
           if len(img.shape) > 2 and img.shape[2] == 4:
               img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
           return img

       img_name = self.csv_contents.iloc[idx,0]

       image = initialize_img(os.path.join(self.data_dir, img_name))
       if self.flipud_gt:
           label = initialize_img(os.path.join(self.label_dir, img_name), flip=True)
       else:
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

class UNet256256(nn.Module):
    def __init__(self, in_shape):
        super(UNet256256, self).__init__()
        channels, height, width = in_shape

        self.down1 = StackEncoder(3, 24, kernel_size=3)  # 256
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        # self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
        #
        # self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, 3, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(256, 256, kernel_size=3, padding=1))

    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)
        # down5, out = self.down5(out)

        out = self.center(out)
        # out = self.up5(out, down5)
        out = self.up4(out, down4)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        # print('yooo', out.shape)
        return out


class UNet128128(nn.Module):
    def __init__(self, in_shape):
        super(UNet128128, self).__init__()
        channels, height, width = in_shape

        self.down1 = StackEncoder(3, 24, kernel_size=3)  # 256
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        #self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        # self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
        #
        # self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        #self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, 3, kernel_size=1, bias=True)

        self.center = nn.Sequential(ConvBnRelu2d(128, 128, kernel_size=3, padding=1))

    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        # down4, out = self.down4(out)
        # down5, out = self.down5(out)

        out = self.center(out)
        # out = self.up5(out, down5)
        # out = self.up4(out, down4)
        out = self.up3(out, down3)
        out = self.up2(out, down2)
        out = self.up1(out, down1)

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        # print('yooo', out.shape)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, widths=[64, 128, 128, 64], zero_init_residual=False, dropout=0):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, widths[0], layers[0])
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=1)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=1)
        self.layer4 = self._make_layer(block, widths[3], layers[3], stride=1)
        self.classify = nn.Conv2d(widths[3], 3, kernel_size=1, bias=True)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
       # self.fc = nn.Linear(widths[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classify(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        # if self.dropout is not None:
        #     x = self.dropout(x)
        # x = self.fc(x)
        return x

def reconResNet():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def reconResNet2():
    return ResNet(BasicBlock, [1, 1, 1, 1])

def reconResNet3():
    return ResNet(BasicBlock, [2, 2, 2, 2], widths=[32, 64, 64, 32])

def reconResNet4():
    return ResNet(BasicBlock, [1, 1, 1, 1], widths=[32, 64, 64, 32])
