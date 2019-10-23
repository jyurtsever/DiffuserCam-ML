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
import json
import scipy.misc as scm
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2

from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Dataset



class DiffuserDatasetClassif(Dataset):
   """Diffuser dataset."""

   def __init__(self, csv_file, data_dir, labels, suffix='.tiff', label_file=None, num_data=None, transform=None, use_gpu = False, flipud_gt=False):
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
       self.label_dir = label_file
       self.transform = transform
       self.use_gpu = use_gpu
       self.suffix = suffix
       if label_file:
           f = open(label_file)
           self.labels = json.load(f)['gt']
           f.close()
       else:
           self.labels = labels

   def __len__(self):
       return len(self.csv_contents)

   def __getitem__(self, idx):
       def initialize_img(path, flip=False):
           if self.suffix != '.tiff':
               path = path[:-5] + self.suffix
           if flip:
               img = np.flipud(cv2.imread(path, -1)).astype(np.float32) / 512
           else:
               img = cv2.imread(path, -1).astype(np.float32)/512
           if len(img.shape) > 2 and img.shape[2] == 4:
               img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
           return img

       img_name = self.csv_contents.iloc[idx,0]

       image = initialize_img(os.path.join(self.data_dir, img_name))
       label = self.labels[int(img_name[2:7])][img_name]
       label = torch.FloatTensor(label)
       if self.transform:
           image = self.transform(image)

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
       elif n==1:
           image_new = np.flipud(image.copy())
       elif n==2:
           image_new = np.fliplr(image.copy())
       return {'image': image_new,
               'label': label}

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
