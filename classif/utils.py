import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import os
import cv2 as cv
import scipy
import skimage
import time
import torch


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse)), mse


def downsample_ax(img, factor):
    n = int(np.log2(factor))
    for i in range(n):
        if len(img.shape) == 2:
            img = .25 * (img[::2, ::2] + img[1::2, ::2]
                + img[::2, 1::2] + img[1::2, 1::2])
        else:
            img = .25 * (img[::2, ::2, :] + img[1::2, ::2, :]
                + img[::2, 1::2, :] + img[1::2, 1::2, :])
    return(img)



def remove_nan_gradients(grads):
    # Get rid of NaN gradients
    for g in range(0,len(grads)):
        if np.any(tf.is_nan(grads[g])):
            new_grad = tf.where(tf.is_nan(grads[g]), tf.zeros_like(grads[g]), grads[g])
            grads[g] = new_grad
    return grads

def cap_grads_by_norm(grads):
    capped_grads = [(tf.clip_by_norm(gradcl, 1.)) for gradcl in grads]
    return capped_grads


def load_psf_image(psf_file, downsample=400, rgb=True):

    if rgb==True:
        my_psf = rgb2gray(np.array(Image.open(psf_file)))
    else:
        my_psf = np.array(Image.open(psf_file))
        
    psf_bg = np.mean(my_psf[0 : 15, 0 : 15])             #102
    psf_down = downsample_ax(my_psf - psf_bg, downsample)
    
    psf_down = psf_down/np.linalg.norm(psf_down)
    
    return(psf_down)

def load_test_image(path):
    
    testim = cv.imread(path, -1).astype(np.float32)/4095. - 0.008273973
    testim = downsample_ax(testim, 4)

    image = testim.transpose((2, 0, 1))
    image = np.expand_dims(image,0)
    
    return image
    

from IPython import display
def print_function(x, i):
    plt.cla()
    plt.imshow(x)
    plt.title('iterations: '+ str(i));
    display.display(plt.gcf())
    display.clear_output(wait=True)
    
    
def gkern(DIMS0, DIMS1, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(DIMS0)
    interval2 = (2*nsig+1.)/(DIMS1)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., DIMS0+1)
    y = np.linspace(-nsig-interval/2., nsig+interval/2., DIMS1+1)
    
    kern1d = np.diff(st.norm.cdf(x))
    kern1d2 = np.diff(st.norm.cdf(y))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d2))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


from torch.utils.data import Dataset, DataLoader 
class DiffuserDataset_preprocessed(Dataset):
    """Diffuser dataset."""

    def __init__(self, csv_file, data_dir, label_dir, ds, transform=None):
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
        self.ds = ds
        
        
    def __len__(self):
        return len(self.csv_contents)

    def __getitem__(self, idx):
        
        t = time.time()
        img_name = self.csv_contents.iloc[idx,0]

        path_diffuser = os.path.join(self.data_dir, img_name) 
        path_gt = os.path.join(self.label_dir, img_name)
        
        image = np.load(path_diffuser[0:-9]+'.npy')
        label = np.load(path_gt[0:-9]+'.npy')
        
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

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
    
class DiffuserDataset_preprocessed_number(Dataset):
    """Diffuser dataset."""

    def __init__(self, csv_file, data_dir, label_dir, ds, num_images, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            data_dir (string): Directory with all the Diffuser images.
            label_dir (string): Directory with all the natural images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_contents = pd.read_csv(csv_file, nrows=num_images)
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.ds = ds
        
        
    def __len__(self):
        return len(self.csv_contents)

    def __getitem__(self, idx):
        
        t = time.time()
        img_name = self.csv_contents.iloc[idx,0]

        path_diffuser = os.path.join(self.data_dir, img_name) 
        path_gt = os.path.join(self.label_dir, img_name)
        
        #image = cv.imread(path_diffuser, -1).astype(np.float32)/4095.
        #label = cv.imread(path_gt, -1).astype(np.float32)/4095. 
        image = np.load(path_diffuser[0:-9]+'.npy')
        label = np.load(path_gt[0:-9]+'.npy')
        
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

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
    


##### Run test #####

def save_model_summary(model, admm, filename, device, description, test_loader):
    #model = model.to(device)

    loss_dict = test_training_images(model, admm, test_loader, device)
    time_gpu= run_timing_test(model, test_loader, device)

    loss_dict['time_gpu'] = time_gpu; #loss_dict['time_cpu'] = time_cpu

    loss_dict['filename'] = filename
    loss_dict['description'] = description


    save_filename = ('saved_models/saved_stats2/'+loss_dict['filename'])[0:-3]
    
    print('\r', 'Saving as:', save_filename, end = '')
    scipy.io.savemat(save_filename, loss_dict)
    return loss_dict



    
def run_timing_test(model, test_loader, device, num_trials = 100):
    print('\r', 'running timing test', end = '')
    t_avg_gpu = 0
    t_avg_cpu = 0
    
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            inputs = sample_batched['image'].to(device); 
            break

    #model = model.to(device)

    print('\r', 'running GPU timing test', end = '')
    for i in range(0,num_trials):
        with torch.no_grad():
            t = time.time()
            output = model(inputs)
            elapsed = time.time() - t
            t_avg_gpu = t_avg_gpu + elapsed

    #model_cpu = model.to('cpu')
    #inputs_cpu = inputs.to('cpu')
    
    #if model.__class__.__name__ == 'MyEnsemble':
    #    model_cpu.admm_model.to('cpu')
    #    model_cpu.denoise_model.to('cpu')
    
    #print('\r', 'running CPU timing test', end = '')
    #for i in range(0,num_trials):
    #    t = time.time()
    #    output_cpu = model_cpu(inputs_cpu)
    #    elapsed = time.time() - t
    #    t_avg_cpu = t_avg_cpu + elapsed


    t_avg_gpu = t_avg_gpu/num_trials 
    #t_avg_cpu = t_avg_cpu/num_trials 
    
    return t_avg_gpu#, t_avg_cpu


##### Plotting functions 
def preplot(image):
    image = np.transpose(image, (1,2,0))
    image_color = np.zeros_like(image); 
    image_color[:,:,0] = image[:,:,2]; image_color[:,:,1]  = image[:,:,1]
    image_color[:,:,2] = image[:,:,0];
    out_image = np.flipud(np.clip(image_color, 0,1))
    return out_image #[60:,62:-38,:]

def preplotn(image):
    image_color = np.zeros_like(image); 
    image_color[:,:,0] = image[:,:,2]; image_color[:,:,1]  = image[:,:,1]
    image_color[:,:,2] = image[:,:,0];
    out_image = np.flipud(np.clip(image_color, 0,1))
    return out_image

def run_color_recon(model, input_image):
    out_image = np.zeros_like(input_image)
    for i in range(0,3):
        out_image[:,:,:,i],_=model(input_image[:,:,:,i])
    return out_image

def run_time_test(model, inputs):
    t = time.time()
    out_color_converged = run_color_recon(model, inputs)
    elapsed = time.time() - t
    
    out_psnr = psnr(inputs, out_color_converged)
    out_mse = np.mean((inputs - out_color_converged) ** 2)
    
    return out_color_converged, elapsed, out_psnr, out_mse

def run_time_test_real(model, inputs, labels):
    t = time.time()
    out_color_converged = run_color_recon(model, inputs)
    elapsed = time.time() - t
    
    out_psnr = psnr(labels, out_color_converged[0]/np.max(out_color_converged[0]))
    out_mse = np.mean((labels - out_color_converged) ** 2)
    
    return out_color_converged, elapsed, out_psnr, out_mse
