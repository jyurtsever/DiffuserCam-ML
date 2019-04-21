import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import tensorflow as tf
import os
import cv2 as cv
import scipy
import skimage
import time
import tensorflow.contrib.eager as tfe
#import tensorflow_probability as tfp


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


def _read_py_function(filename, filename_gt, ds):
    path = '/home/jyurtsever/research/mirflickr25k/'
    
    name = filename.decode()
    path_diffuser = os.path.join(path, 'diffuser_images_2_14_auto', name)
    path_gt = os.path.join(path, 'gt_images_2_14_auto/', name)
    image_diffuser = cv.imread(path_diffuser, -1).astype(np.float32)/2048. - 0.008273973
    image_gt = cv.imread(path_gt, -1).astype(np.float32)/256. 
    image_diffuser_out = downsample_ax(image_diffuser, ds)
    # Apply calibration 
    calib_data = scipy.io.loadmat('../recon_files/calibration_2_15_v2.mat')
#     image_gt_out = cv.undistort(image_gt, calib_data['mtx'], calib_data['dist'])  # Lens correction
#     image_gt_out = np.flipud(downsample_ax(image_gt_out, ds))                      # Downsample and flip
    image_gt_out = np.flipud(image_gt)
#     #Warp image to align with Diffuser: 
#     image_gt_out = cv.warpAffine(image_gt_out, calib_data['M'], (image_gt_out.shape[1], image_gt_out.shape[0]))
    return image_diffuser_out, image_gt_out


def read_and_downsample_im(filename, filename_gt, ds):
    path = '/home/jyurtsever/research/mirflickr25k/'
    name = filename
    path_diffuser = os.path.join(path, 'diffuser_images_2_14_auto', name)
    path_gt = os.path.join(path, 'gt_images_2_14_auto/', name)
    image_diffuser = cv.imread(path_diffuser, -1).astype(np.float32) / 2048. - 0.008273973
    image_gt = cv.imread(path_gt, -1).astype(np.float32) / 256.
    image_diffuser_out = downsample_ax(image_diffuser, ds)
    # Apply calibration
    calib_data = scipy.io.loadmat('../recon_files/calibration_2_15_v2.mat')
    #     image_gt_out = cv.undistort(image_gt, calib_data['mtx'], calib_data['dist'])  # Lens correction
    #     image_gt_out = np.flipud(downsample_ax(image_gt_out, ds))                      # Downsample and flip
    image_gt_out = np.flipud(image_gt)
    #     #Warp image to align with Diffuser:
    #     image_gt_out = cv.warpAffine(image_gt_out, calib_data['M'], (image_gt_out.shape[1], image_gt_out.shape[0]))
    return image_diffuser_out, image_gt_out

def make_dataset(csv_file, ds, start=0):
    csv_contents = pd.read_csv(csv_file)
    
    filenames_diffuser = []
    filenames_gt = []
    filenames = []
    for i in range(start, len(csv_contents)):
        filenames_diffuser.append(csv_contents.iloc[i,0])
        filenames_gt.append(csv_contents.iloc[i,0])
        filenames.append(csv_contents.iloc[i,0])
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, filenames))
    dataset = dataset.map(lambda filenames_diffuser, filename_gt: tuple(tf.py_func(
        _read_py_function, [filenames_diffuser, filenames_gt,ds], [tf.float32, tf.float32])))
    

    
    return dataset, len(csv_contents), filenames_diffuser

def preplot(image):
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
    



""" Includes helper functions that are used in admm.py and model.py
Last updated: 2/22/2019 

Overview:

    * Padding and cropping functions
    * FFT shifting functions
    * Forward Model (H, Hadj)
    * Soft thresholding functions
    * TV forward/adjoint operators 
"""


####### Padding and cropping functions #####
def pad(model, x):
    num_dims = len(x.shape)
    if num_dims == 3:
        PADDING = tf.constant([[0, 0], [model.PAD_SIZE0, model.PAD_SIZE0], [model.PAD_SIZE1, model.PAD_SIZE1]])
    if num_dims == 2:
        PADDING = tf.constant([[model.PAD_SIZE0, model.PAD_SIZE0], [model.PAD_SIZE1, model.PAD_SIZE1]])
    return(tf.pad(x,  PADDING, "CONSTANT"))

def pad_dim3(model, x):
    PADDING = tf.constant([[0, 0], [model.PAD_SIZE0, model.PAD_SIZE0], [model.PAD_SIZE1, model.PAD_SIZE1]])
    return(tf.pad(x,  PADDING, "CONSTANT"))

def pad_dim3_rgb(model, x):
    PADDING = tf.constant([[0, 0], [model.PAD_SIZE0, model.PAD_SIZE0], [model.PAD_SIZE1, model.PAD_SIZE1], [0, 0]])
    return(tf.pad(x,  PADDING, "CONSTANT"))

def pad_dim2(model, x):
    PADDING = tf.constant([[model.PAD_SIZE0, model.PAD_SIZE0], [model.PAD_SIZE1, model.PAD_SIZE1]])
    return(tf.pad(x,  PADDING, "CONSTANT"))

def crop(model, x):
    C01 = model.PAD_SIZE0; C02 = model.PAD_SIZE0 + model.DIMS0              # Crop indices 
    C11 = model.PAD_SIZE1; C12 = model.PAD_SIZE1 + model.DIMS1              # Crop indices 
    return x[:, C01:C02, C11:C12]


####### FFT Shifting #####
def tf_ifftshift2(model, x):
    left, right = tf.split(x, [model.DIMS0,model.DIMS0], axis=0)
    x = tf.concat([right, left],0)
    top, bottom = tf.split(x, [model.DIMS1,model.DIMS1],1)
    x = tf.concat([bottom, top],1)
    return(x)

def tf_ifftshift(model, x):
    dim0 = tf.floordiv(x.shape[1], 2)
    dim1 = tf.floordiv(x.shape[2], 2)
    left, right = tf.split(x, [dim0, dim0], axis=1)
    x = tf.concat([right, left],1)
    top, bottom = tf.split(x, [dim1, dim1],2)
    x = tf.concat([bottom, top],2)
    return(x)
    
####### Forward Model #####
def Hfor(model, x):
    x = tf.fft2d(tf.complex(tf_ifftshift(model, x), tf.zeros([model.batch_size, model.DIMS0*2, model.DIMS1*2], 
                                        dtype=tf.float32)))
    x = model.H*x
    x = tf_ifftshift(model, tf.ifft2d(x))
    return tf.real(x)

def Hadj(model, x):
    x = tf.fft2d(tf.complex(tf_ifftshift(model, x), tf.zeros([model.batch_size, model.DIMS0*2, model.DIMS1*2], 
                                        dtype=tf.float32)))
    x = model.HT*x
    x = tf_ifftshift(model, tf.ifft2d(x))
    return tf.real(x)

def Hfor_rgb(model, x):
    x = tf.fft3d(tf.complex(tf_ifftshift(model, x), tf.zeros([model.batch_size, model.DIMS0*2, model.DIMS1*2, 3], 
                                        dtype=tf.float32)))
    x = tf.expand_dims(tf.expand_dims(model.H, 0), 3)*x
    x = tf_ifftshift(model, tf.ifft3d(x))
    return tf.real(x)

def Hadj_rgb(model, x):
    x = tf.fft3d(tf.complex(tf_ifftshift(model, x), tf.zeros([model.batch_size, model.DIMS0*2, model.DIMS1*2, 3], 
                                        dtype=tf.float32)))
    x = tf.expand_dims(tf.expand_dims(model.HT, 0), 3)*x
    x = tf_ifftshift(model, tf.ifft3d(x))
    return tf.real(x)
        
    
####### Soft Thresholding Functions  #####
def soft_2d_gradient(v, h, tau):
    mag = tf.sqrt(v*v + h*h)
    magt = tf.maximum(mag - tau, 0)
    mag = tf.maximum(mag - tau, 0) + tau

    mmult = magt/mag

    return v*mmult, h*mmult

def soft_2d_gradient2(model, v,h,tau):

    vv = tf.concat([v, tf.zeros((model.batch_size, 1, model.DIMS1*2))] , 1)
    hh = tf.concat([h, tf.zeros((model.batch_size, model.DIMS0*2, 1))] , 2)

    mag = tf.sqrt(vv*vv + hh*hh)
    magt = tf.maximum(mag - tau, 0)
    mag = tf.maximum(mag - tau, 0) + tau
    mmult = magt/mag

    return v*mmult[:, :-1,:], h*mmult[:, :,:-1]

def soft_2d_gradient2_rgb(model, v,h,tau):

    vv = tf.concat([v, tf.zeros((model.batch_size, 1, model.DIMS1*2, 3))] , 1)
    hh = tf.concat([h, tf.zeros((model.batch_size, model.DIMS0*2, 1, 3))] , 2)

    mag = tf.sqrt(vv*vv + hh*hh)
    magt = tf.maximum(mag - tau, 0)
    mag = tf.maximum(mag - tau, 0) + tau
    mmult = magt/mag

    return v*mmult[:, :-1,:], h*mmult[:, :,:-1]

def soft_thresh(x, beta):
    res = tf.maximum(tf.abs(x)-beta,0)*tf.sign(x)
    return res

####### Add Noise #####
def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise
    
######## ADMM Parameter Update #########
def param_update2(mu, res_tol, mu_inc, mu_dec, r, s):
    mu_up = tf.cond(tf.greater(r, res_tol * s), lambda: (mu * mu_inc), lambda: mu)
    mu_up = tf.cond(tf.greater(s, res_tol * r), lambda: (mu_up/mu_dec), lambda: mu_up)
    
    return mu_up
    
###### Things I saw on TV ###########
def make_laplacian(model):
    lapl = np.zeros([model.DIMS0*2,model.DIMS1*2])
    lapl[0,0] =4.; 
    lapl[0,1] = -1.; lapl[1,0] = -1.; 
    lapl[0,-1] = -1.; lapl[-1,0] = -1.; 

    LTL = np.abs(np.fft.fft2(lapl))
    return LTL


def DT(dx, dy):  # Use convolution instead?  
        with tf.device("/cpu:0"):
            out = (tf.manip.roll(dx, 1, axis = 1) - dx) + (tf.manip.roll(dy, 1, axis = 2) - dy)
        return out

def D(x):
    with tf.device("/cpu:0"):
        xroll = tf.manip.roll(x, -1, axis = 1)
        yroll = tf.manip.roll(x, -1, axis = 2)
    return (xroll - x), (yroll - x)
    
    
def L_tf(a): # Not using
    xdiff = a[:, 1:, :]-a[:, :-1, :]
    ydiff = a[:, :, 1:]-a[:, :, :-1]
    return -xdiff, -ydiff

def Ltv_tf(a, b): # Not using
    return tf.concat([a[:,0:1,:], a[:, 1:, :]-a[:, :-1, :], -a[:,-1:,:]], axis = 1) + tf.concat([b[:,:,0:1], b[:, :, 1:]-b[:, :,  :-1], -b[:,:,-1:]], axis = 2)


def TVnorm_tf(x):
    x_diff, y_diff = L_tf(x)
    result = tf.reduce_sum(tf.abs(x_diff)) + tf.reduce_sum(tf.abs(y_diff))
    return result

    


def admm_rgb(model, in_vars, alpha2k_1, alpha2k_2, CtC, Cty, mu_auto, n, y):  
    
    sk = in_vars[0];  alpha1k = in_vars[1]; alpha3k = in_vars[2]
    Hskp = in_vars[3]; 
    
    #alpha2k_1 = in_vars[4];  alpha2k_2 = in_vars[5]
    
    if model.autotune == True:
        mu1 = mu_auto[0];  mu2 = mu_auto[1];  mu3 = mu_auto[2]

    else:
        mu1 = model.mu_vals[0][n];  mu2 = model.mu_vals[1][n];  mu3 = model.mu_vals[2][n]
        
    tau = model.mu_vals[3][n]
    
    dual_resid_s = [];  primal_resid_s = []
    dual_resid_u = [];  primal_resid_u = []
    dual_resid_w = []
    primal_resid_w = []
    cost = []

    Smult = tf.expand_dims(tf.expand_dims(1/(mu1*model.HtH + mu2*model.LtL + mu3), 0), 3)
    Vmult = 1/(CtC + mu1)
    
    ###############  update u = soft(Ψ*x + η/μ2,  tau/μ2) ###################################
    
    Lsk1, Lsk2 = L_tf(sk)

    ukp_1, ukp_2 = soft_2d_gradient2_rgb(model, Lsk1 + alpha2k_1/mu2, Lsk2 + alpha2k_2/mu2, tau)
    
    ################  update      ######################################
    vkp = Vmult*(mu1*(alpha1k/mu1 + Hskp) + Cty)

    ################  update w <-- max(alpha3/mu3 + sk, 0) ######################################
    wkp = tf.maximum(alpha3k/mu3 + sk, 0)
   
    if 'prox' in model.learning_options['learned_vars']:   # use learned prox
        ista_res, symm = learned_prox_rgb(model, sk, n)
        skp_numerator = mu3*(wkp - alpha3k/mu3) + mu1 * Hadj_rgb(model, vkp - alpha1k/mu1) + mu2*ista_res
    
    else:   # no learned prox 
        skp_numerator = mu3*(wkp - alpha3k/mu3) + mu1 * Hadj_rgb(model, vkp - alpha1k/mu1) + mu2*Ltv_tf(ukp_1 - alpha2k_1/mu2, ukp_2 - alpha2k_2/mu2) 

        symm = []

    skp = tf.real(tf.ifft3d(tf.complex(Smult, tf.zeros_like(Smult)) * tf.fft3d(tf.complex(skp_numerator, tf.zeros_like(skp_numerator)))))
    
    Hskp_up = Hfor_rgb(model, skp)
    r_sv = Hskp_up - vkp
    dual_resid_s.append(mu1 * tf.linalg.norm(Hskp - Hskp_up))
    primal_resid_s.append(tf.linalg.norm(r_sv))

    # Autotune
    if model.autotune == True:
        mu1_up = param_update2(mu1, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_s[-1], dual_resid_s[-1])

    else: 
        if n == model.iterations-1:
            mu1_up = model.mu_vals[0][n]
        else:
            mu1_up = model.mu_vals[0][n+1]

    alpha1kup = alpha1k + mu1*r_sv

    Lskp1, Lskp2 = L_tf(skp)
    r_su_1 = Lskp1 - ukp_1
    r_su_2 = Lskp2 - ukp_2

    dual_resid_u.append(mu2*tf.sqrt(tf.linalg.norm(Lsk1 - Lskp1)**2 + tf.linalg.norm(Lsk2 - Lskp2)**2))
    primal_resid_u.append(tf.sqrt(tf.linalg.norm(r_su_1)**2 + tf.linalg.norm(r_su_2)**2))

    if model.autotune == True:
        mu2_up = param_update2(mu2, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_u[-1], dual_resid_u[-1])
    else:
        if n == model.iterations-1:
            mu2_up = model.mu_vals[1][n]
        else:
            mu2_up = model.mu_vals[1][n+1]

    alpha2k_1up= alpha2k_1 + mu2*r_su_1
    alpha2k_2up= alpha2k_2 + mu2*r_su_2

    r_sw = skp - wkp
    dual_resid_w.append(mu3*tf.linalg.norm(sk - skp))
    primal_resid_w.append(tf.linalg.norm(r_sw))

    if model.autotune == True:
        mu3_up = param_update2(mu3, model.resid_tol, model.mu_inc, model.mu_dec, primal_resid_w[-1], dual_resid_w[-1])
    else:
        if n == model.iterations-1:
            mu3_up = model.mu_vals[2][n]
        else:
            mu3_up = model.mu_vals[2][n+1]

    alpha3kup = alpha3k + mu3*r_sw

    data_loss = tf.linalg.norm(crop(model, Hskp_up)-y)**2
    tv_loss = tau*TVnorm_tf(skp)

    
    if model.printstats == True:
        
        admmstats = {'dual_res_s': dual_resid_s[-1].numpy(),
                     'primal_res_s':  primal_resid_s[-1].numpy(),
                     'dual_res_w':dual_resid_w[-1].numpy(),
                     'primal_res_w':primal_resid_w[-1].numpy(),
                     'dual_res_u':dual_resid_s[-1].numpy(),
                     'primal_res_u':primal_resid_s[-1].numpy(),
                     'data_loss':data_loss.numpy(),
                     'total_loss':(data_loss+tv_loss).numpy()}
        
        
        print('\r',  'iter:', n,'s:', dual_resid_s[-1].numpy(), primal_resid_s[-1].numpy(), 
         'u:',  dual_resid_u[-1].numpy(), primal_resid_u[-1].numpy(),
          'w:', dual_resid_w[-1].numpy(), primal_resid_w[-1].numpy(), end='')
    else:
        admmstats = []

    

    out_vars = tf.stack([skp, alpha1kup, alpha3kup, Hskp_up])

 
    mu_auto_up = tf.stack([mu1_up, mu2_up, mu3_up])
    
    return out_vars, alpha2k_1up, alpha2k_2up, mu_auto_up, symm, admmstats




class Model(tf.keras.Model):
    def __init__(self, batch_size, h, iterations, learning_options = {'learned_vars': []}, 
                 init_vars = {'initialize_vars': False}):
        super().__init__() #super(Model, self).__init__()
        
        self.iterations = iterations              # Number of unrolled iterations
        self.batch_size = batch_size              # Batch size 
        self.autotune = False                     # Using autotune (True or False)
        self.realdata = True                      # Real Data or Simulated Measurements
        self.printstats = False                   # Print ADMM Variables
        self.init_vars = init_vars
        
        self.addnoise = False                     # Add noise (only if using simulated data)
        self.noise_std = 0.05                     # Noise standard deviation 
        
        
        # Leared structure options   
        self.learning_options = learning_options

        
        ## Initialize constants 
        self.DIMS0 = h.shape[0]  # Image Dimensions
        self.DIMS1 = h.shape[1]  # Image Dimensions
        
        self.PAD_SIZE0 = int((self.DIMS0)//2)                           # Pad size
        self.PAD_SIZE1 = int((self.DIMS1)//2)                           # Pad size
        
        # Initialize Variables 
        self.initialize_learned_variables(learning_options)
        
        if 'h' in learning_options['learned_vars']: 
            if init_vars['initialize_vars'] == True:
                self.h_var = tfe.Variable(init_vars['h'], name = 'h', dtype = tf.float32, constraint=lambda t: tf.clip_by_value(t, 0., 1.))
            else:
                self.h_var = tfe.Variable(h, name = 'h', dtype = tf.float32, constraint=lambda t: tf.clip_by_value(t, 0., 1.))
                
            self.H = tf.fft2d(tf_ifftshift2(self, tf.complex(pad(self, self.h_var), tf.zeros([self.DIMS0*2, 
                                                                              self.DIMS1*2], dtype=tf.float32))))
        else:
            self.H = tf.fft2d(tf_ifftshift2(self, tf.complex(pad(self, h.astype('float32')), tf.zeros([self.DIMS0*2, 
                                                                              self.DIMS1*2], dtype=tf.float32))))
        self.H_adj = tf.conj(self.H)
        self.HT = tf.conj(self.H)
        self.HtH = tf.abs(self.H*self.HT)
        
        self.LtL = tf.constant(make_laplacian(self), dtype = tf.float32) 
        
        self.resid_tol = tf.constant(1.5, dtype = tf.float32)
        self.mu_inc = tf.constant(1.2, dtype = tf.float32)
        self.mu_dec = tf.constant(1.2, dtype = tf.float32)




    def initialize_learned_variables(self, learning_options):
        
        if 'mus' in learning_options['learned_vars']:  # Make mu parameters learnable
            if self.init_vars['initialize_vars'] == True:
                self.mu1= tfe.Variable(self.init_vars['mu1'], name = 'mu1', dtype = tf.float32)
                self.mu2= tfe.Variable(self.init_vars['mu2'], name = 'mu2', dtype = tf.float32)
                self.mu3= tfe.Variable(self.init_vars['mu3'], name = 'mu3', dtype = tf.float32)
            else:
                self.mu1= tfe.Variable(np.ones(self.iterations)*1e-4, name = 'mu1', dtype = tf.float32)
                self.mu2= tfe.Variable(np.ones(self.iterations)*1e-4, name = 'mu2', dtype = tf.float32)
                self.mu3= tfe.Variable(np.ones(self.iterations)*1e-4, name = 'mu3', dtype = tf.float32)
        else:                                          # Not learnable
            self.mu1=  tf.ones(self.iterations, dtype = tf.float32)*1e-4
            self.mu2=  tf.ones(self.iterations, dtype = tf.float32)*1e-4
            self.mu3 = tf.ones(self.iterations, dtype = tf.float32)*1e-4


        if 'tau' in learning_options['learned_vars']:  # Make tau parameter learnable
            if self.init_vars['initialize_vars'] == True:
                self.tau= tfe.Variable(self.init_vars['tau'], name = 'tau', dtype = tf.float32)
            else:
                self.tau= tfe.Variable(np.ones(self.iterations)*2e-3, name = 'tau', dtype = tf.float32)
        else:
            self.tau= tf.ones(self.iterations, dtype = tf.float32)*2e-3 
    
    
        if 'prox' in learning_options['learned_vars'] or 'network_denoiser' in learning_options['learned_vars']:
            filt_size = learning_options['filter_depth']
            sz = learning_options['filt_size']
            scaling = learning_options['init_scaling']

            if learning_options['shared_weights'] == True:
                conv_filter_shape1 = [1, sz, sz, 3, filt_size]
                conv_filter_shape = [1, sz, sz, filt_size, filt_size]
                conv_filt_shape_last = [1, sz, sz, filt_size, 3]
            else:
                conv_filter_shape1 = [self.iterations, sz, sz, 3, filt_size]
                conv_filter_shape = [self.iterations, sz, sz, filt_size, filt_size]
                conv_filt_shape_last = [self.iterations, sz, sz, filt_size, 3]
                
        if 'prox' in learning_options['learned_vars']:  # Make learnable prox function 
            if learning_options['prox_type']== 'istanet':
                self.soft_thr = tfe.Variable(np.ones(self.iterations)*0.1, dtype=tf.float32)

                self.weights0 = tfe.Variable(
                        np.random.normal(size = conv_filter_shape1).astype('float32')*scaling, name='istanet_w0')
                self.weights1 = tfe.Variable(
                        np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='istanet_w1')
                self.weights2 = tfe.Variable(
                        np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='istanet_w2')
                self.weights3 = tfe.Variable(
                        np.random.normal(size = conv_filt_shape_last).astype('float32')*scaling, name='istanet_w3')

            elif learning_options['prox_type']== 'istanet_res':
                self.soft_thr = tfe.Variable(np.ones(self.iterations)*0.1, dtype=tf.float32)

                self.weights0 = tfe.Variable(
                    np.random.normal(size = conv_filter_shape1).astype('float32')*scaling, name='istanet_prox_w0')
                self.weights1 = tfe.Variable(
                    np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='istanet_prox_w1')
                self.weights2 = tfe.Variable(
                    np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='istanet_prox_w2')
                self.weights3 = tfe.Variable(
                    np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='istanet_prox_w3')
                self.weights4 = tfe.Variable(
                    np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='istanet_prox_w4')
                self.weights5 = tfe.Variable(
                    np.random.normal(size = conv_filt_shape_last).astype('float32')*scaling, name='istanet_prox_w5')
                
            elif learning_options['prox_type'] == 'test':
                
                self.scale1 = tfe.Variable(tf.ones(self.iterations)*.001, dtype = tf.float32)
                self.beta1 = tfe.Variable(tf.zeros(self.iterations), dtype = tf.float32)
                self.epsilon = 1e-3

                self.weights0 = tfe.Variable(
                    np.random.normal(size = conv_filter_shape1).astype('float32')*scaling, name='istanet_prox_w0')
                self.weights1 = tfe.Variable(
                    np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='istanet_prox_w1')
                self.weights2 = tfe.Variable(
                    np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='istanet_prox_w2')
                self.weights3 = tfe.Variable(
                    np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='istanet_prox_w3')
                self.weights4 = tfe.Variable(
                    np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='istanet_prox_w4')
                self.weights5 = tfe.Variable(
                    np.random.normal(size = conv_filt_shape_last).astype('float32')*scaling, name='istanet_prox_w5')




        if 'network_denoiser' in learning_options['learned_vars']:  # Make learnable prox function 
                self.weights0s = tfe.Variable(
                    np.random.normal(size = conv_filter_shape1).astype('float32')*scaling, name='weights0s')
                self.weights1s = tfe.Variable(
                    np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='weights1s')
                self.weights2s = tfe.Variable(
                    np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='weights2s')
                self.weights3s = tfe.Variable(
                    np.random.normal(size = conv_filter_shape).astype('float32')*scaling, name='weights3s')



    def call(self, inputs):    
        
        self.mu_vals = tf.stack([self.mu1, self.mu2, self.mu3, self.tau])
        
        self.admmstats = {'dual_res_s': [], 'dual_res_u': [], 'dual_res_w': [], 
             'primal_res_s': [], 'primal_res_u': [], 'primal_res_w': [],
             'data_loss': [], 'total_loss': []}
        
        if self.autotune==True:
            self.mu_auto_list = {'mu1': [], 'mu2': [], 'mu3': []}
        
        
        
        # If using simulated data, input the raw image and run through forward model
        if self.realdata == False: 
            y = crop(self, self.Hfor(pad_dim2(self, inputs)))
            if self.addnoise == True:
                y = self.gaussian_noise_layer(y, self.noise_std)
        
        # Otherwise, input is the normalized Diffuser Image 
        else:
            y = inputs/tf.linalg.norm(inputs)
        
            
        Cty = pad_dim3_rgb(self, y)                   # Zero padded input
        CtC = pad_dim3_rgb(self, tf.ones_like(y))     # Zero padded ones 
        
        # Create list of inputs/outputs         
        in_vars = []; in_vars1 = []
        in_vars2 = []; Hsk_list = []
        a2k_1_list=[]; a2k_2_list= []

        sk = tf.zeros((self.batch_size, self.DIMS0*2, self.DIMS1*2, 3), dtype = tf.float32)
        alpha1k = tf.zeros((self.batch_size, self.DIMS0*2, self.DIMS1*2, 3), dtype = tf.float32)
        alpha3k = tf.zeros((self.batch_size, self.DIMS0*2, self.DIMS1*2, 3), dtype = tf.float32)
        Hskp = tf.zeros_like(sk)
        #alpha2k_1 = tf.zeros_like(sk)
        #alpha2k_2  = tf.zeros_like(sk)
        
        alpha2k_1 = tf.zeros_like(sk[:,:-1,:])  
        alpha2k_2 = tf.zeros_like(sk[:,:,:-1])
        
        mu_auto = tf.stack([self.mu1[0], self.mu2[0], self.mu3[0], self.tau[0]])
        #mu_auto = tf.stack([tf.exp(self.mu1[0]), tf.exp(self.mu2[0]), tf.exp(self.mu3[0]), tf.exp(self.tau[0])])
    
        #in_vars.append(tf.stack([sk, alpha1k, alpha3k, Hskp, alpha2k_1, alpha2k_2]))
        in_vars.append(tf.stack([sk, alpha1k, alpha3k, Hskp]))
        
        a2k_1_list.append(alpha2k_1)
        a2k_2_list.append(alpha2k_2)
        

        for i in range(0,self.iterations):
            # print(i, self.iterations)
            if 'network_denoiser' in self.learning_options['learned_vars']:

                out_vars, _ , symm, admmstats= admm_rgb(self, in_vars[-1], i, CtC, Cty, [], i)
                
                out_vars_residual = residual_cnn(self, out_vars[0], i)
                out_final = tf.stack([out_vars_residual, out_vars[1], out_vars[2], out_vars[3], out_vars[4], out_vars[5]])

                in_vars.append(out_final)


            else:
                if self.autotune==True:
                    #out_vars, mu_auto, symm = admm(self, in_vars[-1], i, CtC, Cty, mu_auto, i)
                    out_vars, a_out1, a_out2, mu_auto , symm, admmstats= admm_rgb(self, in_vars[-1], 
                                                              a2k_1_list[-1], a2k_2_list[-1], CtC, Cty, mu_auto, i, y)
                    
                    self.mu_auto_list['mu1'].append(mu_auto[0])
                    self.mu_auto_list['mu2'].append(mu_auto[1])
                    self.mu_auto_list['mu3'].append(mu_auto[2])

                else:
                    #out_vars, _ , symm = admm(self, in_vars[-1], i, CtC, Cty, [], i)
                    out_vars, a_out1, a_out2, _ , symm, admmstats = admm_rgb(self, in_vars[-1], 
                                                              a2k_1_list[-1], a2k_2_list[-1], CtC, Cty, [], i, y)

                in_vars.append(out_vars)
                a2k_1_list.append(a_out1)
                a2k_2_list.append(a_out2)

                if self.printstats == True:                   # Print ADMM Variables
                    self.admmstats['dual_res_s'].append(admmstats['dual_res_s'])
                    self.admmstats['primal_res_s'].append(admmstats['primal_res_s'])
                    self.admmstats['dual_res_w'].append(admmstats['dual_res_w'])
                    self.admmstats['primal_res_w'].append(admmstats['primal_res_w'])
                    self.admmstats['dual_res_u'].append(admmstats['dual_res_u'])
                    self.admmstats['primal_res_u'].append(admmstats['primal_res_u'])
                    self.admmstats['data_loss'].append(admmstats['data_loss'])
                    self.admmstats['total_loss'].append(admmstats['total_loss'])
                
                
            x_out = crop(self, in_vars[-1][0])
            

        return x_out, symm

        

    
    
