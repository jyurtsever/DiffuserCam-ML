
import numpy as np
import scipy.io as io
import scipy.misc
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
import math
from ipywidgets import interact, widgets
import cv2
from DiffuserCamUtils import *
import sys, os


####        INITIALIZE        #####
def initialize(image_file, psf_file, f_lat = 1, f_ax = 1, type = 'pco', color = 'rgb', dim = '2d', ind = 0):
    try:
        im_type = image_file[-3:]
        psf_type = psf_file[-3:]
        # image = np.array(np.load(image_file))
        # psf = np.array(np.load(psf_file))
        # image = np.array(Image.open(image_file))[:,:,ind].astype('float32')

        image = np.array(cv2.imread(image_file ,-1))[: ,: ,ind].astype('float32')

        psf = io.loadmat(psf_file)['psf'] if psf_type == 'mat' else rgb2gray \
            (np.array(cv2.imread(psf_file ,-1)).astype('float32'))
    except IOError as ex:
        print("I/O error: " + str(ex.strerror))
    if dim == '2d':  # embed into a 3d array
        psf = np.expand_dims(psf, 2)
    psf_bg = np.mean(psf[0 : 15, 0 : 15])  # 102
    image_bg = np.mean(image[0 : 15, 0 : 15])  # should be around 100

    psf_down = downsample_ax(psf - psf_bg, f_lat)
    image = downsample_ax(image -bg_val, f_lat)

    if dim == '3d':
        psf_down = downsample_lat(psf_down, f_ax)

    image /= np.max(image)
    psf_down /= norm(psf_down)

    return psf_down, image


####        FFT         #####
def fft2c(x):
    return 1 / np.sqrt(np.prod(x.shape)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))


def ifft2c(y):
    return np.sqrt(np.prod(y.shape)) * np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(y)))


### RGB to GRAY  ###
def rgb2gray(rgb):
    return np.dot(rgb[... ,:3], [0.299, 0.587, 0.114])


####        GRADIENT and ERROR        #####
grad_func = lambda x : grad(x, A, AH, b)
error = lambda x : objective(x, A, b, opt.tau)

####        Transform Groundtruth     ######
def transform_groundtruth(gt_image, M, ds, mtx, dist):
    gt_out = cv2.undistort(gt_image, mtx, dist)
    gt_out = np.flipud(downsample_ax(gt_out, ds))
    gt_out = cv2.warpAffine(gt_out, M, (gt_out.shape[1], gt_out.shape[0]))

    return gt_out


# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__


####  RECONSTRUCTS 'IMAGE_FILE' PATH AND SAVES TO 'SAVE_FILE' PATH ###
def reconstruct_and_save(image_file, gt_file, save_file_diffuser, save_file_crop_diffuser, save_file_gt, max_itr):
    x_rgb = np.zeros((h.shape[0], h.shape[1], 3))
    opt.max_itr = max_itr

    for i in range(0, 3):
        # psf, b = initialize(image_file, psf_file, dim = dimensions, f_lat = 1, ind = i)
        psf, b = initialize(image_file, psf_file, dim=dimensions, f_lat=4, ind=i)

        opt.b = b

        grad_func = lambda x: grad(x, A, AH, b)
        error = lambda x: objective(x, A, b, opt.tau)

        x, error_list = solver(grad_func, error, non_negative, opt)
        # x = crop2d(x)
        x_rgb[:, :, i] = x[:, :, 0]

    scipy.misc.imsave(save_file_diffuser, np.flipud(x_rgb))
    scipy.misc.imsave(save_file_crop_diffuser, crop2d(np.flipud(x_rgb)))

    # gt_image = cv2.imread(gt_file, -1).astype('float32')
    # gt_out = transform_groundtruth(gt_image, M, ds, mtx, dist)
    # scipy.misc.imsave(save_file_gt, np.flipud(gt_out))
    return x_rgb


#####  ITERATES THROUGH
def run_recon_and_crop(num_photos, max_itr, im_path, gt_path, save_path, start=1):
    for i in range(start, start + num_photos):
        enablePrint()
        print('photo ', i, ' of ', num_photos + start, '\n\n', end="\r")
        blockPrint()
        im_file = im_path + 'im' + str(i) + '.jpg.tiff'
        gt_file = gt_path + 'im' + str(i) + '.jpg.tiff'
        save_file_diffuser = save_path + 'recon/im' + str(i) + '.tiff'
        crp_path = save_path + 'recon_cropped/im' + str(i) + '.tiff'
        save_file_gt = save_path + 'gt/im' + str(i) + '.tiff'
        reconstruct_and_save(im_file, gt_file, save_file_diffuser, crp_path, save_file_gt, max_itr)


def run_recon_and_crop_gt(num_photos, gt_path, save_path, start=1):
    for i in range(start, start + num_photos):
        enablePrint()
        print('photo ', i, ' of ', num_photos + start, '\n\n', end="\r")
        blockPrint()
        gt_file = gt_path + 'im' + str(i) + '.jpg.tiff'
        save_file_gt = save_path + 'gt/im' + str(i) + '.tiff'
        transform_and_save_gt(gt_file, save_file_gt)


def transform_and_save_gt(gt_file, save_file_gt):
    gt_image = cv2.imread(gt_file, -1).astype('float32')
    gt_out = transform_groundtruth(gt_image, M, ds, mtx, dist)
    scipy.misc.imsave(save_file_gt, np.flipud(gt_out))


def main():
    run_recon_and_crop(num_photos, num_iters, im_path, gt_path, save_file_path, start=start)



if __name__ == '__main__':
    ###params###
    num_photos = sys.argv[1]
    num_iters = sys.argv[2]
    start = sys.argv[3]

    ## Background ###
    bg_file = '../recon_files/diffuser_background.tiff';
    bg_image = np.array(cv2.imread(bg_file, -1)).astype('float32')

    #####   Diffuser Image     #####
    image_file = '../mirflickr25k/diffuser_raw/im333.jpg.tiff'
    im_path = '../mirflickr25k/diffuser_raw/'

    #####    GT IMAGE     #####
    gt_path = '../mirflickr25k/gt'

    #####   SAVE FILE    #####
    save_file_path = '../mirflickr25k_recon/'

    ####       PSF       ####
    psf_file = '../recon_files/psf_white_LED_Nick.tiff'
    # psf_file = 'D:/Kristina/2_8_2019/greenpsf.bmp'


    #####        PARAMETERS         #####
    dimensions = '2d'

    bg_val = np.mean(bg_image)
    psf, b = initialize(image_file, psf_file, dim=dimensions, f_lat=4, ind=0)
    r = 0  # proportion of pixels to crop out
    num_cropped = int(r * b.size)
    crop2d, crop3d, pad2d, pad3d, pix_crop = get_crop_pad(psf, N=num_cropped)
    h = pad2d(psf)  # pad the input stack of h's

    obj_shape = h.shape
    up_shape = psf.shape
    A, AH = get_ops(h, crop2d, pad2d, crop3d, pad3d, up_shape)

    alg = 'admm'
    max_itr = 100
    opt = Options(dimensions, alg, max_itr)
    # opt.gamma = np.real(1.8 / np.max(Hstar * H))
    opt.gamma = 1
    # opt.eps = 7.4e-3        #7.4e-3 for nesterov, 4e-3 for fista
    opt.del_pixels = True
    opt.psf = h
    opt.b = b
    opt.crop2d, opt.pad2d = crop2d, pad2d
    opt.crop3d, opt.pad3d = crop3d, pad3d
    opt.up_shape, opt.pad_shape = up_shape, obj_shape
    opt.autotune = True
    opt.beta = 1.1
    opt.alpha = 1.01

    # tune regularization parameters.
    # 2d tuning: default to 1e-4 on each, tau = 2e-3
    # cartoony: 1e-3, 5e-2, 1e-3, 2e-5
    # to actually see cost function going down, use tau = 1, other mu's = 1e-4
    opt.mu1 = 1e-4
    opt.mu2 = 1e-4
    opt.mu3 = 1e-4
    opt.tau = 2e-3

    ### H and XHAT ###

    H = fft2c(psf[:, :, 0])
    Xhat = ifft2c(fft2c(b) / H)

    ####  CALIBRATION #####
    calibration_2_15 = io.loadmat('../recon_files/calibration_2mv _15_v2.mat')

    M = calibration_2_15['M']
    mtx = calibration_2_15['mtx']
    dist = calibration_2_15['dist']
    ds = calibration_2_15['ds']
    main()