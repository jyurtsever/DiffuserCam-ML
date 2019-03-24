import tensorflow as tf

# tfe = tf.contrib.eager
import tensorflow.contrib.eager as tfe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import skimage
import scipy.io
from skimage.transform import rescale

from ADMM_tf_utils import *
#from admm_rgb import *
#from model_unrolled_layered import *

# import model_color as my_model_color
import progressbar
import scipy
import sys

import cv2 as cv



def main():
    csv_file_path = '../mirflickr25k/filenames.csv';
    csv_file_path_test = '../mirflickr25k/filenames.csv';

    train_batch_size = batch_size
    test_batch_size = batch_size

    dataset_train, len_train_dataset = make_dataset(csv_file_path, opts['down_sizing']);

    dataset_test, len_test_dataset = make_dataset(csv_file_path_test, opts['down_sizing']);

    dataset_test = dataset_test.batch(batch_size=test_batch_size)
    dataset_train = dataset_train.batch(batch_size=train_batch_size)

    dataset_test = dataset_test.repeat()
    dataset_train = dataset_train.repeat()

    model = Model(batch_size=train_batch_size, h=h, iterations=num_iters,
                  learning_options=learning_options1)  # my_model_color.
    model.autotune = True
    model.realdata = True
    model.noise_std = 0.0
    model.tau = model.tau * 1  # Regularization parameter



    if eager_enabled == False:
        sess = tf.Session()
        train_iterator = tf.data.Dataset.make_one_shot_iterator(dataset_train)
        test_iterator = tf.data.Dataset.make_one_shot_iterator(dataset_test)

        train_iterator_handle = sess.run(train_iterator.string_handle())
        test_iterator_handle = sess.run(test_iterator.string_handle())

        handle = tf.placeholder(tf.string, shape=[])

        iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types)

        next_element = iterator.get_next()

        image, label = sess.run(next_element, feed_dict={handle: train_iterator_handle})
        data_diffuser = tf.placeholder(tf.float32, shape=(None, DIMS0, DIMS1, 3))
        data_label = tf.placeholder(tf.float32, shape=(None, DIMS0, DIMS1, 3))

        diffuser_batch, label_batch = sess.run(next_element, feed_dict={handle: test_iterator_handle})
        feed_dict_train = {data_diffuser: diffuser_batch,
                           data_label: label_batch}
        sess.run(tf.global_variables_initializer())
        out_image_tf, symm = model(data_diffuser)
        sess.run(tf.global_variables_initializer())
        out_image = sess.run(out_image_tf, feed_dict_train)


    elif eager_enabled == True:
        bar = progressbar.ProgressBar(maxval=num_photos, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        i = 1
        bar.start()
        for diffuser_batch, label_batch in dataset_test:
            # print(i)
            if i < start:
                i += 1
                continue
            if i > start + num_photos:
                break
            bar.update(i)
            inputs = diffuser_batch
            labels = label_batch
            out_image, symm = model(inputs)
            for ind in range(test_batch_size):
                save_file_diffuser = save_path + 'im' + str(i) + '.tiff'
                im = preplot(out_image[ind] / np.max(out_image[ind]))
                scipy.misc.imsave(save_file_diffuser, im)
            i += ind + 1
        bar.finish()

if __name__ == '__main__':
    num_photos = int(sys.argv[1])
    num_iters = int(sys.argv[2])
    start = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    #####   SAVE FILE    #####
    save_path = sys.argv[5]
    learning_options1 = {'learned_vars': []}

    eager_enabled = True
    if eager_enabled == True:
        tf.enable_eager_execution()
    config = tf.ConfigProto()
    opts = {'psf_file': '../recon_files/psf_white_LED_Nick.tiff',
            'down_sizing': 4,
            }
    ##PSF###
    psf_diffuser = load_psf_image(opts['psf_file'], downsample=1, rgb=False)
    print('The shape of the loaded diffuser is:' + str(psf_diffuser.shape))

    psf_diffuser = np.sum(psf_diffuser, 2)

    ds = opts['down_sizing']

    h = skimage.transform.resize(psf_diffuser,
                                 (psf_diffuser.shape[0] // ds, psf_diffuser.shape[1] // ds),
                                 mode='constant', anti_aliasing=True)
    [DIMS0, DIMS1] = h.shape

    num_photos = int(sys.argv[1])
    num_iters = int(sys.argv[2])
    start = int(sys.argv[3])
    main()

