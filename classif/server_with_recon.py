import sys
sys.path.append('./models/')
import pickle
import socket
import os
import torchvision
import torch.nn as nn
import numpy as np
import requests
import skimage
import argparse
import imagiz
import torch
import cv2
import json
import struct
import time
import admm_model as admm_model_plain

from utils import load_psf_image, preplot
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict

HOST = ''
IMG_PORT = 8098
ARR_PORT = 8097
# ARR_PORT_2 = 8099
# body_parts = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
#               'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle',
#               'Left Hip', 'Left Knee', 'LAnkle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Background']


# def initialize(frame, flip=False):
#     return trans(frame)

def net_forward(frame):
    image = Image.fromarray(frame)
    image = trans(image).view(1, 3, 224, 224)
    if use_gpu:
        image = image.cuda()

    #print(image.shape)
    out = net(image).cpu() #.detach().numpy()
    return out #[0]


def get_classes(out):
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0]
    return [(classes[idx.item()], round(percentage[idx.item()].item(),2)) for idx in indices[0][:7]]


def get_recon(frame):
    frame_float = (frame/np.max(frame)).astype('float32') 
    perm = torch.tensor(frame_float.transpose((2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        inputs = perm.to(my_device)
        out = admm_converged2(inputs)

    return preplot(out[0].cpu().detach().numpy())


def main():
    print("Connecting...")
    server = imagiz.Server(port=IMG_PORT)
    print("Connected...")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    while True:
        try:
            message = server.receive()
            frame = cv2.imdecode(message.image,1)

            out = net_forward(frame)
            recon = get_recon(frame)
            recon = (recon*255).astype('uint8')
            r, recon_encode = cv2.imencode('.jpg', recon, encode_param)

            ###Send
            img_data_string = pickle.dumps(recon_encode)
            conn.sendall(struct.pack(">L", len(img_data_string)) + img_data_string)

            picked_classes = get_classes(out)
            print(picked_classes)
            #
            # data_string = pickle.dumps(picked_classes)
            #
            # ###Send
            # conn.send(data_string)

            cv2.waitKey(1)
        except KeyboardInterrupt:
            s.close()
            cv2.destroyAllWindows()
            break
    print("\nSession Ended")

def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str)
    parser.add_argument("-imagenet_train_dir", type=str,
                         default='/home/jyurtsever/research/sim_train/data/imagenet_forward/train')
    parser.add_argument("-psf_file", type=str, default= '../../recon_files/psf_white_LED_Nick.tiff')
    args = parser.parse_args()


    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print('Socket created')

    s.bind((HOST, ARR_PORT))

    print('Socket bind complete')

    # Read the network from Memory
    print("Initializing Model")
    net = models.resnet18(num_classes=1000)
    checkpoint = torch.load(args.model_dir)
    #print(checkpoint.keys())
    net.load_state_dict(fix_state_dict(checkpoint['state_dict']))

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()
    net.eval()

    trans = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # from http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    ])

    # $url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/' \
    #       'raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt'
    class_info_json_filepath = './imagenet_class_info.json'
    with open(class_info_json_filepath) as class_info_json_f:
        class_info_dict = json.load(class_info_json_f)

    class_wnids = os.listdir(args.imagenet_train_dir)
    class_wnids.sort()
    classes = [class_info_dict[class_wnid]["class_name"] for class_wnid in class_wnids]

    print("Model created")

    print("Creating Recon Model")
    my_device = 'cuda:0'

    psf_diffuser = load_psf_image(args.psf_file, downsample=1, rgb=False)

    ds = 4  # Amount of down-sampling.  Must be set to 4 to use dataset images

    print('The shape of the loaded diffuser is:' + str(psf_diffuser.shape))

    psf_diffuser = np.sum(psf_diffuser, 2)

    h = skimage.transform.resize(psf_diffuser,
                                 (psf_diffuser.shape[0] // ds, psf_diffuser.shape[1] // ds),
                                 mode='constant', anti_aliasing=True)

    var_options = {'plain_admm': [],
                   'mu_and_tau': ['mus', 'tau'],
                   }

    learning_options_none = {'learned_vars': var_options['plain_admm']}

    admm_converged2 = admm_model_plain.ADMM_Net(batch_size=1, h=h, iterations=10,
                                                learning_options=learning_options_none, cuda_device=my_device)

    admm_converged2.tau.data = admm_converged2.tau.data * 1000
    admm_converged2.to(my_device)
    print("Recon Model Created")
    s.listen(1)
    print('Socket now listening')
    conn, addr = s.accept()





    main()
