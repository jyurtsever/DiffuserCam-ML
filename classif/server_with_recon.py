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




class Ensemble(nn.Module):
    def __init__(self, denoiser, classifier):
        super(Ensemble, self).__init__()
        self.denoiser = denoiser
        self.classifier = classifier

    def forward(self, x):
        recon  = self.denoiser(x)
        out = self.classifier(recon)
        return out, recon

    def to(self, indevice):
        super().to(indevice)
        self.denoiser.to(indevice)
        self.denoiser.h_var.to(indevice)
        self.denoiser.h_zeros.to(indevice)
        self.denoiser.h_complex.to(indevice)
        self.denoiser.LtL.to(indevice)
        return self

def net_forward(frame):
    image = Image.fromarray(frame)
    image = trans(image).view(1, 3, 224, 224)
    if use_gpu:
        image = image.cuda()

    out = classifier(image).cpu()
    return out


def get_classes(out):
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0]
    return [(classes[idx.item()], round(percentage[idx.item()].item(),2)) for idx in indices[0][:7]]

def ensemble_forward(frame):
    input_ = get_admm_input(frame)
    if use_gpu:
        input_ = input_.cuda()

    out, recon = model(input_)
    recon = recon[0].cpu().detach()
    recon = ((preplot(recon.numpy())*255).astype('uint8'))[...,::-1] #used to flip
    out = out.cpu()
    return out, recon


def get_admm_input(frame):
    frame_norm = frame.astype(np.float32)/255
    frame_float = frame_norm.astype('float32') #(frame/np.max(frame)).astype('float32') 
    perm = torch.tensor(frame_float.transpose((2, 0, 1))).unsqueeze(0)
    return perm

def get_recon(frame):

    frame_norm = frame.astype(np.float32)/255
    frame_float = frame_norm.astype('float32') #(frame/np.max(frame)).astype('float32') 
    perm = torch.tensor(frame_float.transpose((2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        inputs = perm.to(my_device)
        out = model(inputs)[0].cpu().detach()
    return np.flipud((preplot(out.numpy())*255).astype('uint8'))[...,::-1]


def main():
    print("Connecting...")
    server = imagiz.Server(port=IMG_PORT)
    print("Connected...")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    while True:
        try:
            message = server.receive()
            frame = cv2.cvtColor(cv2.imdecode(message.image, 1), cv2.IMREAD_COLOR)#[:, :, ::-1]
            if not args.use_ensemble:
                recon = get_recon(frame)

            #reconstruct before running classifier
            if args.use_recon:
                out = net_forward(recon)

            #use ensemble model
            elif args.use_ensemble:
                out, recon = ensemble_forward(frame)

            #use raw diffuser
            else:
                out = net_forward(frame)
            #print(recon)
            r, recon_encode = cv2.imencode('.jpg', recon, encode_param)

            ###Send
            img_data_string = pickle.dumps(recon_encode)
            conn.sendall(struct.pack(">L", len(img_data_string)) + img_data_string)

            picked_classes = get_classes(out)
            print(picked_classes)

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
                         default='/home/jyurtsever/research/sim_train/data/imagenet_forward_2/train')
    parser.add_argument("-psf_file", type=str, default= '../../recon_files/psf_white_LED_Nick.tiff')
    parser.add_argument("-recon_iters", type=int, default=10)
    parser.add_argument("-use_recon", dest='use_recon', action='store_true')
    parser.add_argument("-use_le_admm", dest='use_le_admm', action='store_true')
    parser.add_argument('-use_ensemble', dest='use_ensemble', action='store_true')
    args = parser.parse_args()


    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print('Socket created')

    s.bind((HOST, ARR_PORT))

    print('Socket bind complete')

    # Read the network from Memory
    print("Initializing Model")
    classifier = models.resnet18(num_classes=1000)
    checkpoint = torch.load(args.model_dir)
    #print(checkpoint.keys())
    if not args.use_ensemble:
        classifier.load_state_dict(fix_state_dict(checkpoint['state_dict']))

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        classifier = classifier.cuda()
    classifier.eval()

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
    if args.use_le_admm:
        learning_options = {'learned_vars': var_options['mu_and_tau']}

    else:
        learning_options = {'learned_vars': var_options['plain_admm']}

    model = admm_model_plain.ADMM_Net(batch_size=1, h=h, iterations=args.recon_iters,
                                                learning_options=learning_options, cuda_device=my_device)

    if args.use_le_admm or args.use_ensemble:
        le_admm = torch.load('../../saved_models/model_le_admm.pt', map_location=my_device)
        le_admm.cuda_device = my_device
        for pn, pd in le_admm.named_parameters():
            for pnn, pdd in model.named_parameters():
                if pnn == pn:
                    pdd.data = pd.data

    model.tau.data = model.tau.data * 1000
    model.to(my_device)

    if args.use_ensemble:
        model = Ensemble(model, classifier)
        model.load_state_dict(fix_state_dict(checkpoint['state_dict']))

    print("Recon Model Created")
    s.listen(1)
    print('Socket now listening')
    conn, addr = s.accept()





    main()
