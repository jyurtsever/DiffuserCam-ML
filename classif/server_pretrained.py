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
import time
from torchvision import models, transforms
from PIL import Image

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
    return [(classes[idx.item()], round(percentage[idx.item()].item(),2)) for idx in indices[0][:3]]

def main():
    print("Connecting...")
    server = imagiz.Server(port=IMG_PORT)
    print("Connected...")
    while True:
        try:
            message = server.receive()
            frame = cv2.imdecode(message.image,1)

            out = net_forward(frame)
            picked_classes = get_classes(out)

            data_string = pickle.dumps(picked_classes)

            ###Send
            conn.send(data_string)

            cv2.waitKey(1)
        except KeyboardInterrupt:
            s.close()
            cv2.destroyAllWindows()
            break
    print("\nSession Ended")

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print('Socket created')

    s.bind((HOST, ARR_PORT))

    print('Socket bind complete')

    # Read the network from Memory
    print("Initializing Model")
    net = models.resnet18(pretrained=True, num_classes=1000)

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

    url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/' \
          'raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt'

    classes = eval(requests.get(url).content)

    print("Model created")
    s.listen(1)
    print('Socket now listening')
    conn, addr = s.accept()

    main()
