import pickle
import socket
import os
import torchvision
import torch.nn as nn
import numpy as np
import skimage
import argparse
import imagiz
import torch
import cv2
from resnet import *


HOST = ''
IMG_PORT = 8098
ARR_PORT = 8097
# ARR_PORT_2 = 8099
# body_parts = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
#               'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle',
#               'Left Hip', 'Left Knee', 'LAnkle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Background']


def initialize(frame, flip=False):
    if flip:
        img = np.flipud(frame).astype(np.float32) / 512
    else:
        img = frame.astype(np.float32) / 512
    if img.shape[-1] != 3:
        img = skimage.color.gray2rgb(img)

    if len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def net_forward(frame):
    inWidth = frame.shape[1]
    inHeight = frame.shape[0]
    image = initialize(frame)
    if use_gpu:
        image = image.cuda()
    out = net(image).cpu().numpy()
    return out


def get_classes(out):
    res = []
    for i, prob in enumerate(out):
        if prob > args.thresh:
            res.append(classes[i])
    return res
# def get_points(out, frame):
#     inWidth = frame.shape[1]
#     inHeight = frame.shape[0]
#
#     H = out.shape[2]
#     W = out.shape[3]
#     # Empty list to store the detected keypoints
#     threshold = .20
#     points = []
#     for i in range(len(body_parts)):
#         # confidence map of corresponding body's part.
#         probMap = out[0, i, :, :]
#
#         # Find global maxima of the probMap.
#         minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
#
#         # Scale the point to fit on the original image
#         x = (inWidth * point[0]) / W
#         y = (inHeight * point[1]) / H
#
#         if prob > threshold:
#             points.append((int(x), int(y)))
#         else:
#             points.append(None)
#     return points

def main():
    print("Connecting...")
    server = imagiz.Server(port=IMG_PORT)
    print("Connected...")
    while True:
        try:
            message = server.receive()
            frame = cv2.imdecode(message.image,1)
            ###Send
            out = forward(frame)
            picked_classes = get_classes(out)
            # pts = get_points(out, frame)
            # data_string = pickle.dumps(pts)
            data_string = pickle.dumps(picked_classes)
            # data_string_2 = pickle.dumps(pts, protocol=2)
            conn.send(data_string)
            # conn2.send(data_string_2)
            cv2.waitKey(1)
        except KeyboardInterrupt:
            s.close()
            cv2.destroyAllWindows()
            break
    print("\nSession Ended")

if __name__ == '__main__':
    #parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-root", type=str)
    parser.add_argument("-thresh", type=float, default=.2)
    parser.add_argument("-model_file", type=str)
    args = parser.parse_args()
    classes = os.listdir(args.root)


    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, ARR_PORT))
    # s2.bind((HOST, ARR_PORT_2))
    print('Socket bind complete')

    # Read the network from Memory
    print("Initializing Model")
    net = resnet18(num_classes=len(classes))
    net.load_state_dict(torch.load(args.model_file)['model_state_dict'])
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()
    net.eval()

    print("Model created")
    s.listen(1)
    # s2.listen(1)
    print('Socket now listening')
    conn, addr = s.accept()
    # conn2, addr2 = s2.accept()
    main()
