import pickle
import socket
import os
import torch.nn as nn
import requests
import argparse
import imagiz
import torch
import cv2
import json
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict

HOST = ''
# IMG_PORT = 8090
# ARR_PORT = 8091

def net_forward(frame):
    """Runs the classifier network given an image"""
    image = Image.fromarray(frame)
    image = trans(image).view(1, 3, 224, 224)
    if use_gpu:
        image = image.cuda()

    #print(image.shape)
    out = net(image).cpu() #.detach().numpy()
    return out #[0]


def get_classes(out):
    """Returns the top 7 classes with their respective probabilities"""
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0]
    return [(classes[idx.item()], round(percentage[idx.item()].item(),2)) for idx in indices[0][:7]]

def main():
    """Recieves Webcam Images from Client and Classifies them"""
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

def fix_state_dict(state_dict):
    """Fixes the state dictionary of a pretrained model in order to load it"""
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
    parser.add_argument("-model_dir", default='', type=str)
    parser.add_argument("-imagenet_train_dir", type=str,
                         default='')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    IMG_PORT = args.port
    ARR_PORT = IMG_PORT + 1

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print('Socket created')

    s.bind((HOST, ARR_PORT))

    print('Socket bind complete')

    # Read the network from Memory
    print("Initializing Model")


    if args.imagenet_train_dir:
        class_info_json_filepath = './imagenet_class_info.json'
        with open(class_info_json_filepath) as class_info_json_f:
            class_info_dict = json.load(class_info_json_f)

        class_wnids = os.listdir(args.imagenet_train_dir)
        class_wnids.sort()
        classes = [class_info_dict[class_wnid]["class_name"] for class_wnid in class_wnids]
        net = models.resnet18(num_classes=1000, pretrained=args.pretrained)

    else:
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/' \
              'raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt'
        classes = eval(requests.get(url).content)
        net = models.resnet18(num_classes=1000, pretrained=args.pretrained)

    if args.model_dir:
        checkpoint = torch.load(args.model_dir)
        net.load_state_dict(fix_state_dict(checkpoint['state_dict']))

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        net = net.cuda()
    net.eval()

    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # from http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    ])

    print("Model created")
    s.listen(1)
    print('Socket now listening')
    conn, addr = s.accept()

    main()
