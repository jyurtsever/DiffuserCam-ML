import cv2
import numpy as np
import socket
import pickle
import os
import struct ### new code
import imagiz
import time
HOST = '128.32.112.46'
IMG_PORT = 8098
ARR_PORT = 8097


# body_parts = ['Nose', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
#               'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Hip', 'Right Knee', 'Right Ankle',
#               'Left Hip', 'Left Knee', 'LAnkle', 'Right Eye', 'Left Eye', 'Right Ear', 'Left Ear', 'Background']
#
# body_dict = {body_parts[i]: i for i in range(len(body_parts))}




def main():
    client = imagiz.Client("cc1",server_ip=HOST, server_port=IMG_PORT)
    vid = cv2.VideoCapture(0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    # cache = {'Neck': None, 'Left Shoulder': None, 'Right Wrist': None, 'Left Wrist': None,
    #         'Right Elbow': None, 'Right Shoulder': None, 'Left Shoulder': None}
    i = 0
    while True:
        try:
            r, frame = vid.read()
            frame = rescale(frame, .45, width=500)
            if r:
                r, image = cv2.imencode('.jpg', frame, encode_param)
                client.send(image)
                ### Recieve Array
                data = s.recv(4096)
                class_names = pickle.loads(data)
                if class_names and i % 5 == 0:
                    print(class_names)
                else:
                    print("Not classiied")
                show_frame(frame)
                i += 1
            else:
                break

        except (KeyboardInterrupt, EOFError) as e:
            s.close()
            vid.release()
            cv2.destroyAllWindows()
            break


"""Displays Live Video In Opencv Window"""


def show_frame(frame):
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
 #   cv2.destroyAllWindows()


"""Recizes image according by the decimal scale (e.g .5)"""


def rescale(img, scale, width=None):
    if width:
        scale = width/img.shape[1]
    else:
        width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized



if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, ARR_PORT))
    main()