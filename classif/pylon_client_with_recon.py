from pypylon import pylon
import cv2
import pickle
import socket
import imagiz
import struct

HOST = '128.32.112.46'
IMG_PORT = 8098
ARR_PORT = 8097
RECON_PORT = 8099


def main():
    client = imagiz.Client("cc1", server_ip=HOST, server_port=IMG_PORT)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    i = 0
    # conecting to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Grabing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    while camera.IsGrabbing():
        try:
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                # Access the image data
                image = converter.Convert(grabResult)
                frame = image.GetArray()
                frame = rescale(frame, .45, width=480)
                r, image = cv2.imencode('.jpg', frame, encode_param)
                client.send(image)

                recon = receive_img(data)
                show_frame(recon)
            grabResult.Release()
        except (KeyboardInterrupt, EOFError) as e:
            # Releasing the resource
            camera.StopGrabbing()

            cv2.destroyAllWindows()

    # Releasing the resource
    camera.StopGrabbing()

    cv2.destroyAllWindows()

def receive_img(data):
    while len(data) < payload_size:
        data += s.recv(4096)

    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += s.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    return frame


def show_frame(frame):
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)


"""Resizes image according by the decimal scale (e.g .5)"""


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
    data = b''
    payload_size = struct.calcsize(">L")
    main()
