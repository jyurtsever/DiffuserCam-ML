from pypylon import pylon
import cv2
import pickle
import socket
import imagiz
import argparse
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
    camera.Open()
    #Setting camera params
    camera.GainAuto.SetValue("Off")
    if args.exposure_time:
        camera.ExposureAuto.SetValue("Off")
        camera.ExposureTime.SetValue(args.exposure_time)
    else:
        minLowerLimit = camera.AutoExposureTimeLowerLimit.GetMin()
        maxUpperLimit = camera.AutoExposureTimeUpperLimit.GetMax()
        camera.AutoExposureTimeLowerLimit.SetValue(minLowerLimit)
        camera.AutoExposureTimeUpperLimit.SetValue(maxUpperLimit)
        camera.ExposureAuto.SetValue("Continuous")
    camera.Gain.SetValue(args.gain)
    camera.BalanceWhiteAuto.SetValue(args.auto_white_balance)
    camera.Gamma.SetValue(args.gamma)
    camera.SensorShutterMode.SetValue(args.shutter_mode)

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
                ### Recieve Array
                data = s.recv(4096)
                class_names = pickle.loads(data)
                if i % 10 == 0:
                    if class_names:
                        print(class_names)
                    else:
                        print("Not classiied")
                show_frame(frame)
                i += 1
            grabResult.Release()
        except (KeyboardInterrupt, EOFError) as e:
            # Releasing the resource
            camera.StopGrabbing()
            camera.Close()
            cv2.destroyAllWindows()

    # Releasing the resource
    camera.StopGrabbing()

    cv2.destroyAllWindows()

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
    parser = argparse.ArgumentParser(description='Exposure and Gain')
    parser.add_argument('-exposure_time', type=int, default=0)
    parser.add_argument('-gain', type=float, default=0.0)
    parser.add_argument('-srgb', type=str, default='rgb')
    parser.add_argument('-gamma', type=int, default=1)
    parser.add_argument('-shutter_mode', type=str, default='GlobalReset')
    parser.add_argument('-auto_white_balance', type=str, default='Off')
    args = parser.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, ARR_PORT))
    main()