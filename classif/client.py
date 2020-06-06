import cv2
import socket
import pickle
import imagiz
import argparse
# HOST = '128.32.112.46'
# IMG_PORT = 8090
# ARR_PORT = 8091


def main():
    client = imagiz.Client("cc1",server_ip=HOST, server_port=IMG_PORT)
    vid = cv2.VideoCapture(0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
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
                if class_names and i % 20 == 0:
                    print(class_names)
                elif not class_names:
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




def show_frame(frame):
    """Displays Live Video In Opencv Window"""

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)




def rescale(img, scale, width=None):
    """Resizes image according by the decimal scale (e.g .5)"""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("ip_addr", type=str)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    IMG_PORT = args.port
    ARR_PORT = IMG_PORT + 1
    HOST = args.ip_addr
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, ARR_PORT))
    main()