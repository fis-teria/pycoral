import zmq
import cv2
import struct
import sys
import numpy as np
from semantic_segmentation import SSImageData

def zmq_img_recive():
    # Connection String
    conn_str      = "tcp://*:5557"

    # Open ZMQ Connection
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(conn_str)

    # Receve Data from C++ Program
    byte_rows, byte_cols, byte_mat_type, data=  sock.recv_multipart()

    # Convert byte to integer
    rows = struct.unpack('i', byte_rows)
    cols = struct.unpack('i', byte_cols)
    mat_type = struct.unpack('i', byte_mat_type)

    if mat_type[0] == 0:
        # Gray Scale
        image = np.frombuffer(data, dtype=np.uint8).reshape((rows[0],cols[0]));
    else:
        # BGR Color
        image = np.frombuffer(data, dtype=np.uint8).reshape((rows[0],cols[0],3));

    # Write BMP Image
    #cv2.imshow("sample", image)
    #cv2.waitKey(100)
    #cv2.destroyAllWindows()
    return image

def zmq_n_recive():
    # Connection String
    conn_str      = "tcp://*:5556"

    # Open ZMQ Connection
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(conn_str)


    # Receve Data from C++ Program
    check_num =  sock.recv()
    n = struct.unpack('i', check_num)
    print("check send ", n)




def zmq_check_serve():
    conn_str="tcp://192.168.2.124:5558"

    args = sys.argv

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(conn_str)
    n = 1
    sock.send_multipart([np.array([n])])

def zmq_img_serve(img):
    conn_str="tcp://192.168.2.124:5555"

    args = sys.argv

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(conn_str)

    height, width = img.shape[:2]
    ndim = img.ndim

    data = [ np.array( [height] ), np.array( [width] ), np.array( [ndim] ), img.data]
    sock.send_multipart(data)
    print("send image")

def main():
    if( "color" == "color"):
        # Color
        img = cv2.imread("../data/images/ex_data/color/000030.jpg", cv2.IMREAD_COLOR)
    else:
        # Gray
        img = cv2.imread("../data/images/ex_data/color/000030.jpg", cv2.IMREAD_GRAYSCALE)
    #zmq_recive()
    print("serve img")
    zmq_img_serve(img)
    print("get loop n")
    zmq_n_recive()
    print("get img")
    zmq_img_recive()
    print("serve check")
    zmq_check_serve()

if __name__ == "__main__":
    main()