import zmq
import cv2
import struct
import sys
import numpy as np
import threading
import time
from typing import Any, Dict

from zmq.utils.monitor import recv_monitor_message

from semantic_segmentation import SSImageData

print(f"libzmq-{zmq.zmq_version()}")
if zmq.zmq_version_info() < (4, 0):
    raise RuntimeError("monitoring in libzmq version < 4.0 is not supported")

EVENT_MAP = {}
print("Event names:")
for name in dir(zmq):
    if name.startswith('EVENT_'):
        value = getattr(zmq, name)
        print(f"{name:21} : {value:4}")
        EVENT_MAP[value] = name


def event_monitor(monitor: zmq.Socket) -> None:
    while monitor.poll():
        evt: Dict[str, Any] = {}
        mon_evt = recv_monitor_message(monitor)
        evt.update(mon_evt)
        evt['description'] = EVENT_MAP[evt['event']]
        print(f"Event: {evt}")
        if evt['event'] == zmq.EVENT_MONITOR_STOPPED:
            break
    monitor.close()
    print()
    print("event monitor thread done!")


def zmq_recive():
    # Connection String
    conn_str      = "tcp://*:5555"

    # Open ZMQ Connection
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    #monitor = sock.get_monitor_socket()
    #t = threading.Thread(target=event_monitor, args=(monitor,))
    #t.start()
    sock.bind(conn_str)    
    
    # Receve Data from C++ Program
    byte_rows, byte_cols, byte_mat_type, data=  sock.recv_multipart()

    # Convert byte to integer
    print(byte_rows)
    if len(byte_rows) == 4:
        rows = struct.unpack('i', byte_rows)
        cols = struct.unpack('i', byte_cols)
        mat_type = struct.unpack('i', byte_mat_type)
    else:
        rows = struct.unpack('q', byte_rows)
        cols = struct.unpack('q', byte_cols)
        mat_type = struct.unpack('q', byte_mat_type)

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

def zmq_check_recive():
    # Connection String
    conn_str      = "tcp://*:5558"

    # Open ZMQ Connection
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(conn_str)


    # Receve Data from C++ Program
    check_num =  sock.recv()
    print("check send img complete")




def zmq_n_serve(img_datas):
    #main system 192.168.1.2
    conn_str="tcp://192.168.1.2:5556"

    args = sys.argv

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)

    #monitor = sock.get_monitor_socket()
    #t = threading.Thread(target=event_monitor, args=(monitor,))
    #t.start()

    sock.connect(conn_str)
    n = len(img_datas)
    data = [np.array([n]), np.array([1])]
    print(data)
    sock.send_multipart(data)
    return n

def zmq_img_serve(img, pos, id):
    #main system 192.168.1.2
    conn_str="tcp://192.168.1.2:5557"

    args = sys.argv

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(conn_str)

    height, width = img.shape[:2]
    ndim = img.ndim

    data = [ np.array( [height] ), np.array( [width] ), np.array( [ndim] ), img.data , np.array([pos[0]]), np.array([pos[1]]), np.array([pos[2]]), np.array([pos[3]]), np.array([id])]
    sock.send_multipart(data)
    print("send image")

def main():
    #if( "color" == "color"):
        # Color
        #img = cv2.imread("ex_data/color/000030.jpg", cv2.IMREAD_COLOR)
    #else:
        # Gray
        #img = cv2.imread("ex_data/color/000030.jpg", cv2.IMREAD_GRAYSCALE)
    print("get image")
    zmq_recive()
    print("get image complete")
    print("loop n serve")
    n = zmq_n_serve([2, 3])
    print("loop ", n, " serve complete")

if __name__ == "__main__":
    main()
