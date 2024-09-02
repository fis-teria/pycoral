import numpy as np

from img_decode import img_Restoration
from detect import detect_image, define_detect_interpreter
from zeroMQ import zmq_n_serve, zmq_img_serve, zmq_recive, zmq_check_recive
from semantic_segmentation import define_ss_interpreter, semantic_segmentation, SSImageData


def main():
    #hight = 480
    #width = 848
    print("Start Python sub system")
    #img = np.zeros((hight, width), np.uint8)
    
    detect_interpreter = define_detect_interpreter()
    ss_interpeter = define_ss_interpreter()
    i = 1
    while i:
        print("waiting recive img")
        img = zmq_recive()
        print("return img ", img.dtype)
        pos, d_id = detect_image(img, detect_interpreter[0], detect_interpreter[1])
        print(pos)
        image_datas, ss_id = semantic_segmentation(img, ss_interpeter[0], ss_interpeter[1], pos, d_id)
        n = zmq_n_serve(image_datas)
        print("Send msg Loop ", n)
        for j in range(0, n):
            zmq_img_serve(image_datas[j].mask_img, image_datas[j].pos, ss_id[j])
            zmq_check_recive()


if __name__ == "__main__":
    main()
