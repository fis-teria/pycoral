import sys
import numpy as np
import cv2
import os
import csv
import pylab as plt
import pandas as pd

from img_decode import img_Restoration
from detect import detect_image, define_detect_interpreter
from semantic_segmentation import define_ss_interpreter, semantic_segmentation, SSImageData


def make_mask(argv):
    path = argv[1]
    test_color_dir = path + "/color"
    test_num = sum(os.path.isfile(os.path.join(test_color_dir, name)) for name in os.listdir(test_color_dir))
    for img_n in range(0, test_num - 1):
        color_img = path + "/color/" + str(int(img_n)).zfill(6) + ".jpg"
        dedge_img = path + "/dedge/" + str(int(img_n)).zfill(6) + ".jpg"
        png_img = path + "/inpaint/color/" + str(int(img_n)).zfill(6) + ".png"
        png_dedge = path + "/inpaint/dedge/" + str(int(img_n)).zfill(6) + ".png"
        mask_dir = path + "/inpaint/mask/" + str(int(img_n)).zfill(6) + ".png"
        
        img = cv2.imread(color_img)
        dedge = cv2.imread(dedge_img)
        height = 480
        width = 848
        
        gridsize=8
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # GBRからLABに変換
        lab_planes = cv2.split(lab)  # LABに分離
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))  # L(明度)に対してGray画像と同様な抽出・処理を実施
        lab_planes = list(lab_planes)
        lab_planes[0] = clahe.apply(lab_planes[0])  # L(明度)に対して明るくする
        lab_planes = tuple(lab_planes)
        lab = cv2.merge(lab_planes)  # LABをマージ
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # LABからBGRに変換

        
        detect_interpreter = define_detect_interpreter()
        ss_interpeter = define_ss_interpreter()
        pos, d_id = detect_image(img, detect_interpreter[0], detect_interpreter[1])
        print(pos)
        print(d_id)
        if len(pos) != 0:
            image_datas, ss_id = semantic_segmentation(img, ss_interpeter[0], ss_interpeter[1], pos, d_id)
            
            mask = np.zeros((height, width, 1), np.uint8)

            mask_img = cv2.resize(image_datas[0].mask_img, (pos[0][1] - pos[0][0], pos[0][3] - pos[0][2]))
            # mask_img = cv2.resize(image_datas[0].mask_img, (width, height))
            print(pos[0][3] - pos[0][2], " x ", pos[0][1] - pos[0][0])
            for y in range(0, height - 1):
                for x in range(0, width - 1):
                    if x > pos[0][0] and x < pos[0][1] and y > pos[0][2] and y < pos[0][3]:
                        if mask_img[y - pos[0][2]][x - pos[0][0]][0] > 0 or mask_img[y - pos[0][2]][x - pos[0][0]][1] > 0 or mask_img[y - pos[0][2]][x - pos[0][0]][2] > 0:
                            # print(y - pos[0][2], " ", x - pos[0][0])
                            mask[y][x] = 255

            cv2.imwrite(png_img, img)
            cv2.imwrite(png_dedge, dedge)
            cv2.imwrite(mask_dir, mask)
        else:
            mask = np.zeros((height, width, 1), np.uint8)
            cv2.imwrite(png_img, img)
            cv2.imwrite(png_dedge, dedge)
            cv2.imwrite(mask_dir, mask)
            print("image save ", mask_dir)


def main():
    argv = sys.argv
    make_mask(argv)


if __name__ == "__main__":
    main()
