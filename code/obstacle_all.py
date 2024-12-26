import sys
import numpy as np
import cv2
import os
import csv
import pandas as pd

from img_decode import img_Restoration
from detect import detect_image, define_detect_interpreter
from semantic_segmentation import define_ss_interpreter, semantic_segmentation, SSImageData

from database import DataBase
from feature_vector import Embedded_FeatureVector
from tflite_support.task import processor
FeatureVector = processor.FeatureVector

def main(argv):
    path = argv[1]
    for img_n in range(0, 1000):
        color_dir = path + "/color/" + str(int(img_n)).zfill(6) + ".jpg"
        dedge_dir = path + "/dedge/" + str(int(img_n)).zfill(6) + ".jpg"
        ip_color_dir = path + "/inpaint/color/" + str(int(img_n)).zfill(6) + ".jpg"
        ip_dedge_dir = path + "/inpaint/dedge/" + str(int(img_n)).zfill(6) + ".jpg"
        img = cv2.imread(color_dir)
        dedge = cv2.imread(dedge_dir)
        ip_img = cv2.imread(ip_color_dir)
        ip_dedge = cv2.imread(ip_dedge_dir)
        height = 480
        width = 848
        print("Start Python sub system")
        #img = np.zeros((hight, width), np.uint8)
        
        detect_interpreter = define_detect_interpreter()
        ss_interpeter = define_ss_interpreter()
        pos, d_id = detect_image(img, detect_interpreter[0], detect_interpreter[1])
        if len(pos) == 0:
            continue
        print(pos)
        print(d_id)
        image_datas, ss_id = semantic_segmentation(img, ss_interpeter[0], ss_interpeter[1], pos, d_id)
        
        obstacle_num = 0
        det_img = img.copy()
        seg_img = img.copy()
        det_dedge = dedge.copy()
        seg_dedge = dedge.copy()
        mask_img = cv2.resize(image_datas[0].mask_img, (pos[0][1]-pos[0][0],pos[0][3]-pos[0][2]))
        #mask_img = cv2.resize(image_datas[0].mask_img, (width, height))
        print(pos[0][3]-pos[0][2], " x ", pos[0][1]-pos[0][0])
        for y in range(0,height-1):
            for x in range(0,width-1):
                if x > pos[0][0] and x < pos[0][1] and y > pos[0][2] and y < pos[0][3]:
                    det_img[y][x] = [0, 0, 0]
                    det_dedge[y][x] = [0, 0, 0]
                    if mask_img[y - pos[0][2]][x - pos[0][0]][0] > 0 or mask_img[y - pos[0][2]][x - pos[0][0]][1] > 0 or mask_img[y - pos[0][2]][x - pos[0][0]][2] > 0:
                        #print(y - pos[0][2], " ", x - pos[0][0])
                        seg_img[y][x] = [0, 0, 0]
                        seg_dedge[y][x] = [0, 0 ,0]
                        obstacle_num += 1
        #cv2.imwrite("../pycoral/test_data/det_img.jpg", det_img)
        #cv2.imwrite("../pycoral/test_data/seg_img.jpg", seg_img)
        #cv2.imwrite("../pycoral/test_data/det_dedge.jpg", det_dedge)
        #cv2.imwrite("../pycoral/test_data/seg_dedge.jpg", seg_dedge)
        #cv2.imwrite("../pycoral/test_data/mask_img.jpg", mask_img)
        
        obstacle_alea0 = ((pos[0][3]-pos[0][2]) * (pos[0][1]-pos[0][0])) / (height*width)
        obstacle_alea1 = obstacle_num / (height * width)
        
        obstacle_res0 = -0.31472175 * obstacle_alea0 * obstacle_alea0 -0.68239725 * obstacle_alea0 + 0.999858
        obstacle_res1 = -0.31472175 * obstacle_alea1 * obstacle_alea1 -0.68239725 * obstacle_alea1 + 0.999858
        
        test_dir = argv[2]
        #run_dir = argv[2]
        csv_dir = "../data/logs/p_test/corridor/day_run4-3"
        test_color_dir = test_dir + "/color"
        test_dedge_dir = test_dir + "/dedge"

        #run_color_dir = run_dir + "/color"
        #run_dedge_dir = run_dir + "/dedge"
        test_db = DataBase(test_color_dir + "/DATABASE.db", test_color_dir + "/database.csv")
        test_dedge_db = DataBase(test_dedge_dir + "/DATABASE.db", test_dedge_dir + "/database.csv")

        
        test_num = sum(os.path.isfile(os.path.join(test_color_dir, name)) for name in os.listdir(test_color_dir))
        #run_num = sum(os.path.isfile(os.path.join(run_color_dir, name)) for name in os.listdir(run_color_dir))

        det_img_fv = Embedded_FeatureVector()
        det_img_fv.embed_from_array(det_img)
        seg_img_fv = Embedded_FeatureVector()
        seg_img_fv.embed_from_array(seg_img)
        ip_color_img_fv = Embedded_FeatureVector()
        ip_color_img_fv.embed_from_array(ip_img)
        ip_dedge_img_fv = Embedded_FeatureVector()
        ip_dedge_img_fv.embed_from_array(ip_dedge)

        det_dedge_fv = Embedded_FeatureVector()
        det_dedge_fv.embed_from_array(det_dedge)
        seg_dedge_fv = Embedded_FeatureVector()
        seg_dedge_fv.embed_from_array(seg_dedge)

        csv_path = csv_dir + "/"+ "obstacle" + str(int(img_n)).zfill(6) + ".csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as res:
            writer = csv.writer(res)
            writer.writerow(["cosine_det_img", "cosine_seg_img", "cosine_det_dedge", "cosine_seg_dedge", "cosine_ip_img", "cosine_ip_dedge"])
            for img_num in range(0, test_num-2):
                #test_fv.embed_from_file("../data/images/amalab/lab_root_3/loop_2/color/000041.jpg")
                
                ti_dir = test_color_dir + "/" + str(int(img_num)).zfill(6) + ".jpg"
                td_dir = test_dedge_dir + "/" + str(int(img_num)).zfill(6) + ".jpg"
                test_img = cv2.imread(ti_dir)
                test_dedge = cv2.imread(td_dir)
                re_img = cv2.resize(test_img ,(216, 120))
                re_dedge = cv2.resize(test_dedge,(216, 120))
                test_color_img_fv = Embedded_FeatureVector()
                test_color_img_fv.embed_from_array(re_img)
                test_dedge_img_fv = Embedded_FeatureVector()
                test_dedge_img_fv.embed_from_array(re_dedge)
                
                test_color_fv = test_db.get_db(img_num)
                test_dedge_fv = test_dedge_db.get_db(img_num)
                
                result_det_img = det_img_fv.cosine_similarity(test_color_fv)
                result_seg_img = seg_img_fv.cosine_similarity(test_color_fv)
                
                result_det_dedge = det_dedge_fv.cosine_similarity(test_dedge_fv)
                result_seg_dedge = seg_dedge_fv.cosine_similarity(test_dedge_fv)
                
                result_ip_color = ip_color_img_fv.cosine_similarity(test_color_img_fv.get_feature_vector())
                result_ip_dedge = ip_dedge_img_fv.cosine_similarity(test_dedge_img_fv.get_feature_vector())


                writer.writerow([result_det_img, result_seg_img, result_det_dedge, result_seg_dedge, result_ip_color, result_ip_dedge])
                        


if __name__ == "__main__":
    main(sys.argv)
