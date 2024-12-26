import os
import cv2
import csv
import sys
import numpy as np

from database import DataBase
from feature_vector import Embedded_FeatureVector
from tflite_support.task import processor
FeatureVector = processor.FeatureVector

def test_fv():
    lb_fv = Embedded_FeatureVector()
    rb_fv = Embedded_FeatureVector()
    lw_fv_dst = Embedded_FeatureVector()
    rw_fv_dst = Embedded_FeatureVector()
    w_fv = Embedded_FeatureVector()
    b_fv = Embedded_FeatureVector()
    
    for fn in range(0, 100):
        frame_num = fn
        db = DataBase("../data/images/amalab/lab_root_3/loop_2/color/DATABASE.db", "../data/images/amalab/lab_root_3/loop_2/color/feature_vector.csv")
        db_dst = DataBase("../data/images/amalab/lab_root_3/loop_2/dst/DATABASE.db", "../data/images/amalab/lab_root_3/loop_2/dst/feature_vector.csv")
        img = cv2.imread("../data/images/amalab/lab_root_3/loop_2/color/" + str(frame_num).zfill(6) + ".jpg")
        dst = cv2.imread("../data/images/amalab/lab_root_3/loop_2/dst/" + str(frame_num).zfill(6) + ".jpg")
        height, width, channel = img.shape[:3]

        test_fv = Embedded_FeatureVector()
        test_fv.embed_from_array(dst)
        #test_fv.embed_from_file("../data/images/amalab/lab_root_3/loop_2/color/000041.jpg")
        
        test_v = db_dst.get_db(frame_num)
        print(test_fv.result.embeddings[0].feature_vector)
        print(test_v)
        print("test result :", test_fv.cosine_similarity(test_v))
        if test_fv.cosine_similarity(test_v) != 1.0 :
            return -1
        
        csv_path = "../data/logs/search_result/test_data/fv_cos_test_"+ str(frame_num).zfill(6) + ".csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as res:
            writer = csv.writer(res)
            writer.writerow(["color_left_black_image", "color_right_black_image", "dst_left_white_image", "dst_right_white_image"])
            for num in range(0, 11):
                lb_img = np.zeros((height, width, 3), np.uint8)
                rb_img = np.zeros((height, width, 3), np.uint8)
                lw_dst = np.zeros((height, width, 3), np.uint8) + 255
                rw_dst = np.zeros((height, width, 3), np.uint8) + 255
                white = np.zeros((height, width, 3), np.uint8)
                black = np.zeros((height, width, 3), np.uint8)

                lb_img = img.copy()
                rb_img = img.copy()
                lw_dst = dst.copy()
                rw_dst = dst.copy()

                for y in range(0, width):
                    for x in range(0, height):
                        white[x, y] = [255,255,255]
                        if y < width*num/10:
                            rb_img[x, y] = [0,0,0]
                            rw_dst[x, y] = [255, 255, 255]
                        else:
                            lb_img[x, y] = [0, 0, 0]
                            lw_dst[x, y] = [255, 255, 255]
                
                
                vec = db.get_db(frame_num)
                vec_dst = db_dst.get_db(frame_num)
                lb_fv.embed_from_array(lb_img)
                rb_fv.embed_from_array(rb_img)
                lw_fv_dst.embed_from_array(lw_dst)
                rw_fv_dst.embed_from_array(rw_dst)
                w_fv.embed_from_array(white)
                b_fv.embed_from_array(black)
                
                lb_res = lb_fv.cosine_similarity(vec)
                rb_res = rb_fv.cosine_similarity(vec)
                lw_res_dst = lw_fv_dst.cosine_similarity(vec_dst)
                rw_res_dst = rw_fv_dst.cosine_similarity(vec_dst)
                
                wb_res = w_fv.cosine_similarity(b_fv.result.embeddings[0].feature_vector)
                
                #with open("../pycoral/test_data/white_and_black_images_feature_vectors.csv", 'w', newline='', encoding='utf-8') as f:
                #    writer = csv.writer(f)
                #    writer.writerow(b_fv.result.embeddings[0].feature_vector.value)
                #    writer.writerow(w_fv.result.embeddings[0].feature_vector.value)


                #cv2.imwrite("../pycoral/test_data/white.jpg", white)
                #cv2.imwrite("../pycoral/test_data/black.jpg", black)
                print("color left black result : ", lb_res)
                print("color right black result : ", rb_res)
                print("dst left black result : ", lw_res_dst)
                print("dst right black result : ", rw_res_dst)
                res = [lb_res, rb_res, lw_res_dst, rw_res_dst]
                writer.writerow(res)
                
                print("white vs black result : ", wb_res)
                path0 = "../data/images/research/relational_fv/lb_col_" + str(num).zfill(6)+ "_img.jpg"
                cv2.imwrite(path0, lb_img)
                path1 = "../data/images/research/relational_fv/rb_col_" + str(num).zfill(6)+ "_img.jpg"
                cv2.imwrite(path1, rb_img)
                path2 = "../data/images/research/relational_fv/lw_dst_" + str(num).zfill(6)+ "_img.jpg"
                cv2.imwrite(path2, lw_dst)
                path3 = "../data/images/research/relational_fv/rw_dst_" + str(num).zfill(6)+ "_img.jpg"
                cv2.imwrite(path3, rw_dst)
                #cv2.imwrite("../data/images/research/relational_fv/rw_dst_img.jpg", rb_img)
    
#running走行の１フレームに対するテスト走行フレーム群のコサイン類似度を求める。    
def test_run(argv):
    test_dir = argv[1]
    run_dir = argv[2]
    csv_dir = argv[3]
    test_color_dir = test_dir + "/color"
    test_dedge_dir = test_dir + "/dedge"
    run_color_dir = run_dir + "/color"
    run_dedge_dir = run_dir + "/dedge"
    test_db = DataBase(test_color_dir + "/DATABASE.db", test_color_dir + "/database.csv")
    test_dedge_db = DataBase(test_dedge_dir + "/DATABASE.db", test_dedge_dir + "/database.csv")
    
    test_num = sum(os.path.isfile(os.path.join(test_color_dir, name)) for name in os.listdir(test_color_dir))
    run_num = sum(os.path.isfile(os.path.join(run_color_dir, name)) for name in os.listdir(run_color_dir))
    with open("../data/logs/save/test_run_save.txt") as f:
        for line in f:
            save = int(line)
    print("now start for ", save)
    print(test_num -2)
    print(run_num -2)
    for fn in range(save, run_num-2):
        frame_num = fn
        run_color = cv2.imread(run_color_dir + "/" + str(frame_num).zfill(6) + ".jpg")
        run_dedge = cv2.imread(run_dedge_dir + "/" + str(frame_num).zfill(6) + ".jpg")

        run_color_fv = Embedded_FeatureVector()
        run_color_fv.embed_from_array(run_color)
        run_dedge_fv = Embedded_FeatureVector()
        run_dedge_fv.embed_from_array(run_dedge)
        
        
        csv_path = csv_dir + "/cosine_"+ str(frame_num).zfill(6) + ".csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as res:
            writer = csv.writer(res)
            writer.writerow(["cosine_color", "cosine_dedge"])
            for img_num in range(0, test_num-2):
                #test_fv.embed_from_file("../data/images/amalab/lab_root_3/loop_2/color/000041.jpg")
                
                test_color_fv = test_db.get_db(img_num)
                test_dedge_fv = test_dedge_db.get_db(img_num)
                
                result_color = run_color_fv.cosine_similarity(test_color_fv)
                result_dedge = run_dedge_fv.cosine_similarity(test_dedge_fv)
                
                writer.writerow([result_color, result_dedge])
                
                print("------------------------------------------------------------------------")
                print("color :: test ", img_num, "frame : running ", frame_num, " frame ->  ", result_color)
                print("dedge :: test ", img_num, "frame : running ", frame_num, " frame ->  ", result_dedge)
                print("------------------------------------------------------------------------")
        save+=1
        with open("../data/logs/save/test_run_save.txt", "w") as f:
            print(save, file=f)
    save = 0
    with open("../data/logs/save/test_run_save.txt", "w") as f:
        print(save, file=f)

                   
    
def main():
    argv = sys.argv
    test_run(argv)

if __name__ == "__main__":
    main()
