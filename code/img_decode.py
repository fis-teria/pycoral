import time
import cv2
import numpy as np

def img_Restoration(img_comp:list, width:int, hight:int):
    print("Will start Restoration")
    print("Check opencv can install", cv2.__version__)
    print("array size ", len(img_comp))

    #height, width
    img = np.zeros((hight, width, 3), np.uint8)
    cl_sec = int(hight/8 * width)
    #print(img_comp)
    start = time.perf_counter()
    for y in range(0, hight):
        for x in range(0,int(width/8)):
            #print(106*y + x, " ", 106*y + cl_sec + x, " ", 106*y + cl_sec*2 + x)
            b_el = 106*y + x
            g_el = b_el + cl_sec
            r_el = b_el + cl_sec*2
            #decode B
            img[y][8*x][0] = (img_comp[b_el] & int('0xFF00000000000000', 16))>>56
            img[y][8*x+1][0] = (img_comp[b_el] & int('0x00FF000000000000', 16))>>48
            img[y][8*x+2][0] = (img_comp[b_el] & int('0x0000FF0000000000', 16))>>40
            img[y][8*x+3][0] = (img_comp[b_el] & int('0x000000FF00000000', 16))>>32
            img[y][8*x+4][0] = (img_comp[b_el] & int('0x00000000FF000000', 16))>>24
            img[y][8*x+5][0] = (img_comp[b_el] & int('0x0000000000FF0000', 16))>>16
            img[y][8*x+6][0] = (img_comp[b_el] & int('0x000000000000FF00', 16))>>8
            img[y][8*x+7][0] = img_comp[b_el] & int('0x00000000000000FF', 16)
            #decode G
            img[y][8*x][1] = (img_comp[g_el] & int('0xFF00000000000000', 16))>>56
            img[y][8*x+1][1] = (img_comp[g_el] & int('0x00FF000000000000', 16))>>48
            img[y][8*x+2][1] = (img_comp[g_el] & int('0x0000FF0000000000', 16))>>40
            img[y][8*x+3][1] = (img_comp[g_el] & int('0x000000FF00000000', 16))>>32
            img[y][8*x+4][1] = (img_comp[g_el] & int('0x00000000FF000000', 16))>>24
            img[y][8*x+5][1] = (img_comp[g_el] & int('0x0000000000FF0000', 16))>>16
            img[y][8*x+6][1] = (img_comp[g_el] & int('0x000000000000FF00', 16))>>8
            img[y][8*x+7][1] = img_comp[g_el] & int('0x00000000000000FF', 16)
            #decode R
            img[y][8*x][2] = (img_comp[r_el] & int('0xFF00000000000000', 16))>>56
            img[y][8*x+1][2] = (img_comp[r_el] & int('0x00FF000000000000', 16))>>48
            img[y][8*x+2][2] = (img_comp[r_el] & int('0x0000FF0000000000', 16))>>40
            img[y][8*x+3][2] = (img_comp[r_el] & int('0x000000FF00000000', 16))>>32
            img[y][8*x+4][2] = (img_comp[r_el] & int('0x00000000FF000000', 16))>>24
            img[y][8*x+5][2] = (img_comp[r_el] & int('0x0000000000FF0000', 16))>>16
            img[y][8*x+6][2] = (img_comp[r_el] & int('0x000000000000FF00', 16))>>8
            img[y][8*x+7][2] = img_comp[r_el] & int('0x00000000000000FF', 16)
    inference_time = time.perf_counter() - start
    print("%.2f ms" % (inference_time * 1000))
    cv2.imshow("sample", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("End Restoration")
    return img
    
def main():
    array = []
    for v in range(0, int(106 * 480 * 3 / 4)):
        array.append(int('0xFFFFFFFFFFFFFF00', 16))
    get_img = img_Restoration(array, int(848/4), int(480/4))
    print(get_img.shape)

if __name__ == '__main__':
    main()