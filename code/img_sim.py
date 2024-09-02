import imgsim
import cv2
import numpy as np

vtr = imgsim.Vectorizer()

img0 = cv2.imread("ex_data/edge/000007.jpg")
img1 = cv2.imread("ex_data/edge/000035.jpg")

vec0 = vtr.vectorize(img0)
vec1 = vtr.vectorize(img1)
vecs = vtr.vectorize(np.array([img0, img1]))

dist = imgsim.distance(vec0, vec1)
print("distance =", dist)

dist = imgsim.distance(vecs[0], vecs[1])
print("distance =", dist)