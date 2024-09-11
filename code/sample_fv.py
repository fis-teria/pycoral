import cv2
from tflite_support.task import vision

# Initialization
image_searcher = vision.ImageSearcher.create_from_file("../pycoral/models/mobilenet-v3-tflite-large-075-224-feature-vector-v1.tflite")

# Run inference
image = vision.TensorImage.create_from_file("../pycoral/test_data/bird.bmp")
result = image_searcher.search(image)