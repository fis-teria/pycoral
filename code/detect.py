import argparse
import time
import glob
import re
import time
import numpy as np
import cv2
import os
import re

from PIL import Image
from PIL import ImageDraw

from dataclasses import dataclass
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from f_sys import f_index

@dataclass
class ParserArgment:
    model: str
    input: str
    labels: str
    output: str
    threshold: float=0.4
    count: int=1

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    print(new_image.ndim )
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline="red")
        draw.text(
            (bbox.xmin + 10, bbox.ymin + 10),
            "%s\n%.2f" % (labels.get(obj.id, obj.id), obj.score),
            fill="red",
        )

def define_detect_interpreter():
    args = ParserArgment(model="../pycoral/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite", input="../pycoral/test_data/grace_hopper.bmp", labels="../pycoral/models/coco_labels.txt", output="../pycoral/test_data/grace_hopper_processed.bmp")

    print("Exit tflite model ", os.path.isfile("../pycoral/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"))
    labels = read_label_file(args.labels) if args.labels else {}
    interpreter = make_interpreter(args.model, device=':0')
    interpreter.allocate_tensors()
    interpreter.invoke()
    return interpreter, labels

def detect_image(img, interpreter, labels):
    print("Start Detect Images")

    pos = []#position list
    id = []

    args = ParserArgment(model="../pycoral/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite", input="../pycoral/test_data/grace_hopper.bmp", labels="../pycoral/models/coco_labels.txt", output="../pycoral/test_data/grace_hopper_processed.bmp")
    print("get image")
    #image = Image.open(args.input)
    #img = cv2.imread(args.input)
    image = cv2pil(img.copy())

    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS)
    )

    print("----INFERENCE TIME----")
    #print(
    #    "Note: The first inference is slow because it includes",
    #    "loading the model into Edge TPU memory.",
    #)
    for _ in range(args.count):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(interpreter, args.threshold, scale)
        print("%.2f ms" % (inference_time * 1000))

    print("-------RESULTS--------")
    if not objs:
        print("No objects detected")

    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print("  id:    ", obj.id)
        print("  score: ", obj.score)
        print("  bbox:  ", obj.bbox)
        #if labels.get(obj.id, obj.id) == "person":
        print(obj.bbox.xmin, obj.bbox.xmax, obj.bbox.ymin, obj.bbox.ymax)
            # wdataset = [str(count), str(obj.bbox.xmin), str(obj.bbox.xmax), str(obj.bbox.ymin), str(obj.bbox.ymax), '\n']
            # f.writelines(wdataset)
        xmin = obj.bbox.xmin
        xmax = obj.bbox.xmax
        ymin = obj.bbox.ymin
        ymax = obj.bbox.ymax
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        
        pos.append([xmin, xmax, ymin, ymax])
        id.append(labels.get(obj.id, obj.id))
        # area = re.findall(r'\d+', obj.bbox)
        # print(area)

        image = image.convert("RGB")
        draw_objects(ImageDraw.Draw(image), objs, labels)
        #pil2cv(image)
        #cv2.imshow("a", image)
        #cv2.waitKey(0)
        image.save("../pycoral/test_data/detect_result.jpg")
        #image.show()
        #time.sleep(0.2)
    
    print(pos)
    print(id)
    for i in range(0, len(id)):
        print([j for j, x in enumerate(id) if x == id[i]])
        same_id = [j for j, x in enumerate(id) if x == id[i]]
        if len(same_id) > 1:
            _pos_a = pos.pop(same_id[0])
            _id = id.pop(same_id[0])
            for j in range(1, len(same_id)):
                _pos_b = pos.pop(same_id[j] - j)
                if _pos_a[0] > _pos_b[0]:
                    _pos_a[0] = _pos_b[0]
                if _pos_a[1] < _pos_b[1]:
                    _pos_a[1] = _pos_b[1]
                if _pos_a[2] > _pos_b[2]:
                    _pos_a[2] = _pos_b[2]
                if _pos_a[3] < _pos_b[3]:
                    _pos_a[3] = _pos_b[3]
                garbage = id.pop(same_id[j] - j)
            pos.append(_pos_a)
            id.append(_id)
            print(pos)
            print(id)
            ImageDraw.Draw(image).rectangle([(_pos_a[0], _pos_a[2]), (_pos_a[1], _pos_a[3])], outline="green")
            image.save("../pycoral/test_data/detect_result.jpg")
        if i >= len(id) - 1:
            break
    
    return pos, id



def main():
    img = cv2.imread("../data/images/corridor/running1/color/000360.jpg")
    model_labels = define_detect_interpreter()
    detect_image(img, model_labels[0], model_labels[1])

if __name__ == "__main__":
    main()
