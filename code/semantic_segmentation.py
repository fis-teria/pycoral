
import argparse
import glob
import cv2
import time

import numpy as np
from PIL import Image

from dataclasses import dataclass
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter
from f_sys import f_open, f_index

@dataclass
class ParserArgment:
    model: str
    keep_aspect_ratio: bool = False

@dataclass
class SSImageData:
    mask_img: np.ndarray
    pos: list


   


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

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def define_ss_interpreter():
    args = ParserArgment(model="../pycoral/models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite", keep_aspect_ratio=False)
    #args = ParserArgment(model="../pycoral/test_data/deeplab_mobilenet_edgetpu_slim_cityscapes_quant_edgetpu.tflite", keep_aspect_ratio=False)
    interpreter = make_interpreter(args.model, device=':0')
    interpreter.allocate_tensors()
    interpreter.invoke()
    return interpreter, args.keep_aspect_ratio
    
def semantic_segmentation(origin_img, interpreter, keep_aspect_ratio, pos, d_id):
    print("Start Semantic Segmentation")
    width, height = common.input_size(interpreter)
    i = 0
    data = f_open("../pycoral/models/pascal_voc_segmentation_labels.txt")
    datas = []
    ss_id = []
    while i < len(pos): 
      if len(pos) == 0:
        img = cv2pil(origin_img.copy())
      else:
        image = origin_img.copy()
        img = cv2pil(image[pos[i][2]:pos[i][3], pos[i][0]:pos[i][1]])      
        #img = cv2pil(image)

      #cv2.imshow("img", pil2cv(img))
      #img.show()
      if keep_aspect_ratio:
          resized_img, _ = common.set_resized_input(
              interpreter, img.size, lambda size: img.resize(size, Image.LANCZOS))
      else:
          resized_img = img.resize((width, height), Image.LANCZOS)
          common.set_input(interpreter, resized_img)
      
      print("----INFERENCE TIME----")
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() -start
      print("%.2f ms" % (inference_time * 1000))

      result = segment.get_output(interpreter)
      if len(result.shape) == 3:
          result = np.argmax(result, axis=-1)

      # If keep_aspect_ratio, we need to remove the padding area.
      new_width, new_height = resized_img.size
      result = result[:new_height, :new_width]
      ss_id.append(f_index(data, d_id[i]))

      for x in range(0,new_width):
        for y in range(0,new_width): 
            if result[y][x] != ss_id[-1]:
               #result[y][x] = 0
               z = 0
              

      mask_img = Image.fromarray(label_to_color_image(result).astype(np.uint8))
      mask_cvimg = pil2cv(mask_img)
      print(mask_cvimg.shape)
      data_expanded = SSImageData(mask_img = mask_cvimg.copy(), pos = pos[i])
      #print(data_expanded)
      datas.append(data_expanded)
      print("ss_label ", f_index(data, d_id[i]))
      i+=1
    #mask_img.show()
    #cv2.waitKey(100)
    
    image_datas = datas
    
    return image_datas, ss_id


    # Concat resized input image and processed segmentation results.
    #output_img = Image.new('RGB', (2 * new_width, new_height))
    #output_img.paste(resized_img, (0, 0))
    #output_img.paste(mask_img, (width, 0))
    #output_img.save(args.output)
    #print('Done. Results saved at', args.output)

def main():
    img = cv2.imread("../pycoral/test_data/bird.bmp")
    pos = [[0, 200, 0, 100], [100,200, 100, 200]]
    d_id = "bird"
    ss_models_keep_aspect_ratio = define_ss_interpreter()
    image_datas = semantic_segmentation(img, ss_models_keep_aspect_ratio[0], ss_models_keep_aspect_ratio[1], pos, d_id)
    print(image_datas)
    print(len(image_datas))

if __name__ == '__main__':
  main()
