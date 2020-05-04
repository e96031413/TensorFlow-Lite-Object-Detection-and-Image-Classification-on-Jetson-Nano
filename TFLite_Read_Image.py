import glob
import argparse
import io
import os
import time
import random
import numpy as np
import cv2
from PIL import Image
from tflite_runtime.interpreter import Interpreter

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument( '--model', help='File path of .tflite file.', default="test.tflite")
parser.add_argument('--labels_', help='File path of labels file.', default="test.txt")
args = parser.parse_args()

labels_ = load_labels(args.labels_)
interpreter_ = Interpreter(args.model)
interpreter_.allocate_tensors()
_, height, width, _ = interpreter_.get_input_details()[0]['shape']
seconds = time.time()
local_time=time.ctime(seconds)
a = glob.glob('*.jpg')

for i in a:
    img = cv2.imread(i)
    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    results = classify_image(interpreter_, image)
    label_id, prob = results[0]
    print("Pic" + i)
    print("LabelName:",labels_[label_id])
    print("ScoreValue:",prob)
    print("Time:",local_time)
    os.remove(i)
