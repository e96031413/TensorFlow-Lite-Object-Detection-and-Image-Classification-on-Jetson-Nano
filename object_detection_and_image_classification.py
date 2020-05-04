import io
import os
import glob
import random
import argparse
import cv2
import numpy as np
import pandas as pd
import sys
import time
from firebase import firebase
from google.cloud import storage
from PIL import Image
from tflite_runtime.interpreter import Interpreter

def gstreamer_pipeline(
    capture_width=3280,                                     
    capture_height=2464,
    display_width=224,
    display_height=224,
    framerate=120,                                          
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
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

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default='Sample_TFLite_model')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--model', help='File path of .tflite file.', default="test.tflite")
parser.add_argument('--labels_', help='File path of labels file.', default="test.txt")

args = parser.parse_args()

labels_ = load_labels(args.labels_)
interpreter_ = Interpreter(args.model)
interpreter_.allocate_tensors()
_, height, width, _ = interpreter_.get_input_details()[0]['shape']

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)


if cap.isOpened():
    window_handle = cv2.namedWindow("Object detector", cv2.WINDOW_AUTOSIZE)
    while cv2.getWindowProperty("Object detector", 0) >= 0:
        
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()
        # Grab frame from video stream
        ret_val, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
            
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # Draw framerate in corner of frame
                cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
                cv2.imshow('Object detector', frame)

    # Calculate framerate
                t2 = cv2.getTickCount()
                time1 = (t2-t1)/freq
                frame_rate_calc= 1/time1

                if object_name == "bird":
                    print("Found bird!!!")
                    file_Name = object_name + str(random.randint(1,99999)) + '.jpg'
                    cv2.imwrite(file_Name, frame)
                    for file in glob.glob("*.jpg"):
                        img = cv2.imread(file)
                        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                        start_time = time.time()
                        results = classify_image(interpreter_, image)
                        elapsed_ms = (time.time() - start_time) * 1000
                        label_id, prob = results[0]
                        seconds = time.time()
                        local_time=time.ctime(seconds)
                        print("LabelName:",labels_[label_id])
                        print("ScoreValue:",prob)
                        print("Time:",local_time)
                        fileName =labels_[label_id] + str(random.randint(1,99999)) + '.jpg'
                        cv2.imwrite(fileName,img)
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="firebase_key.json"   
                        db_url='https://test-e7b86.firebaseio.com'    
                        fdb=firebase.FirebaseApplication(db_url,None)
                        # Upload image to Firebase
                        client = storage.Client()
                        bucket = client.get_bucket('test-e7b86.appspot.com')
                        imagePath = "/home/e96031413/tflite1/OK/" + fileName    
                        imageBlob = bucket.blob(fileName)
                        imageBlob.upload_from_filename(imagePath)    # Upload image to firebase
                        imageBlob.make_public()
                        publicURL = imageBlob.public_url
                        firebase_data=[{'label':labels_[label_id],'Score':prob,'Time':local_time,'fileName':fileName,'image_url':publicURL},]
                        for data in firebase_data:
                           fdb.post('bird-data',data)
                        # Save to CSV file with pandas
                        pandas_data = {'Label':labels_[label_id],'Score':prob,'Time':local_time,'fileName':fileName,'image_url':publicURL}
                        df = pd.DataFrame(data=pandas_data,index=[0])
                        df.to_csv('bird_data.csv',mode='a',encoding='utf8')
                
                # Once the image and data have been uploaded to Firebase,
                # Delete the images saved in local
                for i in glob.glob("*.jpg"):
                    os.remove(i)
                      

                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()