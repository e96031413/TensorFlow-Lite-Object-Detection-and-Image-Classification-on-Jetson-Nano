# TensorFlow Lite Object Detection and Image Classification on Jetson Nano

Live Object Detection and Image Classification System (PiCamera+OpenCV+TensorFlow Lite+Firebase) on Jetson Nano

---

#### A Python script that:

[1] Load Pre-trained (Object Detection) and Self-trained (Image Classification)TFLite Model with Argument.

[2] Read image from PiCamera with OpenCV to do Real-Time Object Detection.

[3] If detect specific object ("bird" in the code), save the image.

[4] Use Self-trained Model to do Image Classification on the image with OpenCV.

[5] Upload the Image and classification result (LabelName, ScoreValue, Time, Pubic-Access Image Url) to Firebase Database

[6] Save the above result (LabelName, ScoreValue, Time, Pubic-Access Image Url) as a csv file with append mode.

[7] Once the image and data have been uploaded to Firebase, delete the local images to prevent running out of disk space.


## Usage

```
python3 object_detection_and_image_classification.py
```

## Project Structure


### ***Folder:***

**Sample_TFLite_model/**:

Contain the object detection model and label

### ***File:***

**object_detection_and_image_classification.py**:

Our main program of this project.

**TFLite_Read_Image.py**:

Read Image with OpenCV to Image Classification.

**test.tflite**:

Image Classification TFLite Model.

**test.txt**:

Image Classification TFLite label.

**firebase_key.json**:

If you want to use firebase to store your data, you should have it.

You can learn how to get one and the API usage, please refer to the following links:

[Learning Firebase(1)：Create Your First Project](https://medium.com/@yanweiliu/learning-firebase-1-create-your-first-project-b5b5e352198c)

[Learning Firebase(2)：CRUD Our Database with Python](https://medium.com/@yanweiliu/learning-firebase-2-crud-our-database-with-python-526c3e46fd8b)

[Learning Firebase(3)：Upload Image to Firebase with Python](https://medium.com/@yanweiliu/learning-firebase-3-upload-image-to-firebase-with-python-32fae4ebc26a)



