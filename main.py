import numpy as np
import cv2 as cv
import os
from datetime import datetime 
from uuid import uuid4
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import logging
import threading
import time, traceback
from multiprocessing import Process



from tensorflow import keras
import firebase_admin
from firebase_admin import credentials, initialize_app, storage, firestore
# import PIL
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

cred = credentials.Certificate("smart-home-surveillance-system-firebase-adminsdk-q8gq5-f21253b372.json")
initialize_app(cred ,{'storageBucket': 'smart-home-surveillance-system.appspot.com'})
db = firestore.client()

#! Variables, constants and paths
activity_class_names = ["NotSuspicious","Suspicious"]
date = datetime.now()
filename = "test-video.mp4"
path = "./videos/"
frames_per_second = 10
video_resolution = '720p'

batch_size = 32
img_height = 64
img_width = 64
    
DIMENSIONS = {
    "480p":(640,480),
    "720p":(1280,720),
    "1080p":(1920,1080),
    "4k":(3840,2160),
}

VIDEOTYPES = {
    '.avi' : cv.VideoWriter_fourcc(*'XVID'),
    '.mp4' : cv.VideoWriter_fourcc(*'mp4v')
} 

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
FONTSCALE = 1
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
RED = (0,0,255)
WHITE = (255,255,255)
PAROT = (21,255,100)
FONTS = cv.FONT_HERSHEY_DUPLEX
org = (50, 50)
# fontScale
fontScale = 1
# line thickness of 2 px
thickness = 2
class_names = []

# ----------------------------------------------------------------------------------
# Function to change camera resolution and video resolution
def change_cam_video_resolution(cap, width, height):
    cap.set(cv.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,height)

# Function to get the video type
def get_Video_Type(filename):
    filename , ext = os.path.splitext(filename)
    # print(ext)
    if ext in VIDEOTYPES:
        return VIDEOTYPES[ext]
    return VIDEOTYPES['avi']

# Function to get the video resolution
def get_dimensions(cap, res = '1080'):
    width, height = DIMENSIONS[res]
    if res in DIMENSIONS:
        width, height = DIMENSIONS[res]
    change_cam_video_resolution(cap, width, height)
    return width, height

# function to detect a class of object in a frame
def object_detector(image):
  classes, scores, boxes = yolo_trained_model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
#   print(classes, scores, boxes)
  data_list = []

#   print(len(boxes))
  for (classid, score, box) in zip(classes, scores, boxes):
    color= COLORS[int(classid) % len(COLORS)]
    label = "%s : %f" % (class_names[classid], score)
    # print(box)
    cv.rectangle(image, box, color, 2)
    cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    # getting the data 
    # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
    if classid == 0: # person class id 
        data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        
  return data_list

def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv.getTextSize(text, fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            # print(new_width)
            return scale/10
    return 1
# -------------------------------------------------------------------------------------------

file_date = date.strftime("%m_%d_%Y")
file_time = date.strftime("%H_%M")
fileName = "D_"+file_date+"T_"+file_time+".mp4"



# All you need to do when doing IP cam is to make VideoCapture(0) 
# into VideoCapture("rtsp://username:password@ip-address") good luck
# cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap = cv.VideoCapture("CCTV2.ts")
# cap = cv.VideoCapture("CCTVNORMAL.ts")
# cap = cv.VideoCapture("CCTVWALK.webm")

recording_video_dimensions = get_dimensions(cap,res = video_resolution)

Video_Type_Name = get_Video_Type(fileName)
recoder = cv.VideoWriter(path+filename, Video_Type_Name, frames_per_second, recording_video_dimensions)
filterRec = cv.VideoWriter(path+fileName, Video_Type_Name, frames_per_second, recording_video_dimensions)
isPerson = False
record = False


print(path+filename, Video_Type_Name, frames_per_second, recording_video_dimensions)

print(Video_Type_Name)
print("File Name: "+fileName)
print("Video Resolution: "+str(recording_video_dimensions))

# -------------------------------------------------------------------------------------------

with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
    
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
my_trained_model = tf.keras.models.load_model('./trained_model/my_model.h5')

yolo_trained_model = cv.dnn_DetectionModel(yoloNet)
yolo_trained_model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Show the model architecture
# my_trained_model.summary()
# print(class_names)


    

while True:
    # reading the frames from the video or camera
    ret, frame = cap.read()
    
    # saving the frames into the video
    recoder.write(frame)
    
    if(record):
        filterRec.write(frame)

    
    # cv.putText(frame, file_date+" "+file_time ,(5,150), FONTS, 0.5, RED, thickness)
    if(frame.all() != None):
        detected_objects = object_detector(frame)

        if(detected_objects != None):
            for d in detected_objects:
                if d[0] =='person':
                    x, y = d[2]
                    isPerson = True
                    if(record == False):
                        cv.putText(frame, "Recording",(5,50), FONTS, FONTSCALE, RED, thickness)
                        record = True
                else:
                    isPerson = False
                    record = False

                # cv.rectangle(frame, (x, y-3), (x+150, y+100),BLACK,-1 )
            if isPerson:    
                
                resized_frame = tf.image.resize(frame, (img_height, img_width))
                resized_frame_array_form = tf.keras.utils.img_to_array(resized_frame)
                # Create a batch
                resized_frame_array_form = tf.expand_dims(resized_frame_array_form, 0) 

                predictions = my_trained_model.predict(resized_frame_array_form)
                score = tf.nn.sigmoid(predictions[0])

                # print(predictions[0][0])
                classes = np.where(predictions> 0.5, 1,0)
                prediction_class_color = np.max(classes) > 0.5 and RED or GREEN
                if(record == False and np.max(classes) > 0.5):
                    scale = get_optimal_font_scale("Recording", frame.shape[1])
                    cv.putText(frame, "Recording",(5,50), FONTS, scale, RED, thickness)
                    record = True
                else:
                    record = False
                    
                cv.putText(frame, "{} with a {:.2f} percent confidence."
                .format(activity_class_names[np.max(classes)], 100 * np.max(score)),(5,15), FONTS, FONTSCALE, prediction_class_color, thickness)

        # print(
        #     "This image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(activity_class_names[np.argmax(score)], 100 * np.max(score))
        # )

    frame = cv.resize(frame, DIMENSIONS["480p"])
    cv.imshow('frame',frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
recoder.release()
filterRec.release()
cv.destroyAllWindows()



# print("Uploading the Video.......")
# # Create new token
# new_token = uuid4()

# # Create new dictionary with the metadata
# metadata  = {"firebaseStorageDownloadTokens": new_token}

# bucket = storage.bucket()
# blob = bucket.blob(fileName)
# blob.metadata = metadata
# blob.upload_from_filename(fileName,content_type='video/mp4')


# # Opt : if you want to make public access from the URL
# blob.make_public()
# data = {
#         'name': fileName,
#         'url':'https://firebasestorage.googleapis.com/v0/b/smart-home-surveillance-system.appspot.com/o/'+fileName+'?alt=media&token='+str(new_token),
# }
# db.collection('videos').document().set(data)
# # Add a new doc in collection 'cities' with ID 'LA'
# print("your file url", blob.public_url)
