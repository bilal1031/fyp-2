import numpy as np
import cv2 as cv
import os
# import firebase_admin
from datetime import datetime 
# from firebase_admin import credentials, initialize_app, storage, firestore
from uuid import uuid4
import matplotlib.pyplot as plt
import numpy as np
import os
# import PIL
import tensorflow as tf

from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

# cred = credentials.Certificate("smart-home-surveillance-system-firebase-adminsdk-q8gq5-f21253b372.json")
# initialize_app(cred ,{'storageBucket': 'smart-home-surveillance-system.appspot.com'})
# db = firestore.client()
new_model = tf.keras.models.load_model('./trained_model/my_model.h5')

# Show the model architecture
new_model.summary()
batch_size = 32
img_height = 64
img_width = 64
activity_class_names = ['Assault', 'Burglary', 'NormalVideos', 'Robbery', 'Shooting', 'Stealing']


dt = datetime.now()
filename = "test-video.mp4"
path = "./videos/"
frames_per_second = 30.0
resolution = '480p'

def Change_res(cap, width, height):
    cap.set(3,width)
    cap.set(4,height)
    
Dimensions = {
    "480p":(640,480),
    "720p":(1280,720),
    "1080p":(1920,1080),
    "4k":(3840,2160),
}

Video_Type = {
    '.avi' : cv.VideoWriter_fourcc(*'XVID'),
    '.mp4' : cv.VideoWriter_fourcc(*'mp4v')
} 

def get_Video_Type(filename):
    filename , ext = os.path.splitext(filename)
    print(ext)
    if ext in Video_Type:
        return Video_Type[ext]
    return Video_Type['avi']

def get_dimensions(cap, res = '1080'):
    width, height = Dimensions['480p']
    if res in Dimensions:
        width, height = Dimensions[res]
    Change_res(cap, width, height)
    return width, height

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3

COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
RED = (0,0,255)
 
FONTS = cv.FONT_HERSHEY_COMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 1
# Line thickness of 2 px
thickness = 2

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
    
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# print(class_names)


model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

def object_detector(image):
  classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
#   print(classes, scores, boxes)
  data_list =[]
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



file_date = dt.strftime("%m_%d_%Y")
file_time = dt.strftime("%H_%M")
fileName = "D_"+file_date+"T_"+file_time+".mp4"
print(fileName)

cap = cv.VideoCapture(0, cv.CAP_DSHOW)
dims = get_dimensions(cap,res = resolution)
Video_Type_Name = get_Video_Type(fileName)
print(Video_Type_Name)
rec = cv.VideoWriter(filename, Video_Type_Name, frames_per_second, dims)
filterRec = cv.VideoWriter(fileName, Video_Type_Name, frames_per_second, dims)

while True:
    ret, frame = cap.read()
    rec.write(frame)
    if(frame.all() != None):
        data = object_detector(frame) 
        # print(data)
        if(data != None):
            for d in data:
                if d[0] =='person':
                    x, y = d[2]
                # cv.rectangle(frame, (x, y-3), (x+150, y+100),BLACK,-1 )
                cv.putText(frame, "Recording",(5,50), FONTS, 0.5, RED, thickness)
                filterRec.write(frame)
                img = tf.image.resize(frame, (img_height, img_width))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0) # Create a batch

                predictions = new_model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                cv.putText(frame, "{} with a {:.2f} percent confidence."
                .format(activity_class_names[np.argmax(score)], 100 * np.max(score)),(5,15), FONTS, 0.5, BLACK, thickness)
        # print(
        #     "This image most likely belongs to {} with a {:.2f} percent confidence."
        #     .format(activity_class_names[np.argmax(score)], 100 * np.max(score))
        # )
   
        
    cv.imshow('frame',frame)
    
    key = cv.waitKey(1)
    if key ==ord('q'):
        break

cap.release()
rec.release()
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
