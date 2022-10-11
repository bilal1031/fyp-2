from ast import While
from asyncio.windows_events import NULL
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
from threading import Thread, Lock
import time, traceback
from multiprocessing import Process
from camvideostream import CamVideoStream



from tensorflow import keras
import firebase_admin
from firebase_admin import credentials, initialize_app, storage, firestore
# import PIL
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

cred = credentials.Certificate("smart-home-surveillance-system-firebase-adminsdk-q8gq5-f21253b372.json")
initialize_app(cred ,{'storageBucket': 'smart-home-surveillance-system.appspot.com'})
db = firestore.client()



def every(delay, task, args=NULL):
  next_time = time.time() + delay
  while True:
    time.sleep(max(0, next_time - time.time()))
    try:
      if args == NULL:
        task()
      else:
        task(args)
    except Exception:
      traceback.print_exc()
      # in production code you might want to have this instead of course:
      # logger.exception("Problem while executing repetitive task.")
    # skip tasks if we are behind schedule:
    next_time += (time.time() - next_time) // delay * delay + delay
    
    
def foo():
  print("foo", time.time())

def uploader(fileName):

    print("Uploading the Video.......")
    # Create new token
    new_token = uuid4()

    # Create new dictionary with the metadata
    metadata  = {"firebaseStorageDownloadTokens": new_token}

    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.metadata = metadata
    blob.upload_from_filename(fileName,content_type='video/mp4')


    # Opt : if you want to make public access from the URL
    blob.make_public()
    data = {
            'name': fileName,
            'url':'https://firebasestorage.googleapis.com/v0/b/smart-home-surveillance-system.appspot.com/o/'+fileName+'?alt=media&token='+str(new_token),
    }
    db.collection('videos').document().set(data)
    # Add a new doc in collection 'cities' with ID 'LA'
    print("your file url", blob.public_url)


# threading.Thread(target=lambda: every(5, foo)).start()
# threading.Thread(target=lambda: every(5, uploader,"CCTV.mp4")).start()


vs = CamVideoStream(0,320,240)
flag = False
vs.start()

vs.show()


# https://superfastpython.com/thread-share-variables/