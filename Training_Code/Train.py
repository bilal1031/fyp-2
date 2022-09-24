# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Getting all the categories from the dir
categories = []
path = './Dataset'

for dir in os.listdir(path):
    categories.append(dir)
    
print(categories)
img_height, img_width = 32, 32
batch_size = 20

train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

# Using matplotlib to show 12 images from the dataset of each category
# for category in categories:
#     fig, _ = plt.subplots(3,4)
#     fig.suptitle(category)
#     fig.patch.set_facecolor('xkcd:white')
#     for k, v in enumerate(os.listdir(path+'/'+category)[:12]):
#         img = plt.imread(path+'/'+category+'/'+v)
#         plt.subplot(3, 4, k+1)
#         plt.axis('off')
#         plt.imshow(img)
#     plt.show()
    
    
# Getting the max and min pixels height and width of the dataset image
imageHeight = []
imageWidth = []

# for category in categories:
#     for files in os.listdir(path+'/'+category):
#         imageHeight.append(plt.imread(path+'/'+category+'/'+ files).shape[0])
#         imageWidth.append(plt.imread(path+'/'+category+'/'+ files).shape[1])
#     print(category, ' => height min : ', min(imageHeight), 'width min : ', min(imageWidth))
#     print(category, ' => height max : ', max(imageHeight), 'width max : ', max(imageWidth))
#     imageHeight = []
#     imageWidth = []
    
# initialize the data and labels
data = []
labels = []
imagePaths = []
HEIGHT = 151
WIDTH = 220
N_CHANNELS = 3

# grab the image paths and randomly shuffle them
for k, category in enumerate(categories):
    for f in os.listdir(path+'/'+category):
        imagePaths.append([path+'/'+category+'/'+f, k]) # k=0 : 'dogs', k=1 : 'panda', k=2 : 'cats'

import random
random.shuffle(imagePaths)
print(len(imagePaths))

# loop over the input images
for imagePath in imagePaths[:10000]:
    # load the image, resize the image to be HEIGHT * WIDTH pixels (ignoring
    # aspect ratio) and store the image in the data list
    image = cv2.imread(imagePath[0])
    # image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    data.append(image)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath[1]
    labels.append(label)
    
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

labels = np.array(labels)
print(data.shape, labels.shape)

# Let's check everything is ok
# fig, _ = plt.subplots(3,4)
# fig.suptitle("Sample Input")
# fig.patch.set_facecolor('xkcd:white')
# for i in range(12):
#     plt.subplot(3,4, i+1)
#     plt.imshow(data[i])
#     plt.axis('off')
#     plt.title(categories[labels[i]])
# plt.show()

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

from warnings import filters
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3), activation='relu', input_shape =(64, 64, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=( 3,3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(11))
cnn.add(tf.keras.layers.Activation('relu'))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(5))
cnn.add(tf.keras.layers.Activation('softmax'))

cnn.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])


cnn.fit(trainX, trainY, epochs = 1)
cnn.evaluate(testX, testY)