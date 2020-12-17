import pickle
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pickle
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import cv2
import numpy as np

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
print("=============================================================================================================")
print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# with open("../checkpoints/checkpoint","rb") as f:
#     ckpt = pickle.load(f)
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
#Convolution2D(no_filters,kernel_height,kernel_width,input_img_shape,activation_func)
classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 4, activation = 'sigmoid'))

classifier.load_weights("weights-improvement.hdf5")

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        '../data/test_set',
        target_size=(128, 128),
        class_mode='categorical',
        batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = classifier.predict_generator(test_generator,steps = nb_samples)
for x in range(len(predict)):
        a = max(predict[x])
        print(a)
        predict[x] = (predict[x]==a)
print(predict)
scores = classifier.evaluate_generator(test_generator,steps = nb_samples)
print(scores)
print(classifier.metrics_names)


