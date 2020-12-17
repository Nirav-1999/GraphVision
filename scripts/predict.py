import pickle
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
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
img_width, img_height = 128, 128
font = cv2.FONT_HERSHEY_COMPLEX
path = "/home/nirav/Desktop/test_images"
out_dir = "/home/nirav/Desktop/test_images_out"
for f in os.listdir(path):
    file_path = os.path.join(path,f)
    print(file_path)
    _image = cv2.imread(file_path)
    img = image.load_img(file_path, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # print(img.shape)

    predict = classifier.predict(img)
    # print("hi")
    # print(predict.max())
    # print(predict)
    predict = (predict==predict.max())
    if(predict[0][0]):
        print("Bar Chart")
        cv2.putText(_image,"Bar Chart",(10,50),font,1,(0,255,0),2)

    elif(predict[0][1]):
        print("Histogram")
        cv2.putText(_image,"Histogram",(10,50),font,1,(0,255,0),2)

    elif(predict[0][2]):
        print("Line Charts")
        cv2.putText(_image,"Line Charts",(10,50),font,1,(0,255,0),2)

    else:
        print("Pie charts")
        cv2.putText(_image,"Pie charts",(10,50),font,1,(0,255,0),2)
    cv2.imshow(f,_image)
    cv2.imwrite(os.path.join(out_dir,f),_image)
    key = cv2.waitKey()
    if key==27:
        cv2.destroyAllWindows() 

# print(type(predict[0][0]))
