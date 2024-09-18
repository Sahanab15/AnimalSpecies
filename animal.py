import os
import keras # type: ignore
import shutil
import numpy as np
import math
import random
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
sns.set()
from distutils.dir_util import copy_tree # type: ignore
import tensorflow as tf # type: ignore
from keras.layers import Convolution2D # type: ignore
from keras.layers import MaxPooling2D # type: ignore
from keras.layers import Flatten # type: ignore
from keras.layers import Dense # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Dropout # type: ignore
from keras_preprocessing.image import ImageDataGenerator # type: ignore
#created data set using console
source='./input/african-wildlife/'
target='./train_data/'
# shutil.copytree(source, target)
# os.mkdir('test_data')
#created data set using console
source='./input/african-wildlife/'
target='./train_data/'
# shutil.copytree(source, target)
# os.mkdir('test_data')
# create test_data by taking 25% images from data

total_train_images,total_test_images,total_train_classes,total_test_classes=0,0,0,0
path="./train_data/"
for file in os.listdir(path):
    total_train_classes+=1
    total_images=len(os.listdir(path+file+"/"))
    test_image_count=(25/100)*total_images #25% for test and 75% for train
    for i in range(math.ceil(test_image_count)):
        img=random.choice(os.listdir(path+file+'/'))
        shutil.move(path+file+'/'+img,'./test_data/'+file+'/')
        #print(img)
    print(file,total_images,math.ceil(test_image_count))
    total_train_images+=(total_images-math.ceil(test_image_count))
    #print(file,math.ceil(test_image_count))
print("total train images are : ",total_train_images," and total train classes are : ",total_train_classes)
model = Sequential()
# pooling layer where we are doing maxpooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#adding one more convolution layer for better model
model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3),
                        strides=(1,1),
                        padding='same', 
                        activation='relu'
                      ))
model.add(MaxPooling2D(pool_size=(2, 2)))
#dropout regularlization
model.add(Dropout(0.5))
#layer in which we are converting 2d/3d image to 1d image i.e flattening
model.add(Flatten())
# layer: appling relu to give positive output from here our hidden layerrs starts
model.add(Dense(units=20, activation='relu'))
#dropout regularlization
model.add(Dropout(0.5))
# output layer : Since we have to do multi-class classification so we'll apply softmax activation function 
# we have 4 classes of animals so output layer would have that many neurons.
model.add(Dense(units=4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#url : https://keras.io/api/preprocessing/image/ 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        './train_data/',
        target_size=(32,32),
        color_mode="grayscale",
        batch_size=64,
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        './test_data/',
        target_size=(32,32),
        color_mode="grayscale",
        batch_size=64,
        class_mode='categorical')
training_set.class_indices # to see classes of our dataset
history = model.fit(
        training_set,
        steps_per_epoch=(10),
        epochs=1000,
        validation_data=test_set,
        validation_steps=(10))
#Graphing our training and validation
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r', label='Training acc')
plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()
model.save("simple_animal_classification_model.h5")#save model
from keras.models import load_model # type: ignore
model=load_model("simple_animal_classification_model.h5") 
from keras.models import load_model # type: ignore
model=load_model("simple_animal_classification_model.h5") 
from keras.preprocessing import image # type: ignore
test_image = image.load_img("./input/african-wildlife/zebra/001.jpg",target_size=(32,32),color_mode='grayscale')
 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = model.predict(test_image)

my_dict=training_set.class_indices
def get_key(val): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
  
    return "key doesn't exist"

pred=list(result[0])
for i in range(len(pred)):
    if pred[i]!=0:
        print(get_key(i))
