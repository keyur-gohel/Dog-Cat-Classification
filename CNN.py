# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:06:49 2020

@author: keyur
"""
# Convolution Neural Network

# Building CNN

# Importing Keras Libraries and Packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing CNN
classifier = Sequential()

# Convolution Layer (step-1)
classifier.add(Conv2D(32, (3, 3),
                      input_shape = (64, 64, 3),
                      activation = 'relu'))

# Pooling (step-2)
classifier.add(MaxPooling2D(pool_size = (2, 2))) 

# Adding a Second Convolution layer
classifier.add(Conv2D(32, (3, 3),
                      activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2))) 

# Flatting (step-3)
classifier.add(Flatten())

# Fully Connected Layer (Classic ANN) (step-4)
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile CNN
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Fitting CNN to Data
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)