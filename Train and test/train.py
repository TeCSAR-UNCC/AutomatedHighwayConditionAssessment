#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
import os
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16


#VGG 16 as feature extractor
vgg_conv = VGG16(weights= 'imagenet',
                 include_top=False,
                 input_shape=(224, 224, 3))


#Defining which layers of VGGNet to be trained
for layer in vgg_conv.layers[0:10]:
	layer.trainable = False


#Batch generation and data preparation
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


def data_prep(n_im,set_dir):
    set_data = np.zeros(shape=(n_im, 224, 224, 3))
    set_labels = np.zeros(shape=(n_im,12)) 

    set_generator = datagen.flow_from_directory(
        set_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    i = 0
    for inputs_batch, labels_batch in set_generator:
        data_batch = inputs_batch
        set_data[i * batch_size : (i + 1) * batch_size] = data_batch
        set_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        print(i)
        if i * batch_size >= n_im:
            break

    set_data = np.reshape(set_data, (n_im, 224, 224, 3))
    return set_data,set_labels



train_dir = './dataset/train'
valid_dir = './dataset/validation'
Train_num = 16428
Valid_num = 3997



train_data, train_labels = data_prep(Train_num,train_dir)
valid_data, valid_labels = data_prep(Valid_num,valid_dir)


# Defining the classifier layers and createing the end to end model
x = layers.Flatten()(vgg_conv.output) 
x = layers.Dense(512, activation='relu')(x) 
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(12, activation = 'softmax')(x)
final_model = keras.Model(input = vgg_conv.input, output = predictions)  #Considering the input image as input of the model to include all layers of VGGNet and classifier as an end to end model   



#loss calculation, training and testing
final_model.compile(optimizer=optimizers.RMSprop(lr=2e-7),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = final_model.fit(train_data,
                    train_labels,
                    epochs=400,
                    batch_size=batch_size,
                    validation_data=(valid_data,valid_labels))



print(history.history.keys())


#Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


#Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


#Saving the model in two ways
#First method of saving (Architecture and weights all in one file)
final_model.save("saved-model.h5")

#Second method of saving the model (separate weights and architecture)
final_model.save_weights('model_weights.h5')

model_json = final_model.to_json()
with open("model_architecture.json", "w") as json_file:
	json_file.write(model_json)

print('model is saved')

