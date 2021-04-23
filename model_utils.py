import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

INPUT_SHAPE = (32, 32,1)
NUM_CLASSES = 32
BATCH_SIZE = 32
EPOCHS= 13

 # squeeze and exite is a good thing
def squeeze_excite_block2D(filters,input):                      
    seb = tf.keras.layers.GlobalAveragePooling2D()(input)
    seb = tf.keras.layers.Reshape((1, filters))(seb) 
    seb = tf.keras.layers.Dense(filters//32, activation='relu')(seb)
    seb = tf.keras.layers.Dense(filters, activation='sigmoid')(seb)
    seb = tf.keras.layers.multiply([input, seb])
    return seb

def make_model():
  s = tf.keras.Input(shape=INPUT_SHAPE) 
  x = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(s)
  x = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
  x = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = squeeze_excite_block2D(128,x)

  x = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
  x = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
  x = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = squeeze_excite_block2D(128,x)
  x = tf.keras.layers.AveragePooling2D(2)(x)

  x = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
  x = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
  x = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = squeeze_excite_block2D(128,x)
  x = tf.keras.layers.AveragePooling2D(2)(x)

  x = tf.keras.layers.concatenate([tf.keras.layers.GlobalMaxPooling2D()(x),tf.keras.layers.GlobalAveragePooling2D()(x)])

  x = tf.keras.layers.Dense(32,activation='softmax')(x) 
  

  return tf.keras.Model(inputs=s, outputs=x)


def data_generator():
  
    traindata_gen = ImageDataGenerator(rotation_range=15, 
                                        width_shift_range=0.1, 
                                        height_shift_range=0.1,
                                        rescale = 1.0/255, 
                                        shear_range=0.2, 
                                        zoom_range = 0.2,
                                        validation_split=0.2)
    
    trainGenerator = traindata_gen.flow_from_directory(DATA_DIR, 
                                                    target_size=(32,32),
                                                    batch_size=BATCH_SIZE,
                                                    color_mode="grayscale",
                                                    class_mode="categorical",
                                                    subset="training")
   
    validGenerator = traindata_gen.flow_from_directory(DATA_DIR, 
                                                    target_size=(32,32),
                                                    batch_size=BATCH_SIZE,
                                                    color_mode="grayscale",
                                                    class_mode="categorical",
                                                    subset="validation")

    
    return trainGenerator,validGenerator



if __name__ == "__main__":
    model = make_model()
    trainGenerator,validGenerator=data_generator()
    
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
    
    checkpoint_cb = ModelCheckpoint("model-2.h5", save_best_only=True, monitor = 'val_loss', mode='min')

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 3, min_lr = 1e-6, mode = 'min', verbose = 1)

    history = model.fit(trainGenerator, epochs=13, validation_data=validGenerator, callbacks=[es, checkpoint_cb, reduce_lr])  



  
