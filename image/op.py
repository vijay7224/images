import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.models import Sequential
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
x_train=x_train/255
x_test=x_test/255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
model=Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(36,activation="relu"))
model.add(Dense(10,activation="softmax"))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
import joblib
joblib.dump(model, 'xyz')
