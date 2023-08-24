#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as pltPIL
import streamlit as st

from PIL import Image

import tensorflow as tf


# In[2]:


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical


# In[3]:


#Validation and Training data
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

#Values are RGB pixels dividing by 255 returns 0-1, better for neural network
X_train = X_train /255
X_val = X_val /255

y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)


# In[5]:


model = Sequential([
    #32 pixels x 32 pixels, 3 color channels RGB
    Flatten(input_shape=(32, 32, 3)),
    # 1000 neurons
    Dense(1000, activation='relu'),
    # 10 neurons, soft max function for all probabilities for possible categories
    Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#train for 10 epochs,
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))
model.save('cifar10_model.h5')


# In[ ]:




