#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf

from PIL import Image


# In[2]:


def main():
    st.title('Cirfar10 Web Classifier')
    st.write('Upload any image that you think fits into one of the classes and see if the prediction is correct')
    
    file = st.file_uploader('Please upload an image', type=['jpg','png'])
    
    #if we have file, then preform, else let them know they haven't uploaded
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True) 
        
        #we have to resize, because our model is based on 32 x 32 pixels
        resized_image = image.resize((32,32))
        
        #reprocessed image to fit for our model
        img_array = np.array(resized_image)/255
        
        #1 image, 32 x 32 pixels, 3 color channels
        img_array = img_array.reshape((1, 32, 32,3))
        
        model = tf.keras.models.load_model('cifar10_model.h5')
        
        predictions = model.predict(img_array)
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        #showing barplot to predictions of each classes
        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_classes))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_classes)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title('CIFAR10 Predictions')
        
        #plotting it in our streamlit application
        st.pyplot(fig)
    else:
        st.text('You have not uploaded an image yet')


# In[ ]:


if __name__ == '__main__':
    main()

