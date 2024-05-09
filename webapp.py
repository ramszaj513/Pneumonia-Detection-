import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import os
import h5py
import cv2
import pandas as pd
import gdown

IMG_SIZE = 224
MODEL_DOWNLOADED = False
MODEL_FILE = "pneumonia_detection_model.h5"

#https://drive.google.com/file/d/1nKHyI1FC5ECrW9_LRKqfBmoz8xrGqX0I/view?usp=sharing
url = 'https://drive.google.com/uc?/export=download&id=1IGxRUQbh3hii-uCDhynISgBfAvcv4jx7'

# Getting the model from google drive
if(MODEL_DOWNLOADED == False):
    gdown.download(url,MODEL_FILE)
    MODEL_DOWNLOADED = True

model = load_model(MODEL_FILE, compile=False)

st.write("""
# Pneumonia detection app         
""")

def main():
    file_uploaded = st.file_uploader("Choose a file", type = ['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded).convert("RGB")
        
        # Getting the predictions
        result, propability = predict_image(image)
        
        # Display the table with results
        data = {
        'Prediction': [result],
        'Certainty': ["{:.2f}".format(propability[0][0])],
        }
        df = pd.DataFrame(data)
        st.table(df)
        
        # Display the image
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        
def preprocess_image(image):
    resized_image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(resized_image)
    image = np.expand_dims(image_array, axis=0)
    image = image / 255.0  
    return image
    
def predict_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    if prediction > 0.5:
        prediction = calculate_certainty(prediction)
        return "PNEUMONIA", prediction
    else:
        prediction = calculate_certainty(prediction)
        return "NORMAL", prediction
    
def calculate_certainty(prediction):
    entropy = - (prediction * np.log(prediction) + (1 - prediction) * np.log(1 - prediction))
    certainty = 1 - entropy
    return certainty
    
if __name__ == "__main__":
    main()  
