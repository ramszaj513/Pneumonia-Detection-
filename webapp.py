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
from pathlib import Path
import requests

from heatmap import GradCAM

IMG_SIZE = 224
MODEL_FILE = "pneumonia_detection_modelv2.h5"

#https://drive.google.com/file/d/1ZgaIquSg1wieAvTLnqPeCUJqX-aH76XR/view?usp=sharing
#https://drive.google.com/file/d/1nKHyI1FC5ECrW9_LRKqfBmoz8xrGqX0I/view?usp=sharing
#https://drive.google.com/uc?/export=download&id=1IGxRUQbh3hii-uCDhynISgBfAvcv4jx7

# Links to models on google drive
url_transfer = 'https://drive.google.com/uc?/d/1IGxRUQbh3hii-uCDhynISgBfAvcv4jx7/view?usp=sharing'
url_greyscale = 'https://drive.google.com/uc?/export=download&id=1ZgaIquSg1wieAvTLnqPeCUJqX-aH76XR'

url = url_greyscale
repository_url = "ramszaj513/Pneumonia Detection"
path_to_check = MODEL_FILE

def check_github_path(repository_url, path):
    api_url = f"https://api.github.com/repos/{repository_url}/contents/{path}"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        return True  # Path exists
    else response.status_code == 404:
        return False  # Path does not exist

if not check_github_path(repository_url, path_to_check):
     gdown.download(url, MODEL_FILE)

# Getting the model from google drive
#if not Path(MODEL_FILE).is_file:
#   gdown.download(url, MODEL_FILE)

# Loading the model
model = load_model(MODEL_FILE, compile=False)

st.write("""
# Pneumonia detection app         
""")

def main():
    file_uploaded = st.file_uploader("Choose a file", type = ['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image_displayed = Image.open(file_uploaded).convert("RGB")
        resized_image = image_displayed.resize((IMG_SIZE, IMG_SIZE))
        resized_image = np.array(resized_image)
        image = Image.open(file_uploaded).convert("L")
        
        # Getting the predictions
        result, confidence, prediction = predict_image(image)
        
        # Displaying the table with results
        data = {
        'Prediction': [result],
        'Confidence': ["{:.2f}".format(confidence[0][0])],
        }
        df = pd.DataFrame(data)
        st.table(df)

        # Creating Slider
        alpha = st.slider("Transparency level", 0.0, 1.0, 0.5, 0.1)

        # Creating heatmap
        i = np.argmax(prediction[0])
        cam = GradCAM(model, i)
        heatmap = cam.compute_heatmap(preprocess_image(image))

        # Creating blended image
        heatmap , blended_image = cam.overlay_heatmap(heatmap, resized_image, alpha, cv2.COLORMAP_JET)

        #Displaying the blended image
        figure = plt.figure()
        plt.imshow(blended_image)
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
        certainty = calculate_certainty(prediction)
        return "PNEUMONIA", certainty, prediction
    else:
        certainty = calculate_certainty(prediction)
        return "NORMAL", certainty, prediction
    
def calculate_certainty(prediction):
    entropy = - (prediction * np.log(prediction) + (1 - prediction) * np.log(1 - prediction))
    certainty = 1 - entropy
    return certainty

if __name__ == "__main__":
    main()  

