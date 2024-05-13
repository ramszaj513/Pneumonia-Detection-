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
import torch 
import torch.nn as nn
from torchvision import transforms

from heatmap import GradCAM

IMG_SIZE1 = 125
IMG_SIZE2 = 100
MODEL_FILE = "pneumonia_detection_modelv3.h5"

#https://drive.google.com/file/d/1ZgaIquSg1wieAvTLnqPeCUJqX-aH76XR/view?usp=sharing
#https://drive.google.com/file/d/1nKHyI1FC5ECrW9_LRKqfBmoz8xrGqX0I/view?usp=sharing
#https://drive.google.com/uc?/export=download&id=1IGxRUQbh3hii-uCDhynISgBfAvcv4jx7
#https://drive.google.com/file/d/1-NpxaTXb1b4O0BG8UN0qpqvBNzUDe4Zx/view?usp=sharing

# Links to models on google drive
#url_transfer = 'https://drive.google.com/uc?/d/1IGxRUQbh3hii-uCDhynISgBfAvcv4jx7/view?usp=sharing'
#url_greyscale = 'https://drive.google.com/uc?/export=download&id=1ZgaIquSg1wieAvTLnqPeCUJqX-aH76XR'
#url_greyscale2 = 'https://drive.google.com/uc?/export=download&id=1-NpxaTXb1b4O0BG8UN0qpqvBNzUDe4Zx'

#url = url_greyscale2
#repository_url = "ramszaj513/Pneumonia Detection"
#path_to_check = MODEL_FILE

#Function to check if the model already exists in github repository
#def check_github_path(repository_url, path):
#    api_url = f"https://api.github.com/repos/{repository_url}/contents/{path}"
#    response = requests.get(api_url)
#    
#    if (response.status_code == 200):
#        return True  # Path exists
#    else:
#        return False  # Path does not exist
#

#If the model doesn't exist we download it from google drive
#if not check_github_path(repository_url, path_to_check):
#     gdown.download(url, MODEL_FILE)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(12500, 1000)  # Input size: 784, Output size: 128
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(1000, 128)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(128, 1)    # Input size: 128, Output size: 10 (for 10 classes)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.fc3(x)
        x = self.sigmoid3(x)
        return x

# Loading the model
model = SimpleNN()
model.load_state_dict(torch.load("model1.pt"))
model.eval()

st.write("""
# Pneumonia detection app         
""")

def main():
    file_uploaded = st.file_uploader("Choose a file", type = ['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image_displayed = Image.open(file_uploaded).convert("RGB")
        #resized_image = image_displayed.resize((IMG_SIZE, IMG_SIZE))
        #resized_image = np.array(resized_image)
        image = Image.open(file_uploaded).convert("L")
        
        # Getting the predictions
        result, confidence, prediction = predict_image(image)
        
        # Calculating percentage of confidence
        percentage = str(int(confidence*100)) + "%"

        # Displaying the table with results
        data = {
        'Prediction': [result],
        'Confidence': [percentage],
        }
        df = pd.DataFrame(data)
        st.table(df)

        # Creating Slider
        #alpha = st.slider("Transparency level", 0.0, 1.0, 0.5, 0.1)

        # Creating heatmap
        #i = np.argmax(prediction[0])
        #cam = GradCAM(model, i)
        #heatmap = cam.compute_heatmap(preprocess_image(image))

        # Creating blended image
        #heatmap , blended_image = cam.overlay_heatmap(heatmap, resized_image, alpha, cv2.COLORMAP_JET)

        #Displaying the blended image
        figure = plt.figure()
        plt.imshow(image_displayed)
        plt.axis('off')
        st.pyplot(figure)
        
def resize_with_borders(img, width, height):
    img = np.array(img)
    border_v = 0
    border_h = 0
    if (height / width) >= (img.shape[0] / img.shape[1]):
        border_v = int((((height / width) * img.shape[1]) - img.shape[0]) / 2)
    else:
        border_h = int((((width / height) * img.shape[0]) - img.shape[1]) / 2)
    img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
    img = cv2.resize(img, (width, height))
    img = Image.fromarray(np.uint8(img))
    return img

transform = transforms.Compose([
    transforms.Resize((125, 100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),           # Convert image to PyTorch tensor
])



def predict_image(image):
    preprocessed_image = resize_with_borders(image,125,100)
    prediction = model(transform(preprocessed_image).view(-1))[0].item()
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
