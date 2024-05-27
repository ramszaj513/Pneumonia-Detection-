import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import torch 

import torchvision
from torchvision import transforms
import preprocess

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image



model = torchvision.models.efficientnet_v2_s()
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=2, bias=True)
device = torch.device('cpu')
model.load_state_dict(torch.load("model_efficientnet.pth", map_location=device))
model.eval()

logo_path = "log_projekt_cut.png" # Path to logo image file
logo = Image.open(logo_path) # Open the image file

col1, mid, col2 = st.columns([10,1,2])
with col1:
    st.write(""" # Pneumonia detection app """)
with col2:
   st.image(logo, width=100)

def main():
    file_uploaded = st.file_uploader("Choose a file", type = ['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image_displayed = Image.open(file_uploaded).convert("RGB")
        #resized_image = image_displayed.resize((IMG_SIZE, IMG_SIZE))
        #resized_image = np.array(resized_image)
        image = Image.open(file_uploaded).convert("RGB")
        image = preprocess.preprocess(image)
        image = Image.merge("RGB", (image, image, image))
        # Getting the predictions
        result, confidence, prediction = predict_image(image)
        
        # Calculating percentage of confidence
        percentage = str(confidence)) + "%"

        # Displaying the table with results
        data = {
        'Prediction': [result],
        'Confidence': [percentage],
        }
        df = pd.DataFrame(data)
        st.table(df)

        # Creating Slider
        alpha = st.slider("Transparency level", 0.0, 1.0, 0.5, 0.1)

        # Creating heatmap
        targets = [ClassifierOutputTarget(prediction)]
        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        rgb_image = image
        img_float = np.array(rgb_image) / 255
        input_tensor = preprocess_image(img_float)
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)

        # Creating blended image
        cam_image = show_cam_on_image(img_float, grayscale_cams[0, :], use_rgb=True, image_weight=(1 - alpha))

        #Displaying the blended image
        figure = plt.figure()
        plt.imshow(cam_image)
        plt.axis('off')
        st.pyplot(figure)

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
                               )



def predict_image(image):
    prediction = model(transform(image).unsqueeze(0))
    _, predicted = torch.max(prediction, 1)
    certainty = calculate_certainty(prediction)
    if predicted == 1:
        return "PNEUMONIA", certainty, prediction
    else:
        return "NORMAL", certainty, prediction
    
def calculate_certainty(prediction):
    vec = prediction
    vec = vec.detach().numpy()
    vec = (np.exp(vec) / np.sum(np.exp(vec))) * 100
    return vec.max()


if __name__ == "__main__":
    main()  
