# Pneumonia-Detection-

## Overview
The purpose of this project is to create a streamlit website where you can uploud an image of x-ray and the deep learning model predicts if the image depicts pneumonia. 
Additionaly a heatmap of most significant pixels is created and displayed blended with the orignial image in order to help potentional users in pnuemonia recognition

## Project Files
model_cnn.py - code used to generate and train the CNN model, it isn't used directly by the streamlit website    
heatmap.py - contains class which is used to generate heatmap by webapp.py   
webbapp.py - creates and manages the website functionality    
requirments.txt - libraries required to run the code    
pneumonia_detection_modelv3.h5 - trained CNN model used by webapp.py  

## Compability
The code should work with neural networks with .h5 and .keras extension

## Future plans for development
Developing more accurate deep learning model (potentially one already created and tested as efficient)  
Creating a python sript which utilizes cloud storage services to efficiently download the model on startup (if the model exeds 100MG)

## Sources
Heatmap algorithm source: https://github.com/wiqaaas/youtube/blob/master/Deep_Learning_Using_Tensorflow/Demystifying_CNN/Gradient%20Visualization.ipynb  
  
CNN architecture source: https://www.kaggle.com/code/jonaspalucibarbosa/chest-x-ray-pneumonia-cnn-transfer-learning  
