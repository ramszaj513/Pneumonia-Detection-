# Pneumonia-Detection-

The purpose of this project is to create a streamlit website where you can uplouad images of x-rays and the deep learning model predicts if the image depicts pneumonia.  
Additionaly a heatmap of most significant pixels is cerated and displayed blended with the orignial images in order to help potentional users in pnuemonia recognition

# The repo contains 3 files: 
model_cnn.py - code used to generate and train a CNN model, it isn't used directly by streamlit website  
heatmap - contains class which is used to generate heatmap  
webbapp - creates and manages the website functions  

# Sources

Heatmap algorithm source: https://github.com/wiqaaas/youtube/blob/master/Deep_Learning_Using_Tensorflow/Demystifying_CNN/Gradient%20Visualization.ipynb  
CNN architecture source: https://www.kaggle.com/code/jonaspalucibarbosa/chest-x-ray-pneumonia-cnn-transfer-learning  
