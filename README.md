# Pneumonia-Detection-
The purpose of this project is to create a streamlit website where you can uploud an image of x-ray and the deep learning model predicts if the image depicts pneumonia. 
Additionaly a heatmap of most significant pixels is cerated and displayed blended with the orignial images in order to help potentional users in pnuemonia recognition

# Project Files
model_cnn.py - code used to generate and train a CNN model, it isn't used directly by streamlit website  
heatmap.py - contains class which is used to generate heatmap  
webbapp.py - creates and manages the website functionality  
requirments.txt - libraries required to run the code  

# Future plans for development
Developing more accurate deep learning model (potentially one already existing and tested as efficient)  
Creating a python sript which utilizes Amazon S3 cloud storage services to efficiently download the model on startup  
Using a CDN (Content delivery network) to make the website faster  

# Sources
Heatmap algorithm source: https://github.com/wiqaaas/youtube/blob/master/Deep_Learning_Using_Tensorflow/Demystifying_CNN/Gradient%20Visualization.ipynb  
  
CNN architecture source: https://www.kaggle.com/code/jonaspalucibarbosa/chest-x-ray-pneumonia-cnn-transfer-learning  
