import cv2 as cv
import numpy as np
from PIL import Image
def resize_with_borders(img, width, height):
    border_v = 0
    border_h = 0
    if (height / width) >= (img.shape[0] / img.shape[1]):
        border_v = int((((height / width) * img.shape[1]) - img.shape[0]) / 2)
    else:
        border_h = int((((width / height) * img.shape[0]) - img.shape[1]) / 2)
    img = cv.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv.BORDER_CONSTANT, 0)
    img = cv.resize(img, (width, height))
    return img

def preprocess(img):
    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    equ_img = cv.equalizeHist(img)
    gauss_img = cv.GaussianBlur(equ_img, (5, 5), 0)
    resized_img = resize_with_borders(gauss_img, 384, 384)
    return Image.fromarray(resized_img)