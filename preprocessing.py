import cv2 as cv
import os
import sys

# Use with arguments: source directory path, destination directory path, width, height
# Example: python preprocessing.py dataset preprocessed 500 500
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


def process(img, width, height):
    equ_img = cv.equalizeHist(img)
    gauss_img = cv.GaussianBlur(equ_img, (5, 5), 0)
    resized_img = resize_with_borders(gauss_img, width, height)
    return resized_img


def recursive_traversal(src, dest, width, height):
    for root, dirs, files in os.walk(src):
        for name in dirs:
            if not os.path.exists(f"{dest}/{name}"):
                os.mkdir(f"{dest}/{name}")
            recursive_traversal(f"{src}/{name}", f"{dest}/{name}", width, height)
        for name in files:
            img = cv.imread(f"{src}/{name}", cv.IMREAD_GRAYSCALE)
            processed_img = process(img, width, height)
            cv.imwrite(f"{dest}/{name[:name.index('.')]}.png", processed_img)
        break


source_path = sys.argv[1]
dest_path = sys.argv[2]
dest_width = int(sys.argv[3])
dest_height = int(sys.argv[4])

if not os.path.exists(dest_path):
    os.mkdir(dest_path)
recursive_traversal(source_path, dest_path, dest_width, dest_height)
