import os
import glob

import numpy as np
import pandas as pd

import cv2
from PIL import Image


# CV2 functions
def binarize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return binary


def min_resize(image, min_size=32):
    if image.width < min_size or image.height < min_size:
        if image.height > image.width:
            factor = min_size / image.height
        else:
            factor = min_size / image.width
        tn_image = image.resize((int(image.width * factor), int(image.height * factor)))
        return tn_image
    else:
        return image


# Settings
IMG_PATH = 'cell_images/training_set/'
OUT_PATH = 'processed_images/training_set/'
IMG_VAL_PATH = 'cell_images/validation_set/'
OUT_VAL_PATH = 'processed_images/validation_set/'


# Import images
all_images = glob.glob(os.path.join(IMG_PATH, '*.jpg'))
all_images.sort()


# Preprocess images
for image in all_images:
    name = image.split('/')[-1].replace('.jpg', '')
    filename = f'{OUT_PATH}{name}.jpg'

    # image = Image.open(image)
    # resized_image = min_resize(image)
    # binarized_image = binarize(np.array(resized_image))
    image = cv2.imread(image)
    binarized_image = binarize(image)
    cv2.imwrite(filename, binarized_image)

all_val_images = glob.glob(os.path.join(IMG_VAL_PATH, '*.jpg'))
all_val_images.sort()

for image in all_val_images:
    name = image.split('/')[-1].replace('.jpg', '')
    filename = f'{OUT_VAL_PATH}{name}.jpg'

    # image = Image.open(image)
    # resized_image = min_resize(image)
    # binarized_image = binarize(np.array(resized_image))
    image = cv2.imread(image)
    binarized_image = binarize(image)
    cv2.imwrite(filename, binarized_image)
