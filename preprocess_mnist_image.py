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


def invert(image):
    image = cv2.bitwise_not(image)
    return image


def noisen(image, mu=80, sigma=30):
    image = image.copy()

    gaussian = 255 - np.abs(np.random.normal(mu, sigma, (image.shape[0], image.shape[1]))).astype(np.uint8)
    image[:, :, 0] = np.minimum(image[:, :, 0], gaussian)
    image[:, :, 1] = np.minimum(image[:, :, 1], gaussian)
    image[:, :, 2] = np.minimum(image[:, :, 2], gaussian)

    gaussian = np.abs(np.random.normal(mu, sigma, (image.shape[0], image.shape[1]))).astype(np.uint8)
    image[:, :, 0] = np.maximum(image[:, :, 0], gaussian)
    image[:, :, 1] = np.maximum(image[:, :, 1], gaussian)
    image[:, :, 2] = np.maximum(image[:, :, 2], gaussian)

    return image


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
IMG_PATH = 'mnist_images/training_set/'
OUT_PATH = 'mnist_processed_images/training_set/'
IMG_VAL_PATH = 'mnist_images/validation_set/'
OUT_VAL_PATH = 'mnist_processed_images/validation_set/'


# Import images
all_images = glob.glob(os.path.join(IMG_PATH, '*.jpg'))
all_images.sort()

all_val_images = glob.glob(os.path.join(IMG_VAL_PATH, '*.jpg'))
all_val_images.sort()


# Preprocess images
for image in all_images:
    name = image.split('/')[-1].replace('.png', '')
    filename = f'{OUT_PATH}{name}.jpg'

    image = cv2.imread(image)
    image = noisen(image)
    inverted_image = invert(image)
    cv2.imwrite(filename, inverted_image)

for image in all_val_images:
    name = image.split('/')[-1].replace('.jpg', '')
    filename = f'{OUT_VAL_PATH}{name}.jpg'

    image = cv2.imread(image)
    image = noisen(image)
    inverted_image = invert(image)
    cv2.imwrite(filename, inverted_image)
