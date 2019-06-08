import os
import glob

from PIL import Image, ImageDraw
import pytesseract
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import pandas as pd


# CV2 preprocess
def binarize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return binary


# Import images
IMG_PATH = 'cell_images/training_set/'
LABELS = 'cell_images/training_set_values.txt'
OUT_PATH = 'processed_images/training_set/'

all_images = glob.glob(os.path.join(IMG_PATH, '*.jpg'))
all_images.sort()

df = pd.read_csv(LABELS, sep=';')


# Get all characters
nums_list = df['value'].values.tolist()
digits_list = [list(item) for item in nums_list]
flat_list = [item for sublist in digits_list for item in sublist]
set(flat_list)


# Preprocess images
for image in all_images:
    name = image.split('/')[-1].replace('.jpg', '')
    filename = f'{OUT_PATH}{name}.png'

    image = cv2.imread(image)
    binarized_image = binarize(image)
    cv2.imwrite(filename, binarized_image)
