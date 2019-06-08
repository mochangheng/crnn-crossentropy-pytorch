import os
import glob

import numpy as np
import pandas as pd

import cv2
from PIL import Image


# Settings
IMG_PATH = 'cell_images/training_set/'
OUT_PATH = 'processed_images/training_set/'
IMG_VAL_PATH = 'cell_images/validation_set/'
OUT_VAL_PATH = 'processed_images/validation_set/'
LABELS = 'cell_images/training_set_values.txt'


# Import data
df = pd.read_csv(LABELS, sep=';')

all_images = glob.glob(os.path.join(IMG_PATH, '*.jpg'))
all_images.sort()


# Preprocess boxes
count = 0
for i, image in enumerate(all_images):
    im = cv2.imread(image)
    im = cv2.resize(im, (0,0), fx=4, fy=4)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (9, 9), 0)

    ret, im_th = cv2.threshold(im_gray, 120, 255, cv2.THRESH_OTSU)
    im_th = cv2.bitwise_not(im_th)

    y = im_th.shape[0]
    x = im_th.shape[1]

    im_th2 = im_th[int(0.1*y):int(0.9*y), int(0.1*x):int(0.9*x)]

    # kernel = np.ones((5, 5), np.uint8)
    # im_th2 = cv2.dilate(im_th2, kernel, iterations=4)
    # im_th2 = cv2.erode(im_th2, kernel, iterations=0)
    # plt.imshow(im_th2)

    ctrs, hier = cv2.findContours(im_th2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    if len(rects) == len(df.iloc[i][1]):
        count += 1

print(count/len(all_images))
