import os
import glob
import random

import numpy as np
import pandas as pd


# Settings
OUT_LABELS = 'mnist_images/training_set_values.txt'
OUT_VAL_LABELS = 'mnist_images/validation_set_values.txt'
IMG_PATH = 'mnist_images/training_set/'
IMG_VAL_PATH = 'mnist_images/validation_set/'


# Import images
all_images = glob.glob(os.path.join(IMG_PATH, '*.jpg'))
all_images.sort()

all_val_images = glob.glob(os.path.join(IMG_VAL_PATH, '*.jpg'))
all_val_images.sort()


# Preprocess images
labels = []
for image in all_images:
    name = image.replace('.jpg', '')
    filename = f'{name}.jpg'
    labels.append((filename, name[-1]))

df = pd.DataFrame(labels, columns=['filename', 'value'])
df.to_csv(OUT_LABELS, index=None, sep=';')

val_labels = []
for image in all_val_images:
    name = image.replace('.jpg', '')
    filename = f'{name}.jpg'
    val_labels.append((filename, name[-1]))

val_df = pd.DataFrame(val_labels, columns=['filename', 'value'])
val_df.to_csv(OUT_VAL_LABELS, index=None, sep=';')
