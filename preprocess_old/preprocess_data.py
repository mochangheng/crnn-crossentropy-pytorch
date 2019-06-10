import os
import glob
import random

import numpy as np
import pandas as pd


# Settings
IMG_PATH = 'cell_images/training_set/'
LABELS = 'cell_images/training_set_values.txt'
OUT_ALPHABET = 'crnn_labels/alphabet.txt'
OUT_LABELS = 'crnn_labels/train.txt'
OUT_VAL_LABELS = 'crnn_labels/dev.txt'
OUT_ALL_LABELS = 'crnn_labels/all.txt'
FRAC = 0.75


# Import images
all_images = glob.glob(os.path.join(IMG_PATH, '*.jpg'))
all_images.sort()

df = pd.read_csv(LABELS, sep=';')


# Get alphabet
nums_list = df['value'].values.tolist()
digits_list = [list(item) for item in nums_list]
flat_list = [item for sublist in digits_list for item in sublist]
set(flat_list)

alphabet = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-']
alphabet_dict = {
    ' ': '0',
    '0': '1',
    '1': '2',
    '2': '3',
    '3': '4',
    '4': '5',
    '5': '6',
    '6': '7',
    '7': '8',
    '8': '9',
    '9': '10',
    '.': '11',
    '-': '12'
}

with open(OUT_ALPHABET, 'w') as f:
    for line in alphabet[1:]:
        f.write(f'\n{line}')


# Get labels
columns_list = []
for index, row in df.iterrows():
    row_list = []
    row_list.append(row[0])
    for char in row[1]:
        row_list.append(alphabet_dict[char])

    columns_list.append(row_list)

with open(OUT_ALL_LABELS, 'w') as f:
    for row_list in columns_list:
        line = ' '.join(row_list)
        f.write(f'{line}\n')

random.seed(1)
random.shuffle(columns_list)
train_list = columns_list[:int(len(columns_list) * FRAC)]
test_list = columns_list[int(len(columns_list) * FRAC):]

with open(OUT_LABELS, 'w') as f:
    for row_list in train_list:
        line = ' '.join(row_list)
        f.write(f'{line}\n')

with open(OUT_VAL_LABELS, 'w') as f:
    for row_list in test_list:
        line = ' '.join(row_list)
        f.write(f'{line}\n')
