# %% parse arguments

from argparse import ArgumentParser
from os import path, mkdir, environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # make tensorflow less verbose

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout, GaussianNoise
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB
from glob import glob
import logging as log
from sys import stdout
import pandas as pd
import numpy as np
from utils.evaluation_callback import EvaluationCallback
from utils.data import load_dataset
from utils.constants import *
from utils.management import *


modelName = 'Custom2'
modelFolderPath = path.join('models', modelName)
modelDataPath = path.join(modelFolderPath, 'data.csv')
modelExists = check_and_create_folder(modelFolderPath)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler(path.join(modelFolderPath, 'training.log')),
        log.StreamHandler(stdout)
    ]
)

check_and_create_folder('models') # make sure model directory exists

printSeparator()


# %% create model

INIT_LR = 1e-3
BS = 8
epochs=10
# build model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_DIMENSIONS_3D))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.50))
model.add(layers.Dense(3, activation="sigmoid"))
model.summary() # prints model summary

optimizer = Adam(lr=INIT_LR, decay=INIT_LR /epochs)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

# %% train model



# %% save model

modelPath = path.join(modelFolderPath, f'epoch_none.h5')
csvPath = path.join(modelFolderPath, 'data.csv')
log.info(f'saving model to: "{modelPath}", saving csv to: "{csvPath}"')
model.save(modelPath, save_format='h5')
df = pd.DataFrame({'loss':[],'accuracy':[],'val_loss':[],'val_accuracy':[]})  # ,loss,accuracy,val_loss,val_accuracy
df.to_csv(csvPath)


