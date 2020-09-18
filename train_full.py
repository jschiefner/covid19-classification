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
from utils.constants import CLASSES, IMG_DIMENSIONS_3D

# set GPU configs, might have to comment out
# these lines if you're working on a cpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

parser = ArgumentParser()
parser.add_argument('dataset', help='path to input folder')
parser.add_argument('-e', '--epochs', default=30, type=int, help='specify how many epochs the network should be trained at max, defaults to 30')
args = vars(parser.parse_args())
# args = {'dataset': '.', 'epochs': 25}  # TODO: comment out

# set modelName
modelName = 'Custom'
modelFolderPath = path.join('models', modelName)

# metadata check # TODO: put this in utils
if not path.exists(args['dataset']):
    print(f'[ERROR] the path "{args["dataset"]}" does not exist. Please supply a valid input folder.')
    exit(1)

# check if necessary folder exists or create it
if not path.isdir(modelFolderPath):
    mkdir(modelFolderPath)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler(path.join('models', modelName, 'training.log')),
        log.StreamHandler(stdout)
    ]
)

# %% load data # TODO: put this in utils

log.info('prepare data')

trainX, valX, trainY, valY, testData, testLabels = load_dataset(args['dataset'])


# %% create model

INIT_LR = 1e-3
BS = 8

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

model.summary()

optimizer = Adam(lr=INIT_LR, decay=INIT_LR / args['epochs'])
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

model.summary()

# %% train model

trainAug = ImageDataGenerator() # TODO: enable for benchmark training
model.fit(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(valX, valY),
    validation_steps=len(valX) // BS,
    epochs=args['epochs'],
)

# %% after action

# save history and epochs for later
# as history gets deleted when model.predict() is called
history = model.history.history.copy()
epochs = len(model.history.epoch)

# %% validate model

predictions = np.argmax(model.predict(testData, batch_size=BS), axis=1)
report = classification_report(testLabels.argmax(axis=1), predictions, target_names=CLASSES) # TODO: do we even need this?
log.info(f'Evaluating trained network\n{report}')

# use as cm[true, pred], from_left: true, from_top: predict
cm = confusion_matrix(testLabels.argmax(axis=1), predictions)
log.info(f'confusion matrix\n{cm}')
