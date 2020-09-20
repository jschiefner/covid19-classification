# %% parse arguments

from argparse import ArgumentParser
from os import path, mkdir, environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # make tensorflow less verbose
from utils.management import *
from utils.constants import *
from utils.base_models import *


parser = ArgumentParser()
parser.add_argument('dataset', help='path to input folder')
parser.add_argument("-m", "--model", default="VGG16", help=f"specify optional network. {get_all_model_names()}")
parser.add_argument('-e', '--epochs', default=30, type=int, help='specify how many epochs the network should be trained at max, defaults to 30')
args = vars(parser.parse_args())
# args = {'dataset': '.', 'model': 'VGG16', 'epochs': 25} # TODO: comment out

# metadata check
check_if_exists_or_exit(args['dataset'])


# model check
modelFunc = get_model_by_name(str(args['model'])) # returns None if model does not exist
if modelFunc is None:
    if not check_if_custom_model_name_exists(str(args['model'])):
        print(f'[ERROR] Choose an appropriate model to continue, must be one out of: {func_names}.')
        print(f'[ERROR] Or choose a custom model lying in folder models.')
        exit(1)

check_and_create_folder('models')
modelFolderPath = path.join('models', args['model'])
modelDataPath = path.join(modelFolderPath, 'data.csv')
modelLogPath = path.join(modelFolderPath, 'training.log')
modelCheckpointsPath = path.join(modelFolderPath, 'checkpoints')
if not check_and_create_folder(modelFolderPath):
    modelExists = False
    check_and_create_folder(modelCheckpointsPath)
else:
    modelExists = path.exists(modelDataPath)

# %% import rest of dependencies and configure logging

import tensorflow as tf
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

# from utils.evaluation import evaluate_confusion_matrix, log_confusion_matrix
from utils.data import *

# set GPU configs, might have to comment out
# these lines if you're working on a cpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler(modelLogPath),
        log.StreamHandler(stdout)
    ]
)

printSeparator()
log.info(f'Starting Training with arguments: {args}')

# %% load model

# initialize some training parameters
INIT_LR = 1e-3
BS = 8

if modelExists:
    model, modelData, trainEpochs, trainedEpochs = load_existing_model(args['model'], modelDataPath, args['epochs'])
else:
    log.info('Model does not exist yet, creating a new one')
    baseModel = modelFunc(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    log.info(f'baseModel: {args["model"]}')      # ehemals "baseModel", aber es soll doch der Name angezeigt werden oder?
    # construct head of model that will be placed on top of the base model
    headModel = baseModel.output
    # headModel = GaussianNoise(stddev=1.0)(headModel)
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name='flatten')(headModel)
    headModel = Dense(64, activation='relu')(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation='softmax')(headModel)

    for layer in baseModel.layers:
        layer.trainable = False
    model = Model(inputs=baseModel.input, outputs=headModel)
    trainEpochs = args['epochs']
    trainedEpochs = 0
    log.info(f'trainEpochs: {trainEpochs}')

check_if_trained_or_exit(trainEpochs, args['epochs'])

# %% prepare data

trainX, valX, trainY, valY, testData, testLabels = load_dataset(args['dataset'])

# %% train model

# initialize the training data augmentation object
# trainAug = ImageDataGenerator(rotation_range=10,      # adapt rotation range? wie schief ist so ein röntgenbild wohl maxial aufgenommen worden
#                               horizontal_flip=True,   # cc: die schieferen scheinen so maximal 20, die allermeisten aber <5; 10 ist denke ich guter Mittelweg
#                               fill_mode='nearest',    # Testweise bessere performance als constant fill
#                               width_shift_range=0.1,  # Horizontal sind die Bilder größtenteils gut zentriert
#                               height_shift_range=0.2) # Vertikal tendenziell etwas schlechter
#                               #zoom_range=0.2)         # 1.0 +- zoom_range, kann also raus oder reinzoomen
trainAug = ImageDataGenerator() # TODO: enable for benchmark training
opt = Adam(lr=INIT_LR, decay=INIT_LR / trainEpochs)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# train network head, H not needed for now
# holds useful information about training progress
try:
    model.fit(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(valX, valY),
        validation_steps=len(valX) // BS,
        epochs=trainEpochs,
        callbacks=[ModelCheckpoint(
            filepath=f'models/{args["model"]}/checkpoints/checkpoint_epoch{trainedEpochs}' + '+{epoch}' + '_ckpt-loss={loss:.2f}.h5',
            monitor='val_loss',
            save_weights_only=True,
            save_best_only=True,
            period=3,
        ), EvaluationCallback( # TODO: specify how often inbetween model should be saved!
            test_data=testData,
            test_labels=testLabels,
            batch_size=BS,
            model_name=args['model'],
            trained_epochs=trainedEpochs,
        )]
    )
except KeyboardInterrupt:
    log.info(f'interrupted Training at epoch: {len(model.history.epoch)+1}')

# save history and epochs for later
# as history gets deleted when model.predict() is called
history = model.history.history.copy()
epochs = len(model.history.epoch)

# %% validate model

predictions = np.argmax(model.predict(testData, batch_size=BS), axis=1)
report = classification_report(testLabels.argmax(axis=1), predictions, target_names=CLASSES) # TODO: do we even need this?
log.info(f'Evaluating trained network\n{report}')

cm = confusion_matrix(testLabels.argmax(axis=1), predictions)
# evaluation = evaluate_confusion_matrix(cm)
# log_confusion_matrix(cm)
# for metric in evaluation: log.info(metric + ': {:.4f}'.format(evaluation[metric]))

# %% serialize model and csv

if modelExists:
    epochs += trainedEpochs
    log.info(f'epoch: {epochs}')
    modelData = modelData.append(pd.DataFrame(history))
    modelData.index = np.arange(0, len(modelData))
else:
    log.info(f'epoch: {epochs}')
    modelData = pd.DataFrame(history)

persist_results(model, modelData, modelFolderPath, epochs)
printSeparator(with_break=True)
