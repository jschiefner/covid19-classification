# %% parse arguments

from argparse import ArgumentParser
from os import path, mkdir, environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # make tensorflow less verbose
from utils.management import *
from utils.constants import *
from utils.base_models import *

# example: python train.py dataset -m "MobileNetV2" -e 10 --evaluate 1 -v
parser = ArgumentParser()
parser.add_argument('dataset', help='path to input folder')
parser.add_argument("-m", "--model", default="VGG16", help=f"specify optional network. ")
parser.add_argument('-e', '--epochs', default=30, type=int, help='specify how many epochs the network should be trained at max, defaults to 30')
parser.add_argument('-v','--visualize', action='store_true', help='set to run tf-explain as callback')
parser.add_argument('--evaluate', default=0,type=int, help='set to evaluate the network after every x epochs')
parser.add_argument('-s','--save',default=0, type=int, help='unreliable computer, save current state after every x epochs')
parser.add_argument('-a','--autostop', action='store_true', help='set to stop training when val_loss rises')
args = vars(parser.parse_args())

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
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout, GaussianNoise, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model, Sequential

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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
    baseModel = modelFunc(weights='imagenet', include_top=False, input_shape=IMG_DIMENSIONS_3D,input_tensor=Input(shape=IMG_DIMENSIONS_3D))
    baseModel.trainable = False
    #baseModel.summary()

    log.info(f'baseModel: {args["model"]}')
    # construct head of model that will be placed on top of the base model
    # denke unser headmodel ist zu klein für 3 klassen, sollten hier noch ein wenig herumprobieren

    # # vgg16 top [bad af]
    # x = Flatten(name='flatten')(baseModel.output)
    # x = Dense(128, activation='relu', name='fc1')(x)
    # x = Dense(4*3, activation='relu', name='fc2')(x)
    # x = Dense(3, activation='softmax', name='predictions')(x)

    # # vgg16 top slighty changed
    ## x = Conv2D(128, 3, padding='same',name='conv_gradcam', activation='relu')(baseModel.output)# test
    x = GlobalAveragePooling2D(name='global_avg_pool2d')(baseModel.output)
    #x = Dropout(0.3)(x)
    # x = Flatten(name='flatten')(baseModel.output)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(3, activation='softmax', name='predictions')(x)

    # #alternative 2
    # #x= Conv2D(128, 3, padding='same', activation='relu')(baseModel.output)
    # x = GlobalAveragePooling2D()(baseModel.output)
    # x = Dropout(0.3)(x)
    # #x = AveragePooling2D(pool_size=(4,4))(x)
    # #x = Dense(64, activation='relu')(x)
    # #x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(3,activation='softmax')(x)

    model = Model(inputs=baseModel.input, outputs=x)
    '''
    model = Sequential([
        baseModel,
        AveragePooling2D(name='pool_head', pool_size=(4, 4)),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        #Conv2D(128, 3, padding='same', activation='relu'),
        #MaxPooling2D(pool_size=(3,3)),
        Flatten(name='flatten1_head'),
        Dense(128, name='dense_relu_head', activation='relu'),
        Dense(3, name='output_softmax_head',activation='softmax')
    ])
    '''

    trainEpochs = args['epochs']
    trainedEpochs = 0
    log.info(f'trainEpochs: {trainEpochs}')

log.info('Model Overview:')
model.summary()

check_if_trained_or_exit(trainEpochs, args['epochs'])

# %% prepare data

trainX, valX, trainY, valY, testData, testLabels = load_dataset(args['dataset'], validation_after_train_split=0.1)

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

# setup callbacks

from utils.my_gradcam_callback import MyGradCAMCallback

callback_gradcam = MyGradCAMCallback(  # nur bilder mit bestätigtem corona nehmen?
        validation_data=(valX, valY),
        class_index=0,
        output_dir=path.join(modelFolderPath,"visualized"),
        limit=25,
        )
callback_modelcheckpoint = ModelCheckpoint(
        filepath=f'models/{args["model"]}/checkpoints/checkpoint_epoch{trainedEpochs}' + '+{epoch}' + '_ckpt-loss={loss:.2f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        period=args['save'],
    )
callback_evaluation = EvaluationCallback( # TODO: specify how often inbetween model should be saved!
        test_data=testData,
        test_labels=testLabels,
        batch_size=BS,
        model_name=args['model'],
        trained_epochs=trainedEpochs,
        freq=args['evaluate']
    )
callback_earlyStopping = EarlyStopping(monitor='val_loss', patience=1)

callbacks = []
if args['visualize']: callbacks.append(callback_gradcam)
if args['evaluate']>0: callbacks.append(callback_evaluation)
if args['save']>0: callbacks.append(callback_modelcheckpoint)
if args['autostop']: callbacks.append(callback_earlyStopping)

# train network head, H not needed for now
# holds useful information about training progress
try:
    model.fit(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(valX, valY),
        validation_steps=len(valX) // BS,
        epochs=trainEpochs,
        callbacks=callbacks,
    )
except KeyboardInterrupt:
    print()
    log.info(f'interrupted Training at epoch: {len(model.history.epoch)+1}')
    if len(model.history.epoch)==0:
        exit(0)

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
