# %% imports

import tensorflow as tf
from os import path
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

print('[INFO] set GPU configs')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# "parse" arguments (for now just specifying them like this
args = {'model': 'VGG16'} # default model

# initialize some training parameters
INIT_LR = 1e-3
EPOCHS = 25
BS = 8

# roadmap:
# prepare images
# split data in training/validation data

# if exists selected model?
    # load in model + according csv
    # keep training until epoch 25
    # finish when reaching epoch 25
# if not exists selected model?
    # create model
    # train
    # save model.h5 + model.csv


# %% load model

models = {
    'VGG16': VGG16
}

if not args['model'] in models:
    print(f'[INFO] Choose an appropriate model to continue, must be one out of: {models.keys()}')
    exit(1)

modelDataPath = f'models/{args["model"]}.csv'
modelExists = path.exists(modelDataPath)
if modelExists:
    print(f'[INFO] Model exists!')
    modelData = pd.read_csv(modelDataPath, index_col=0)
    trainedEpochs = len(modelData)
    print(f'[INFO] trainedEpochs: {trainedEpochs}')
    trainEpochs = EPOCHS - trainedEpochs
    print(f'[INFO] trainEpochs: {trainEpochs}')
    modelPaths = glob(f'models/{args["model"]}_*.h5')
    print(f'[INFO] modelPaths: {modelPaths}')
    model = load_model(modelPaths[-1]) # load latest model
else:
    print(f'[INFO] Model does not exist yet, creating a new one')
    # baseModel = models[args['model']](weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    print(f'[INFO] baseModel: {baseModel}')
    # construct head of model that will be placed on top of the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name='flatten')(headModel)
    headModel = Dense(64, activation='relu')(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation='softmax')(headModel)

    for layer in baseModel.layers:
        layer.trainable = False
    model = Model(inputs=baseModel.input, outputs=headModel)
    trainEpochs = EPOCHS
    print(f'[INFO] trainEpochs: {trainEpochs}')

if (trainEpochs < 0):
    print(f'[INFO] network is already trained on {EPOCHS} epochs, exiting')
    exit(0)


# %% prepare data
print('[INFO] prepare data')

metadata = pd.read_csv('metadata.csv', usecols=['File', 'Covid'], dtype={'File': np.str, 'Covid': np.bool})
metadata = metadata[100:200] # for now only use 100 samples (5 positive, 95 negative)

covid_list = metadata[metadata['Covid'] == True]
healthy_list = metadata[metadata['Covid'] == False]

data = []
labels = []

for idx, (file, covid) in metadata.iterrows():
    label = 'covid' if covid else 'normal'
    image = imread(f'images/{file}')
    image = cvtColor(image, COLOR_BGR2RGB)
    image = resize(image, (224, 224))

    data.append(image)
    labels.append(label)

data = np.array(data) / 255.0
labels = np.array(labels)

print('[INFO] encode labels')

# one hot encoding labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# datasplit
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# %% train model

# initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=15, fill_mode='nearest')
opt = Adam(lr=INIT_LR, decay=INIT_LR / trainEpochs)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# train network head, H not needed for now
# holds useful information about training progress
try:
    model.fit(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=trainEpochs,
    )
except KeyboardInterrupt:
    print(f'\n[INFO] interrupted Training at epoch: {len(model.history.epoch)}')

# %% serialize model and csv

if modelExists:
    epoch = len(model.history.epoch) + trainedEpochs
    print(f'[INFO] epoch: {epoch}')
    modelData = modelData.append(pd.DataFrame(model.history.history))
else:
    epoch = len(model.history.epoch)
    print(f'[INFO] epoch: {epoch}')
    modelData = pd.DataFrame(model.history.history)
modelPath = f'models/{args["model"]}_{epoch}.h5'
csvPath = f'models/{args["model"]}.csv'
print(f'[INFO] saving model to: {modelPath}, saving csv to: {csvPath}')
model.save(modelPath, save_format='h5')
modelData.to_csv(csvPath)