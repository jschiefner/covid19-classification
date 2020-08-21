# %% parse arguments

from argparse import ArgumentParser
from os import path

parser = ArgumentParser()
parser.add_argument('dataset', help='file to input folder')
parser.add_argument("-m","--model", default="VGG16", help="specify optional network")
args = vars(parser.parse_args())

# metadata check
if not path.exists(args['dataset']):
    print(f'[ERROR] the path "{args["dataset"]}" does not exist. Please supply a valid input folder.')
    exit(1)

# model check
from tensorflow.keras.applications import *
MODELS = [
    Xception,
    VGG16,
    VGG19,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet50V2,
    ResNet101V2,
    ResNet152V2,
    InceptionV3,
    InceptionResNetV2,
    MobileNet,
    MobileNetV2,
    DenseNet121,
    DenseNet169,
    DenseNet201,
    NASNetMobile,
    NASNetLarge,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
]
func_names = [m.__name__ for m in MODELS]
funcs_dict = dict(zip(func_names, MODELS))
if not str(args['model']) in funcs_dict:
    print(f'[ERROR] Choose an appropriate model to continue, must be one out of: {func_names}.')
    exit(1)
modelFunc = funcs_dict[str(args['model'])]

modelDataPath = f'models/{args["model"]}.csv'
modelExists = path.exists(modelDataPath)

# %% import rest of dependencies

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout, GaussianNoise
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB
from glob import glob
import pandas as pd
import numpy as np
print('[INFO] set GPU configs')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# %% load model

# initialize some training parameters
INIT_LR = 1e-3
EPOCHS = 25
BS = 8

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
    baseModel = modelFunc(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    print(f'[INFO] baseModel: {baseModel}')
    # construct head of model that will be placed on top of the base model
    headModel = baseModel.output
    # headModel = GaussianNoise(stddev=1.0)(headModel)
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

if trainEpochs <= 0:
    print(f'[INFO] network is already trained on {EPOCHS} epochs, exiting')
    exit(0)

# %% prepare data
print('[INFO] prepare data')

metadata = pd.read_csv(path.join(args['dataset'], "metadata.csv"), usecols=['File', 'Covid'], dtype={'File': np.str, 'Covid': np.bool})
metadata = metadata[1000:1100] # for now only use 100 samples (10 positive, 90 negative)

covid = metadata[metadata['Covid'] == True]
healthy = metadata[metadata['Covid'] == False]

data = []
labels = []

for idx, (file, covid) in metadata.iterrows():
    label = 'covid' if covid else 'normal'
    image = imread(path.join(args['dataset'], "images", file))
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
# random_stateint Controls the shuffling applied to the data before applying the split.
# pass an int for reproducible output across multiple function calls. See Glossary.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels ) # random_state=42

# %% train model

# initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=10,      # adapt rotation range? wie schief ist so ein röntgenbild wohl maxial aufgenommen worden
                              horizontal_flip=True,   # cc: die schieferen scheinen so maximal 20, die allermeisten aber <5; 10 ist denke ich guter Mittelweg
                              fill_mode='nearest',    # Testweise bessere performance als constant fill
                              width_shift_range=0.1,  # Horizontal sind die Bilder größtenteils gut zentriert
                              height_shift_range=0.2) # Vertikal tendenziell etwas schlechter
                              #zoom_range=0.2)         # 1.0 +- zoom_range, kann also raus oder reinzoomen
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
