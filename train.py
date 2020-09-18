# %% parse arguments

from argparse import ArgumentParser
from os import path, mkdir, environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # make tensorflow less verbose

parser = ArgumentParser()
parser.add_argument('dataset', help='path to input folder')
parser.add_argument("-m", "--model", default="VGG16", help="specify optional network")
parser.add_argument('-e', '--epochs', default=30, type=int, help='specify how many epochs the network should be trained at max, defaults to 30')
args = vars(parser.parse_args())
# args = {'dataset': '.', 'model': 'VGG16', 'epochs': 25} # TODO: comment out

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

if not path.isdir('models'):
    mkdir('models') # create models directory if it doesnt exist yet
modelFolderPath = path.join('models', args['model'])
modelDataPath = path.join(modelFolderPath, 'data.csv')
modelLogPath = path.join(modelFolderPath, 'training.log')
modelCheckpointsPath = path.join(modelFolderPath, 'checkpoints')
if not path.isdir(modelFolderPath):
    modelExists = False
    mkdir(modelFolderPath) # create model specific directory
    mkdir(modelCheckpointsPath)
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

log.info('======================================================================')
log.info(f'Starting Training with arguments: {args}')

# %% load model

# initialize some training parameters
INIT_LR = 1e-3
BS = 8

if modelExists:
    log.info('Model exists!')
    modelData = pd.read_csv(modelDataPath, index_col=0)
    trainedEpochs = len(modelData)
    trainEpochs = args['epochs'] - trainedEpochs
    log.info(f'trained epochs: {trainedEpochs}, train epochs: {trainEpochs}, (total: {trainEpochs + trainedEpochs})')
    modelPaths = glob(f'models/{args["model"]}/epoch_*.h5')
    log.info(f'modelPaths: {modelPaths}')
    latestModelPath = modelPaths[-1]
    model = load_model(latestModelPath) # load latest model
    log.info(f'successfully loaded "{latestModelPath}"')
else:
    log.info('Model does not exist yet, creating a new one')
    baseModel = modelFunc(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    log.info(f'baseModel: {baseModel}')
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

if trainEpochs <= 0:
    log.info(f'network is already trained on {args["epochs"]} epochs, exiting')
    log.info('if you want to train the network for longer, set the optional --epochs argument')
    exit(0)

# %% prepare data
log.info('prepare data')

metadata = pd.read_csv(path.join(args['dataset'], "metadata.csv"), usecols=['File', 'Covid', 'No Finding'], dtype={'File': np.str, 'Covid': np.bool, 'No Finding': np.bool})
# metadata = metadata[1000:1100] # for now only use 100 samples (10 positive, 90 negative) # TODO: comment out before comitting

data = []
labels = []

for _idx, (file, covid, noFinding) in metadata.iterrows():
    if covid: label = 'covid'
    elif noFinding: label = 'healthy'
    else: label = 'other'
    image = imread(path.join(args['dataset'], "images", file))
    image = cvtColor(image, COLOR_BGR2RGB)
    image = resize(image, (224, 224))

    data.append(image)
    labels.append(label)

data = np.array(data) / 255.0
labels = np.array(labels)
dataLength = len(data)
log.info(f'successfully loaded {dataLength} images')

log.info('encode labels')

# one hot encoding labels
lb = LabelBinarizer().fit(['covid', 'healthy', 'other'])
labels = lb.transform(labels)
# labels = to_categorical(labels) # TODO: is this really not necessary?

# datasplit
log.info('splitting data')
# use last 1/3rd of data for validation

third = (dataLength // 3) * 2
trainValidationData = data[:third]
trainValidationLabels = labels[:third]
testData = data[third:]
testLabels = labels[third:]

log.info(f'selected {len(testData)} images for validation after training')

# random_stateint Controls the shuffling applied to the data before applying the split.
# pass an int for reproducible output across multiple function calls. See Glossary.
(trainX, valX, trainY, valY) = train_test_split(trainValidationData, trainValidationLabels, test_size=0.2, stratify=trainValidationLabels) # random_state=42
log.info(f'selected {len(trainX)} images for training and {len(valX)} images for validation (during training)')

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
    log.info(f'interrupted Training at epoch: {len(model.history.epoch)}')

# save history and epochs for later
# as history gets deleted when model.predict() is called
history = model.history.history.copy()
epochs = len(model.history.epoch)

# %% validate model

predictions = np.argmax(model.predict(testData, batch_size=BS), axis=1)
report = classification_report(testLabels.argmax(axis=1), predictions, target_names=lb.classes_) # TODO: do we even need this?
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

modelPath = path.join('models', args['model'], f'epoch_{epochs}.h5')
csvPath = path.join('models', args['model'], 'data.csv')
log.info(f'saving model to: "{modelPath}", saving csv to: "{csvPath}"')
model.save(modelPath, save_format='h5')
modelData.to_csv(csvPath)
log.info('======================================================================\n\n')
