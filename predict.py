# %% imports

from os import path, environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # make tensorflow less verbose
from argparse import ArgumentParser
from glob import glob
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB
import logging as log
from sys import stdout
import tensorflow as tf

# set GPU configs, might have to comment out
# these lines if you're working on a cpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[log.StreamHandler(stdout)]
)

# %% parse arguments

parser = ArgumentParser()
parser.add_argument('dataset', help='folder containing images to predict')
parser.add_argument('output', help='filepath where output file should be stored')
parser.add_argument('-m', '--model', required=False, default='VGG16', help='models to be used (VGG16 by default), possible to specify multiple: -m "VGG16 MobileNetV2"')
args = vars(parser.parse_args())
# args = {'dataset': 'images', 'output': 'output.csv', 'model': 'VGG16'} # for development

# verify dataset path
if not path.isdir(args['dataset']):
    log.error(f'the dataset path "{args["dataset"]}" is not valid.')
    log.error('Please supply the path to dataset folder')
    exit(1)

# verify/correct output path
outpath, ext = path.splitext(args['output'])
if len(ext) == 0:
    args['output'] = f'{outpath}.csv'

# check if multiple models given, its ensemble classification then
models = args['model'].split(" ")
number_of_models = len(models)

# verify model paths
modelDataPaths = []
for model in models:
    modelDataPaths.append(path.join('models', model, 'data.csv'))
    if not path.exists(modelDataPaths[-1]):
        log.error(f'the model "{model}" has not been trained yet.')
        log.error('Please train the model first before predicting with it.')
        exit(1)

# %% load images
# evtl abfangen falls nicht bilddateien in diesem pfad liegen?
data = []
imageList = glob(path.join(args['dataset'], '*'))
imageList = imageList[0:100] # TODO: remove this, just for development purposes
for file in imageList:
    image = imread(file)
    image = cvtColor(image, COLOR_BGR2RGB)
    image = resize(image, (224, 224))
    data.append(image)
log.info(f'successfully loaded {len(data)} images from "{args["dataset"]}" directory.')
data = np.array(data) / 255.0

# %% load models
predictionsList = []
for modelDataPath, model in zip(modelDataPaths, models):
    modelData = pd.read_csv(modelDataPath, index_col=0)
    epochs = len(modelData)
    log.info(f'model "{model}" was trained for {epochs} epochs')
    modelPath = path.join('models', model, f'epoch_{epochs}.h5')
    model = load_model(modelPath)
    log.info(f'successfully loaded "{modelPath}"')
    predictionsList.append(model.predict(data))

log.info('predictions have been calculated')

# save prediction to output file
fileNames = [path.basename(file) for file in imageList]
outputDict = {'File': fileNames}

for predictions, model in zip(predictionsList, models):
    outputDict.update({f'Covid [{model}]': [np.argmax(prediction) == 0 for prediction in predictions]})

# ensemble stuff here
if number_of_models > 1:
    temp = np.zeros(len(fileNames))
    for predictions, model in zip(predictionsList, models):
        temp += [(1 if np.argmax(prediction) == 0 else 0) for prediction in predictions]
    outputDict.update({f'Covid [Ensemble Majority]': ["True" if t*2 >= number_of_models else "False" for t in temp]})

for predictions, model in zip(predictionsList, models):
    outputDict.update({f'Covid (probability)[{model}]': predictions[:, 0]})

for predictions, model in zip(predictionsList, models):
    outputDict.update({f'No Finding (probability)[{model}]': predictions[:, 1]})

df = pd.DataFrame(outputDict)
df.set_index('File')
try:
    df.to_csv(f'{args["output"]}', index=False)
    log.info(f'Predictions have been saved to "{args["output"]}"')
except PermissionError as e:
    log.error('Error while saving file')
    log.error(f'{e}')
