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
from utils.management import find_and_load_models
from utils.constants import CLASSES

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
parser.add_argument('-m', '--model', required=False, default='all', help='models to be used (uses all it can find by default), possible to specify multiple: -m "VGG16 MobileNetV22"')
args = vars(parser.parse_args())
# args = {'dataset': 'images', 'output': 'output.csv', 'model': 'VGG16 ResNet50'} # for development

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
if args['model']=="all":
    models = find_and_load_models()
else:
    models = args['model'].split(" ")

# verify model paths
modelDataPaths = []
for model in models:
    _path = path.join('models', model, 'data.csv')
    if not path.exists(_path):
        log.error(f'the model "{model}" has not been trained yet.')
        log.error('Please train the model first before predicting with it.')
        exit(1)
    else:
        modelDataPaths.append(_path)

number_of_models = len(models)


# %% load images
# evtl abfangen falls nicht bilddateien in diesem pfad liegen?
data = []
imageList = glob(path.join(args['dataset'], '*'))
#imageList = imageList[0:500] # TODO: remove this, just for development purposes
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
    log.info(f'predicting...')
    predictionsList.append(model.predict(data))
    log.info(f'done')

log.info('predictions have been calculated')

# save prediction to output file
fileNames = [path.basename(file) for file in imageList]
outputDict = {'File': fileNames}




# diese ausgabe ist wenig nützlich
for predictions, model in zip(predictionsList, models):
    outputDict[f'argmax [{model}]'] = [CLASSES[np.argmax(prediction)] for prediction in predictions]


# TODO ensemble classikator

# ensemble stuff here
#if number_of_models > 1:
#    temp = np.zeros(len(fileNames))
#    for predictions, model in zip(predictionsList, models):
#        temp += [(1 if np.argmax(prediction) == 0 else 0) for prediction in predictions] # 100% covid == [1,0,0]
#    outputDict['Covid [Ensemble Majority]'] = [t*2 >= number_of_models for t in temp]

# ensemble stuff here
if number_of_models > 1:
    n = len(predictionsList[0])
    ensemble = []
    for i in range(n):
        count = [0,0,0]
        for j in range(number_of_models):
            count[np.argmax(predictionsList[j][i])]+=1

        if count[0]>count[1] and count[0]>count[2]:
            ensemble.append(CLASSES[0])
        elif count[1] > count[0] and count[1] > count[2]:
            ensemble.append(CLASSES[1])
        elif count[2] > count[0] and count[2] > count[1]:
            ensemble.append(CLASSES[2])
        else:
            ensemble.append("")
    outputDict['Covid [Ensemble Majority]'] = ensemble
    #print(outputDict['Covid [Ensemble Majority]'])
    ensemble = []
    for i in range(n):
        s = 0.0
        for j in range(number_of_models):
            s+=predictionsList[j][i][0]
        s/=number_of_models
        ensemble.append(s)
    outputDict['Covid Ensemble'] = ensemble

    #print(outputDict['Covid Ensemble'])





for predictions, model in zip(predictionsList, models):
    outputDict[f'Covid (probability)[{model}]'] = predictions[:, 0]
    outputDict[f'Healthy (probability)[{model}]'] = predictions[:, 1]
    outputDict[f'Other (probability)[{model}]'] = predictions[:, 2]

df = pd.DataFrame(outputDict)
df.set_index('File')
try:
    df.to_csv(f'{args["output"]}', index=False)
    log.info(f'Predictions have been saved to "{args["output"]}"')
except PermissionError as e:
    log.error('Permission Error while saving file')
    log.error(f'{e}')
