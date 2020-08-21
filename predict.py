# %% imports

from os import path
from argparse import ArgumentParser
from glob import glob
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB

# %% parse arguments

parser = ArgumentParser()
parser.add_argument('dataset', help='folder containing images to predict')
parser.add_argument('output', help='filepath where output file should be stored')
parser.add_argument('-m', '--model', required=False, default='VGG16', help='models to be used (VGG16 by default), possible to specify multiple: -m "VGG16 MobileNetV2"')
args = vars(parser.parse_args())
# args = {'dataset': 'images', 'output': 'output.csv', 'model': 'VGG16'} # for development

# verify dataset path
if not path.exists(args['dataset']):
    print(
        f'[ERROR] the dataset path "{args["dataset"]}" is not valid.',
        'Please supply the path to dataset folder',
    )
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
    modelDataPaths.append(f'models/{model}.csv')
    if not path.exists(modelDataPaths[-1]):
        print(
            f'[ERROR] the model "{model}" has not been trained yet.',
            'Please train the model first before predicting with it.',
        )
        exit(1)

# %% load images

data = []
imageList = glob(path.join(args['dataset'], '*'))
imageList = imageList[0:10] # TODO: remove this, just for development purposes
for file in imageList:
    image = imread(file)
    image = cvtColor(image, COLOR_BGR2RGB)
    image = resize(image, (224, 224))
    data.append(image)
print(f'[INFO] successfully loaded {len(data)} images from "{args["dataset"]}" directory.')
data = np.array(data) / 255.0

# %% load models
predictionsList = []
for modelDataPath, model in zip(modelDataPaths,models):
    modelData = pd.read_csv(modelDataPath, index_col=0)
    epochs = len(modelData)
    print(f'[INFO] model "{model}" was trained for {epochs} epochs')
    modelPath = f'models/{model}_{epochs}.h5'
    model = load_model(modelPath)
    print(f'[INFO] successfully loaded "{modelPath}"')

    # %% make prediction
    predictionsList.append(model.predict(data))

print(f'[INFO] predictions have been calculated')

# save prediction to output file
fileNames = [path.basename(file) for file in imageList]
outputDict = {
        'File': fileNames
    }

for predictions, model in zip(predictionsList,models):
    outputDict.update({f'Covid [{model}]': [np.argmax(prediction) == 0 for prediction in predictions]})

#ensemble stuff here
if number_of_models>1:
    temp = np.zeros(len(fileNames))
    for predictions, model in zip(predictionsList,models):
        temp += [np.argmax(prediction) for prediction in predictions]
    temp/=number_of_models
    outputDict.update({f'Covid [Ensemble Majority]': temp})

for predictions, model in zip(predictionsList,models):
    outputDict.update({f'Covid (probability)[{model}]': predictions[:, 0]})

for predictions, model in zip(predictionsList,models):
    outputDict.update({f'No Finding (probability)[{model}]': predictions[:, 1]})

df = pd.DataFrame(outputDict)
df.set_index('File')
try:
    df.to_csv(f'{args["output"]}', index=False)
    print(f'[INFO] Predictions have been saved to {args["output"]}')
except PermissionError as e:
    print(f'[ERROR] Error while saving file')
    print(f'{e}')
