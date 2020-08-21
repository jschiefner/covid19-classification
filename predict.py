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
parser.add_argument('-m', '--model', required=False, default='VGG16', help='model to be used (VGG16 by default)')
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

# verify model path
modelDataPath = f'models/{args["model"]}.csv'
if not path.exists(modelDataPath):
    print(
        f'[ERROR] the model "{args["model"]}" has not been trained yet.',
        'Please train the model first before predicting with it.',
    )
    exit(1)

# %% load model

modelData = pd.read_csv(modelDataPath, index_col=0)
epochs = len(modelData)
print(f'[INFO] model "{args["model"]}" was trained for {epochs} epochs')
modelPath = f'models/{args["model"]}_{epochs}.h5'
model = load_model(modelPath)
print(f'[INFO] successfully loaded "{modelPath}"')

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

# %% make prediction

predictions = model.predict(data)
fileNames = [path.basename(file) for file in imageList]
outputDict = {
    'File': fileNames,
    'Covid': [np.argmax(prediction) == 0 for prediction in predictions],
    'Covid (probability)': predictions[:,0],
    'No Finding (probabilty)': predictions[:,1],
}
df = pd.DataFrame(outputDict)
df.set_index('File')
df.to_csv(args['output'], index=False)