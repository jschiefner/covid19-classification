# %% parse arguments

from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # make tensorflow less verbose

from tensorflow.keras import models, layers
from utils.management import *
from utils.constants import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-m', '--model', required=True, help='how your custom model should be called')
args = vars(parser.parse_args())

modelFolderPath = path.join('models', args['model'])
modelDataPath = path.join(modelFolderPath, 'data.csv')
if check_and_create_folder(modelFolderPath):
    print(f'The model "{args["model"]}" already exists. Please choose another name.')
    exit(1)

check_and_create_folder('models') # make sure model directory exists
printSeparator()

# %% create model

# build model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_DIMENSIONS_3D))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.GaussianNoise(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation="softmax"))

model.summary() # prints model summary

# %% save model

modelPath = path.join(modelFolderPath, f'epoch_none.h5')
csvPath = path.join(modelFolderPath, 'data.csv')
print(f'saving model to: "{modelPath}", saving csv to: "{csvPath}"')
model.save(modelPath, save_format='h5')
df = pd.DataFrame({'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []})
df.to_csv(csvPath)
printSeparator()
