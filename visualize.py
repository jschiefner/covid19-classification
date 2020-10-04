# %% parse arguments

from argparse import ArgumentParser
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = ArgumentParser()
parser.add_argument('-m', '--model', required=True, help='Model Name')
args = vars(parser.parse_args())
modelDataPath = path.join('models', args['model'], 'data.csv')

if not path.exists(modelDataPath):
    print(
        f'[ERROR] the model "{args["model"]} has not been trained yet.',
        f'Please train the model first and re-run this file.',
    )
    exit(1)

# %% run

modelData = pd.read_csv(modelDataPath, index_col=0)
epochs = np.arange(0, len(modelData))
plt.style.use('ggplot')
plt.title(f'Training Loss/Accuracy for {args["model"]} model')
plt.xlabel('# Epochs')
plt.ylabel('Loss/Accuracy')
plt.ylim(0, 1)

plt.plot(epochs, modelData.loss, label='loss')
plt.plot(epochs, modelData.accuracy, label='accuracy')
plt.plot(epochs, modelData.val_loss, label='val_loss')
plt.plot(epochs, modelData.val_accuracy, label='val_accuracy')

plt.legend(loc='center right')
plt.show()
