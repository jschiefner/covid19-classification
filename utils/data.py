import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB
from os import path
from utils.constants import CLASSES, IMG_DIMENSIONS
import logging as log

def load_dataset(datasetPath):
    log.info(f'loading data from "{datasetPath}"')

    metadata = pd.read_csv(path.join(datasetPath, 'metadata.csv'), usecols=['File', 'Covid', 'No Finding'],
                           dtype={'File': np.str, 'Covid': np.bool, 'No Finding': np.bool})
    # metadata = metadata[1000:1100] # for now only use 100 samples (10 positive, 90 negative) # TODO: comment out before comitting

    data = []
    labels = []

    for _idx, (file, covid, noFinding) in metadata.iterrows():
        if covid: label = CLASSES[0] # covid
        elif noFinding: label = CLASSES[1] # healthy
        else: label = CLASSES[2] # other
        image = imread(path.join(datasetPath, 'images', file))
        image = cvtColor(image, COLOR_BGR2RGB)
        image = resize(image, IMG_DIMENSIONS)

        data.append(image)
        labels.append(label)

    data = np.array(data) / 255.0
    labels = np.array(labels)
    dataLength = len(data)
    log.info(f'successfully loaded {dataLength} images')

    log.info('encode labels')

    # one hot encoding labels
    lb = LabelBinarizer().fit(CLASSES)
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
    (trainX, valX, trainY, valY) = train_test_split(trainValidationData, trainValidationLabels, test_size=0.2,
                                                    stratify=trainValidationLabels)  # random_state=42
    log.info(f'selected {len(trainX)} images for training and {len(valX)} images for validation (during training)')
    return trainX, valX, trainY, valY, testData, testLabels
