import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB
from os import path
from utils.constants import CLASSES, IMG_DIMENSIONS
import logging as log
from progress.bar import Bar

def load_dataset(datasetPath, validation_after_train_split=0.33):
    log.info(f'loading data from "{datasetPath}"')

    metadata = pd.read_csv(path.join(datasetPath, 'metadata.csv'), usecols=['File', 'No Finding', 'Covid'],
                           dtype={'File': np.str, 'No Finding': np.bool, 'Covid': np.bool})
    # metadata = metadata[0:200] # for now only use 100 samples (10 positive, 90 negative) # TODO: comment out before comitting
    data = []
    labels = []
    with Bar('Loading images', max=len(metadata)) as bar:
        for _idx, (file, noFinding, covid) in metadata.iterrows():
            bar.next()
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
    lb = LabelBinarizer()
    lb.fit(CLASSES) # !!! changes order of classes: covid, other, healthy
    labels = lb.transform(labels)

    log.info(f'overall label prevalence count: {prevalence_count(labels)}')

    # datasplit
    log.info('splitting data')
    # first chunk for train data
    splitPoint = dataLength-int(dataLength*validation_after_train_split)
    trainValidationData = data[:splitPoint]
    trainValidationLabels = labels[:splitPoint]
    # second chunk for the testing after training with fresh images
    testData = data[splitPoint:]
    testLabels = labels[splitPoint:]

    log.info(f'train/validation label prevalence count: {prevalence_count(trainValidationLabels)}')
    log.info(f'test label prevalence count: {prevalence_count(testLabels)}')

    log.info(f'selected {len(trainValidationData)} images for training/validation during training')
    log.info(f'selected {len(testData)} images for testing after each epoch and after training')

    # random_stateint Controls the shuffling applied to the data before applying the split.
    # pass an int for reproducible output across multiple function calls. See Glossary.
    (trainX, valX, trainY, valY) = train_test_split(trainValidationData, trainValidationLabels, test_size=0.2,
                                                    stratify=trainValidationLabels)  # random_state=42
    log.info(f'selected {len(trainX)} images for training and {len(valX)} images for validation (during training)')
    return trainX, valX, trainY, valY, testData, testLabels

def prevalence_count(labels):
    count = [0 for _ in range(len(CLASSES))]
    for lab in labels:
        count[np.argmax(lab)] += 1
    return f"{count} with {class_specifier()}"

def class_specifier():
    lb = LabelBinarizer()
    lb.fit(CLASSES)
    a = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    b = lb.inverse_transform(a)
    print(b)
    return b
