# %% imports
print('import lots of stuff')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, NASNetLarge, MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# %% arg "parse"

args = {'plot': 'out/plot.png', 'model': 'out/model.h5'}

# %% prepare data
print('prepare data')

metadata = pd.read_csv('metadata.csv', usecols=['File', 'Covid'], dtype={'File': np.str, 'Covid': np.bool})
# metadata = metadata[100:200] # for now only use 100 samples (5 positive, 95 negative)

covid_list = metadata[metadata['Covid'] == True]
healthy_list = metadata[metadata['Covid'] == False]

data = []
labels = []

for idx, (file, covid) in metadata.iterrows():
    label = 'covid' if covid else 'normal'
    image = cv2.imread(f'images/{file}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    data.append(image)
    labels.append(label)

data = np.array(data) / 255.0
labels = np.array(labels)

# %% encode labels and split data
print('encode labels')

# one hot encoding labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# datasplit
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# %% create models
print('generate models')

baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# construct head of model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4,4))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(64, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

# loop over all layers in the base model and freeze them so they
# will *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# place the head FC model on top of base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# %% compiling the model
print('compile model')

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 8

# initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=15, fill_mode='nearest')
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# train network head, H not needed for now
# holds useful information about training progress
H = model.fit(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
)

# %% evaluate model
print("[INFO] evaluating network...")

predIDxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIDxs = np.argmax(predIDxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIDxs, target_names=lb.classes_))
# compute the confusion matrix and use it to derive the raw
# accuracy, sensitivity and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIDxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[1, 0] + cm[1, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# %% plot training loss and accuracy

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['accuracy'], label='train_accuracy')
plt.plot(np.arange(0, N), H.history['val_accuracy'], label='val_accuracy')
plt.title('Training Loss and Accuracy on Covid-19 Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])

# %% serialize model
print('[INFO] saving covid19 detector model')
model.save(args['model'], save_format='h5')