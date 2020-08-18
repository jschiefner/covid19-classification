# %% imports
print('import lots of stuff')


from tf_explain.core.grad_cam import GradCAM
from tf_explain.callbacks.grad_cam import GradCAMCallback

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
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import time
import os

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

cur_time_secs = lambda : int(round(time.time() * 1000 * 1000))

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 1
BS = 8


# %% arg "parse"

args = {}

# %% prepare data
print('prepare data')

metadata = pd.read_csv('metadata.csv', usecols=['File', 'Covid'], dtype={'File': np.str, 'Covid': np.bool})
metadata = metadata[100:200] # for now only use 100 samples (5 positive, 95 negative)

covid = metadata[metadata['Covid'] == True]
healthy = metadata[metadata['Covid'] == False]

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

# initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=15, fill_mode='nearest')

# %% VGGnet model

#nets = [NASNetLarge, VGG16, MobileNetV2] # https://keras.io/api/applications/#:~:text=Keras%20Applications%20are%20deep%20learning,They%20are%20stored%20at%20~%2F.
nets = [MobileNetV2]

print('generate model')
baseModels = [] # array with basemodels
for net in nets:
    baseModel = net(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    for layer in baseModel.layers:
        layer.trainable = False
    baseModels.append(baseModel) # downloads weights.h5 file

for baseModel in baseModels:

    # construct head of model that will be placed on top of the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4,4))(headModel)
    headModel = Flatten(name='flatten')(headModel)
    headModel = Dense(64, activation='relu')(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation='softmax')(headModel)

    # place the head FC model on top of base model (this will become the actuel model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they iwll *not* be updated during the first training process


    # %% compiling the model
    print('compile model')



    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


    #### tf explain test callbacks
    callbacks = [
        GradCAMCallback(
            validation_data=(testX, testY),
            layer_name="activation_1",
            class_index=0,
            output_dir="tf-explain-test",
        )
    ]

    ### 

    # train network head
    H = model.fit(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save("covid19model"+str(cur_time_secs())+".h5", save_format="h5")

    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,	target_names=lb.classes_))

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))


