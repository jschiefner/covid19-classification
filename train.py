# %% imports

print('import lots of stuff')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import *
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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# %% arg "parse"

availablemodels = [Xception,VGG16,VGG19,ResNet50,ResNet101,ResNet152,ResNet50V2,ResNet101V2,ResNet152V2,InceptionV3,InceptionResNetV2,MobileNet,MobileNetV2,DenseNet121,DenseNet169,DenseNet201,NASNetMobile,NASNetLarge,EfficientNetB0,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6,EfficientNetB7]

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input", required=True, help="path to folder containing the xray images") #
parser.add_argument("-d", "--dataset", required=True, help="input metadata.csv")
parser.add_argument("-m","--model", default="VGG16", help="specify optional network")
args = vars(parser.parse_args())


func_names = [m.__name__ for m in availablemodels]
funcs_dict = dict(zip(func_names, availablemodels))
modelFunc = funcs_dict[str(args['model'])]


# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 8


# %% prepare data
print('prepare data')

metadata = pd.read_csv(args['dataset'], usecols=['File', 'Covid'], dtype={'File': np.str, 'Covid': np.bool})
#metadata = metadata[100:200] # for now only use 100 samples (5 positive, 95 negative)

covid = metadata[metadata['Covid'] == True]
healthy = metadata[metadata['Covid'] == False]

data = []
labels = []

for idx, (file, covid) in metadata.iterrows():
    label = 'covid' if covid else 'normal'
    #image = cv2.imread(f'images/{file}')
    image = cv2.imread(args['input']+f'{file}')
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
#random_stateint Controls the shuffling applied to the data before applying the split.
#Pass an int for reproducible output across multiple function calls. See Glossary.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels ) # random_state=42
# initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=15, fill_mode='nearest') # adapt rotation range? wie schief ist so ein röntgen bild wohl maxial aufgenommen worden


# check if this kind of model already exists
# falls ja lade dieses und trainiere weiter

# sonst


# %% model
print('generate model')
baseModel = modelFunc(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3))) # downloads weights.h5 file

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
for layer in baseModel.layers:
    layer.trainable = False

# %% compiling the model
print('compile model')



opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
exit(0)
# train network head
H = model.fit(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
)

# todo XX
XX= 1

model.save("covid19_model"+args['model']+"_epoch"+XX+".h5", save_format="h5")
