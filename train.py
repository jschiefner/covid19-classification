# %% parse arguments

from argparse import ArgumentParser
from os import environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # make tensorflow less verbose
from utils.management import *
from utils.constants import *
from utils.base_models import *

# example: python train.py dataset -m "MobileNetV2" -e 10 --evaluate 1 -v
parser = ArgumentParser()
parser.add_argument('dataset', help='path to input folder')
parser.add_argument("-m", "--model", default="VGG16", help=f"specify optional network. ")
parser.add_argument('-e', '--epochs', default=30, type=int, help='specify how many epochs the network should be trained at max, defaults to 30')
parser.add_argument('-v','--visualize', type=int, default=16, help='set to run tf-explain as callback, set 0 to deactivate, pos. value take x images, neg. value use all images')
parser.add_argument('--evaluate', default=0,type=int, help='set to evaluate the network after every x epochs')
parser.add_argument('-s','--save', default=0, type=int, help='unreliable computer, save current state after every x epochs')
parser.add_argument('-a','--autostop', action='store_true', help='set to stop training when val_loss rises')
args = vars(parser.parse_args())

# metadata check
check_if_exists_or_exit(args['dataset'])

# model check
modelFunc = get_model_by_name(str(args['model'])) # returns None if model does not exist
if modelFunc is None:
    if not check_if_custom_model_name_exists(str(args['model'])):
        print(f'[ERROR] Choose an appropriate model to continue, must be one out of: {func_names}.')
        print(f'[ERROR] Or choose a custom model lying in folder models.')
        exit(1)

check_and_create_folder('models')
modelFolderPath = path.join('models', args['model'])
modelDataPath = path.join(modelFolderPath, 'data.csv')
modelLogPath = path.join(modelFolderPath, 'training.log')
modelCheckpointsPath = path.join(modelFolderPath, 'checkpoints')
if not check_and_create_folder(modelFolderPath):
    modelExists = False
    check_and_create_folder(modelCheckpointsPath)
else:
    modelExists = path.exists(modelDataPath)

# %% import rest of dependencies and configure logging

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sys import stdout
from utils.evaluation_callback import EvaluationCallback

from utils.data import *

# set GPU configs, might have to comment out
# these lines if you're working on a cpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler(modelLogPath),
        log.StreamHandler(stdout)
    ]
)

printSeparator()
log.info(f'Starting Training with arguments: {args}')

# %% load model

if modelExists:
    model, modelData, trainEpochs, trainedEpochs = load_existing_model(args['model'], modelDataPath, args['epochs'])
else:
    log.info('Model does not exist yet, creating a new one')
    baseModel = modelFunc(weights='imagenet', include_top=False, input_shape=IMG_DIMENSIONS_3D,input_tensor=Input(shape=IMG_DIMENSIONS_3D))
    baseModel.trainable = False

    log.info(f'baseModel: {args["model"]}')

    # # vgg16 top adapted
    x = GlobalAveragePooling2D(name='global_avg_pool2d')(baseModel.output)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.3, name='dropout1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(3, activation='softmax', name='predictions')(x)

    model = Model(inputs=baseModel.input, outputs=x)
    model.summary()
    trainEpochs = args['epochs']
    trainedEpochs = 0
    log.info(f'trainEpochs: {trainEpochs}')

log.info('Model Overview:')
model.summary()

check_if_trained_or_exit(trainEpochs, args['epochs'])

# %% prepare data

trainX, valX, trainY, valY, testData, testLabels = load_dataset(args['dataset'], validation_after_train_split=0.1)

# %% train model

# initialize the training data augmentation object
trainAug = ImageDataGenerator(rotation_range=10,
                              horizontal_flip=True,   # cc: die schieferen scheinen so maximal 20, die allermeisten aber <5
                              fill_mode='nearest',    # Testweise bessere performance als constant fill
                              width_shift_range=0.1,  # Horizontal sind die Bilder größtenteils gut zentriert
                              height_shift_range=0.2) # Vertikal tendenziell etwas schlechter
opt = Adam(lr=INIT_LR, decay=INIT_LR / trainEpochs)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# setup callbacks
from utils.my_gradcam_callback import MyGradCAMCallback

callback_gradcam = MyGradCAMCallback(
        validation_data=(valX, valY),
        class_index=0,
        trained_epochs=trainedEpochs,
        output_dir=path.join(modelFolderPath,"visualized"),
        limit=args['visualize'],
        )
callback_modelcheckpoint = ModelCheckpoint(
        filepath=f'models/{args["model"]}/checkpoints/checkpoint_epoch{trainedEpochs}' + '+{epoch}' + '_ckpt-loss={loss:.2f}.h5',
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        period=args['save'],
    )
callback_evaluation = EvaluationCallback(
        test_data=testData,
        test_labels=testLabels,
        batch_size=BATCH_SIZE,
        model_name=args['model'],
        trained_epochs=trainedEpochs,
        freq=args['evaluate']
    )
callback_earlyStopping = EarlyStopping(
    monitor='val_loss',
    patience=1,
)

callbacks = []
if args['visualize']!=0: callbacks.append(callback_gradcam)
if args['evaluate']>0: callbacks.append(callback_evaluation)
if args['save']>0: callbacks.append(callback_modelcheckpoint)
if args['autostop']: callbacks.append(callback_earlyStopping)

# train network head, H not needed for now
# holds useful information about training progress
try:
    model.fit(
        trainAug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        validation_data=(valX, valY),
        validation_steps=len(valX) // BATCH_SIZE,
        epochs=trainEpochs,
        callbacks=callbacks,
    )
except KeyboardInterrupt:
    print()
    log.info(f'interrupted Training at epoch: {len(model.history.epoch)+1}')
    if len(model.history.epoch) == 0:
        exit(0)

# save history and epochs for later
# as history gets deleted when model.predict() is called
history = model.history.history.copy()
epochs = len(model.history.epoch)

# %% validate model

predictions = np.argmax(model.predict(testData, batch_size=BATCH_SIZE), axis=1)
report = classification_report(testLabels.argmax(axis=1), predictions, target_names=CLASSES)
log.info(f'Evaluating trained network\n{report}')
cm = confusion_matrix(testLabels.argmax(axis=1), predictions)

# %% serialize model and csv

if modelExists:
    epochs += trainedEpochs
    log.info(f'epoch: {epochs}')
    modelData = modelData.append(pd.DataFrame(history))
    modelData.index = np.arange(0, len(modelData))
else:
    log.info(f'epoch: {epochs}')
    modelData = pd.DataFrame(history)

persist_results(model, modelData, modelFolderPath, epochs)
printSeparator(with_break=True)
