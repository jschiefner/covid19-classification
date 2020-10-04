# %% imports

from os import environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # make tensorflow less verbose
from argparse import ArgumentParser
from sys import stdout
import tensorflow as tf
from utils.management import *
from utils.constants import *
from utils.data import *
from progress.bar import Bar
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

# set GPU configs, might have to comment out
# these lines if you're working on a cpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler('evaluation.log'),
        log.StreamHandler(stdout),
    ]
)

printSeparator()

# %% parse arguments

parser = ArgumentParser()
parser.add_argument('dataset', help='path to input folder')
parser.add_argument('-m', '--model', required=False, default='all', help='models to be used (uses all it can find by default), possible to specify multiple: -m "VGG16 MobileNetV22"')
args = vars(parser.parse_args())

# verify dataset path
if not path.isdir(args['dataset']):
    log.error(f'the dataset path "{args["dataset"]}" is not valid.')
    log.error('Please supply the path to dataset folder')
    exit(1)

metadataPath = path.join(args['dataset'], "metadata.csv")
check_if_exists_or_exit(metadataPath)

# check if multiple models given, its ensemble classification then
if args['model'] == 'all':
    models = find_and_load_models()
else:
    models = args['model'].split(" ")

# verify model paths
modelDataPaths = []
for model in models:
    _path = path.join('models', model, 'data.csv')
    if not path.exists(_path):
        log.error(f'the model "{model}" has not been trained yet.')
        log.error('Please train the model first before predicting with it.')
        exit(1)
    else:
        modelDataPaths.append(_path)

number_of_models = len(models)

log.info(f'Starting Evaluation with arguments: {args}')

# %% load images for real

metadata = pd.read_csv(
    metadataPath,
    usecols=['File', 'No Finding', 'Covid'],
    dtype={'File': np.str, 'No Finding': np.bool, 'Covid': np.bool}
)
data, labels = [], []
dataLength = len(metadata)
splitPoint = dataLength - int(dataLength * 0.1)
metadata = metadata[splitPoint:]
with Bar('Loading images', max=len(metadata)) as bar:
    for _idx, (file, noFinding, covid) in metadata.iterrows():
        bar.next()
        if covid: label = CLASSES[0] # covid
        elif noFinding: label = CLASSES[1] # healthy
        else: label = CLASSES[2] # other
        image = imread(path.join(args['dataset'], 'images', file))
        image = cvtColor(image, COLOR_BGR2RGB)
        image = resize(image, IMG_DIMENSIONS)

        data.append(image)
        labels.append(label)

data = np.array(data) / 255.0
labels = np.array(labels)
log.info(f'successfully loaded {len(metadata)} images')

log.info('encode labels') # one hot encoding labels
lb = LabelBinarizer()
lb.fit(CLASSES) # changes order of classes: covid, other, healthy
labels = lb.transform(labels)

# %% run predictions models
predictionsList = []
for modelDataPath, model in zip(modelDataPaths, models):
    modelData = pd.read_csv(modelDataPath, index_col=0)
    epochs = len(modelData)
    log.info(f'model "{model}" was trained for {epochs} epochs')
    modelPath = path.join('models', model, f'epoch_{epochs}.h5')
    model = load_model(modelPath)
    log.info(f'successfully loaded "{modelPath}"')
    log.info(f'predicting...')
    predictionsList.append(model.predict(data))

predictionsList = np.array(predictionsList)
summedUpPredictions = predictionsList.sum(axis=0)
log.info('predictions have been calculated')

# %% print classification report and confusion matrix
report = classification_report(
    labels.argmax(axis=1),
    summedUpPredictions.argmax(axis=1),
    target_names=CLASSES,
)
cm = confusion_matrix(
    labels.argmax(axis=1),
    summedUpPredictions.argmax(axis=1),
)
log.info(f'Classification report:\n{report}')
log.info(f'Confusion Matrix:\n{cm}')
printSeparator(with_break=True)
