from os import path, mkdir, listdir
from tensorflow.keras.models import load_model
from glob import glob
import pandas as pd
import logging as log

def check_and_create_folder(dir):
    """
    checks if a directory is present and creates it if it isnt
    :param dir: path to directory
    :return: True if directory existed, False if not
    """
    if not path.isdir(dir):
        mkdir(dir)
        return False
    return True

# TODO: make sure model exists instead of folder
def check_if_exists_or_exit(file_or_folder):
    if not path.exists(file_or_folder):
        print(f'[ERROR] the path "{file_or_folder}" does not exist. Please supply a valid path.')
        exit(1)

def check_if_trained_or_exit(train_epochs, requested_epochs):
    if train_epochs <= 0:
        log.info(f'network is already trained on {requested_epochs} epochs, exiting')
        log.info('if you want to train the network for longer, set the optional --epochs argument')
        exit(0)

def load_existing_model(model_name, model_data_path, epochs):
    log.info('Model exists!')
    log.info(f'If you want to train the model from the beginning, remove the models/{model_name}/ folder')
    modelData = pd.read_csv(model_data_path, index_col=0)
    trainedEpochs = len(modelData)
    trainEpochs = epochs - trainedEpochs
    log.info(f'trained epochs: {trainedEpochs}, train epochs: {trainEpochs}, (total: {trainEpochs + trainedEpochs})')
    modelPaths = glob(f'models/{model_name}/epoch_*.h5')
    log.info(f'modelPaths: {modelPaths}')
    latestModelPath = modelPaths[-1]
    model = load_model(latestModelPath) # load latest model
    log.info(f'successfully loaded "{latestModelPath}"')
    return model, modelData, trainEpochs, trainedEpochs

def persist_results(model, model_data, model_folder_path, epochs):
    """
    persists training results such that it is all available for further training
    :param model: model to be saved
    :param model_data: modelData (training history)
    :param model_folder_path: path to the modelFolder
    :param epochs: epochs the model has been trained in total
    """
    modelPath = path.join(model_folder_path, f'epoch_{epochs}.h5')
    csvPath = path.join(model_folder_path, 'data.csv')
    log.info(f'saving model to: "{modelPath}", saving csv to: "{csvPath}"')
    model.save(modelPath, save_format='h5')
    model_data.to_csv(csvPath)

def find_and_load_models():
    files = listdir("models")
    models = []
    for f in files:
        _path = path.join('models',f, 'data.csv')
        #print(_path)
        if path.exists(_path):
            models.append(f)
    #print(files)
    #print(models)
    #exit(0)
    return models

def check_if_custom_model_name_exists(m):
    mm = find_and_load_models()
    print(m)
    print(mm)
    if m in mm:
        return True
    return False

