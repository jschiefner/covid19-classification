from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
import numpy as np
import logging as log
from sklearn.metrics import confusion_matrix
from os import path, remove
from utils.evaluation import evaluate_confusion_matrix, log_confusion_matrix

class SaveCallback(Callback):
    def __init__(self, test_data, test_labels, result_frame, batch_size, model_name, trained_epochs):
        super().__init__()
        self._testData = test_data
        self._testLabels = test_labels
        self._resultFrame = result_frame
        self._BS = batch_size
        self._modelName = model_name
        self._epoch = trained_epochs

    def on_epoch_end(self, _epoch, logs=None):
        self._epoch += 1 # increment epoch counter, ignore epoch passed by tensorflow
        # if self._epoch % 5 != 0: return # TODO?: early return, only proceed every 5 epochs
        print('')
        log.info(f'Intermediate network evaluation at epoch: {self._epoch}')

        # save and load model to avoid mutating the model thats being trained
        modelPath = path.join('models', self._modelName, f'intermediate_{self._epoch}.h5')
        self.model.save(modelPath, save_format='h5')
        model = load_model(modelPath)
        remove(modelPath) # TODO: decide whether to keep/remove model
        predictions = np.argmax(model.predict(self._testData, batch_size=self._BS), axis=1)

        cm = confusion_matrix(self._testLabels.argmax(axis=1), predictions)
        evaluation = evaluate_confusion_matrix(cm)
        log_confusion_matrix(cm)
        for metric in evaluation: log.info(metric + ': {:.4f}'.format(evaluation[metric]))

        # updates same reference as passed in on initialization
        self._resultFrame.loc[len(self._resultFrame)] = {
            'TP': cm[0, 0], 'FN': cm[0, 1],
            'FP': cm[1, 0], 'TN': cm[1, 1],
        }
