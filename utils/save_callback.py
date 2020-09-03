from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
import numpy as np
import logging as log
from sklearn.metrics import confusion_matrix
from os import remove

class SaveCallback(Callback):
    def __init__(self, test_data, test_labels, batch_size, trained_epochs):
        super().__init__()
        self._testData = test_data
        self._testLabels = test_labels
        self._BS = batch_size
        self._epoch = trained_epochs

    def on_epoch_end(self, _epoch, logs=None):
        self._epoch += 1 # increment epoch counter, ignore epoch passed by tensorflow
        if self._epoch % 5 != 0: return # early return, only proceed every 5 epochs
        print('')
        log.info(f'Intermediate network evaluation at epoch: {self._epoch}')

        # save tmp and load model to avoid mutating
        # the model thats being trained
        # TODO: keep intermediate saved models (path: ../intermediate_<epoch>.h5)
        modelPath = "model.tmp.h5"
        self.model.save(modelPath, save_format='h5')
        model = load_model(modelPath)
        remove(modelPath)
        predictions = np.argmax(model.predict(self._testData, batch_size=self._BS), axis=1)

        cm = confusion_matrix(self._testLabels.argmax(axis=1), predictions)
        total = sum(sum(cm))
        acc = (cm[0, 0] + cm[1, 1]) / total
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        log.info(f'TP: {cm[0, 0]}, FN: {cm[0, 1]}')
        log.info(f'FP: {cm[1, 0]}, TN: {cm[1, 1]}')
        log.info('acc: {:.4f}'.format(acc))
        log.info('sensitivity: {:.4f}'.format(sensitivity))
        log.info('specificity: {:.4f}'.format(specificity))
