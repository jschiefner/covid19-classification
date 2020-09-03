import logging as log

def evaluate_confusion_matrix(cm):
    if cm.shape != (2, 2): raise ValueError('cm needs to be of shape (2,2)')

    # TODO: add more evaluations
    # use as follows: cm[row, col]
    total = sum(sum(cm))
    return {
        'accuracy': (cm[0, 0] + cm[1, 1]) / total,
        'error_rate': (cm[0, 1] + cm[1, 0]) / total,
        'precision': cm[0, 0] / (cm[0, 0] + cm[1, 0]),
        'sensitivity': cm[0, 0] / (cm[0, 0] + cm[0, 1]),
        'specificity': cm[1, 1] / (cm[1, 0] + cm[1, 1]),
        'false_alarm': cm[1, 0] / (cm[1, 0] + cm[1, 1]),
    }

def log_confusion_matrix(cm):
    if cm.shape != (2, 2): raise ValueError('cm needs to be of shape (2,2)')

    log.info(f'TP: {cm[0, 0]}, FN: {cm[0, 1]}')
    log.info(f'FP: {cm[1, 0]}, TN: {cm[1, 1]}')
