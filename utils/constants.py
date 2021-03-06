import logging as log


CLASSES = ['covid', 'healthy', 'other']
IMG_DIMENSIONS = (224, 224)
IMG_DIMENSIONS_3D = (224, 224, 3)
BATCH_SIZE = 8
INIT_LR = 1e-3


def printSeparator(with_break=False):
    if with_break:
        log.info('======================================================================\n\n')
    else:
        log.info('======================================================================')
