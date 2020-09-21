import logging as log


CLASSES = ['covid', 'other', 'healthy']
IMG_DIMENSIONS = (224, 224)
IMG_DIMENSIONS_3D = (224, 224, 3)



def printSeparator(with_break=False):
    if with_break:
        log.info('======================================================================\n\n')
    else:
        log.info('======================================================================')
