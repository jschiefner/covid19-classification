from tensorflow.keras.applications import *
''' # tf==2.3.0
MODELS = [
    Xception,
    VGG16,
    VGG19,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet50V2,
    ResNet101V2,
    ResNet152V2,
    InceptionV3,
    InceptionResNetV2,
    MobileNet,
    MobileNetV2,
    DenseNet121,
    DenseNet169,
    DenseNet201,
    NASNetMobile,
    NASNetLarge,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
]
'''

MODELS = [   # tf 2.2.0
    Xception,
    VGG16,
    VGG19,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet50V2,
    ResNet101V2,
    ResNet152V2,
    InceptionV3,
    InceptionResNetV2,
    MobileNet,
    MobileNetV2,
    DenseNet121,
    DenseNet169,
    DenseNet201,
    NASNetMobile,
    NASNetLarge,
]

func_names = [m.__name__ for m in MODELS]
funcs_dict = dict(zip(func_names, MODELS))

def get_all_model_names():
    return func_names

def get_model_by_name(m):
    if not m in funcs_dict:
        return None
    return funcs_dict[m]

def get_all_model_names():
    return func_names

def get_model_by_name(m):
    if not m in funcs_dict:
        return None
    return funcs_dict[m]


