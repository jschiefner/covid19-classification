from keras.models import load_model
from keras.utils.vis_utils import plot_model


model = load_model("models/VGG16/epoch_2.h5")
# baseModel = Sequential()
# x = GlobalAveragePooling2D(name='global_avg_pool2d')
# # x = Flatten(name='flatten')(baseModel.output)
# x = Dense(256, activation='relu', name='fc1')(x)
# x = Dropout(0.3)(x)
# x = Dense(128, activation='relu', name='fc2')(x)
# x = Dropout(0.5)(x)
# x = Dense(3, activation='softmax', name='predictions')(x)

plot_model(model, to_file='horizontalvgg16model.png',show_shapes=True,rankdir='TB')
