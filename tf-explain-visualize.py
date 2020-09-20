
import tf_explain
from tensorflow.keras.models import load_model
import tensorflow as tf
import PIL

# Load pretrained model or your own
model = load_model("models/MobileNetV2/mobilenetv2_epoch_25.h5")
#model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights="imagenet", include_top=True)

model.summary()
#exit(0)
# Load a sample image (or multiple ones)
img = tf.keras.preprocessing.image.load_img("dataset/images/10318.jpg", target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)
data = ([img], None)

# Start explainer
explainer = tf_explain.core.grad_cam.GradCAM()

grid = explainer.explain(data, model,class_index=0)  # 281 is the tabby cat index in ImageNet
explainer.save(grid, ".", "grad_cam.png")