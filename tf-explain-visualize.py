
import tf_explain
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
from argparse import ArgumentParser
import os
import PIL


parser = ArgumentParser()
parser.add_argument('images', help='path to images')
parser.add_argument("model", help=f"specify model ")
args = vars(parser.parse_args())

# Load pretrained model or your own
model = load_model(args['model'])

for layer in model.layers:
    if "conv" in layer.name.lower():
        gradcam_layer=layer.name

explainer = tf_explain.core.grad_cam.GradCAM()

# folgende bilder alle covid positiv
files = """94188.jpeg
25497.png
47248.jpeg
33536.jpeg
39130.jpeg
50862.jpeg
39162.jpg
90702.jpeg
80115.jpeg
27561.jpg
99471.jpeg
58635.jpeg
21498.jpg
73377.jpeg
22165.jpg
70391.jpg
64422.jpg
13916.jpeg
16596.jpg
88101.png
69202.jpeg
80404.jpeg
32051.jpeg
15643.jpeg
91777.jpg""".split("\n")
print(files)

imgdata = []
for f in files:
    try:
        # Load a sample image (or multiple ones)
        img = tf.keras.preprocessing.image.load_img(f"{args['images']}/{f}", target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img /= 255.0
        imgdata.append(img)
    except:
        print(f"Error encountered: {f}")

data = (imgdata, None)
# Start explainer
grid = explainer.explain(data, model,class_index=0, layer_name=gradcam_layer)  #
explainer.save(grid, "visualized", f"grad_cam_corona_30_09.jpg")
print(f"GradCam image created and saved: ")




'''
for f in files:
    try:
        # Load a sample image (or multiple ones)
        img = tf.keras.preprocessing.image.load_img(f"{sys.argv[2]}/{f}", target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        data = ([img], None)
        # Start explainer
        grid = explainer.explain(data, model,class_index=1)  #
        explainer.save(grid, "visualized", f"grad_cam_{f.split('.')[0]}.png")
        print(f"GradCam image created and saved: {f}")
    except:
        print("Error encountered")
'''