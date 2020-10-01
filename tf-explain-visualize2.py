
import tf_explain
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
from argparse import ArgumentParser
import os
import PIL
import pandas as pd
import numpy as np
from os import path
from progress.bar import Bar
from utils.constants import CLASSES, IMG_DIMENSIONS
import random
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB

parser = ArgumentParser()
parser.add_argument("model", help=f"specify model ")
parser.add_argument("-c","--classindex",default=0,type=int)
parser.add_argument('-d','--dataset',default="dataset")
args = vars(parser.parse_args())
data = []

from os import listdir
from os.path import isfile, join
mypath = args['dataset']
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for f in onlyfiles:
    print(f)
    image = imread(path.join(args['dataset'], f))
    image = cvtColor(image, COLOR_BGR2RGB)
    image = resize(image, IMG_DIMENSIONS)
    data.append(image)



data = np.array(data) / 255.0


explainer = tf_explain.core.grad_cam.GradCAM()


data = (data, None)

model = load_model(args['model'])


for layer in model.layers:
    if "conv" in layer.name.lower():
        gradcam_layer=layer.name


# Start explainer
grid = explainer.explain(data, model,class_index=args['classindex'], layer_name=gradcam_layer)  #
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H_%M_%S")
#saveloc=path.join(path.dirname(args['model']),'visualized')
saveloc = mypath
filename = f"{current_time}_class_{CLASSES[args['classindex']]}"
filename+=".jpg"
try:
    explainer.save(grid,f"{saveloc}", f"{filename}")
    print(f"GradCam image created and saved to {saveloc}/{filename}")
except:
    print("F")




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