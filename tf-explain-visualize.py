import random
from argparse import ArgumentParser
from os import path

import numpy as np
import pandas as pd
import tf_explain
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB
from progress.bar import Bar
from tensorflow.keras.models import load_model

from utils.constants import CLASSES, IMG_DIMENSIONS

parser = ArgumentParser()
parser.add_argument("model", help=f"specify model ")
parser.add_argument("-c","--classindex",default=0,type=int)
parser.add_argument("-f","--classfilter",default=-1,type=int)
parser.add_argument("-l","--limit",default=25,type=int)
parser.add_argument('-r','--random', action='store_true')
parser.add_argument('-d','--dataset',default="dataset")
args = vars(parser.parse_args())

metadata = pd.read_csv(path.join(args['dataset'], 'metadata.csv'), usecols=['File', 'No Finding', 'Covid'],
                           dtype={'File': np.str, 'No Finding': np.bool, 'Covid': np.bool})
#metadata = metadata[0:4000] # for now only use 100 samples (10 positive, 90 negative) # TODO: comment out before comitting
data = []
labels = []
count=0
with Bar('Loading images', max=len(metadata)) as bar:
    for _idx, (file, noFinding, covid) in metadata.iterrows():
        bar.next()
        if covid: label = CLASSES[0] # covid
        elif noFinding: label = CLASSES[1] # healthy
        else: label = CLASSES[2] # other

        if args['classfilter']>-1 and label!=CLASSES[args['classfilter']]:
            continue

        image = imread(path.join(args['dataset'], 'images', file))
        image = cvtColor(image, COLOR_BGR2RGB)
        image = resize(image, IMG_DIMENSIONS)

        data.append(image)
        count+=1
        if not args['random'] and count==args['limit']:
            break

if args['random']:
    x = random.sample(range(0,len(data)),args['limit'])
    randomdata = []
    for i in x:
        randomdata.append(data[i])
    data=randomdata
    del randomdata

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
saveloc=path.join(path.dirname(args['model']),'visualized')
filename = f"{current_time}_class_{CLASSES[args['classindex']]}"
if args['classfilter']>-1:
    filename+=f"_only_{CLASSES[args['classfilter']]}"
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
