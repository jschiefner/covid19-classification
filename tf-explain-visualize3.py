import tf_explain
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
import os
import PIL
from utils.constants import CLASSES, IMG_DIMENSIONS
import numpy as np
from os import path
# Load pretrained model or your own
model = load_model(sys.argv[1])

#model.summary()

explainer = tf_explain.core.grad_cam.GradCAM()

#exit(0)
#files = os.listdir(sys.argv[2])
#files = files[:25]

# folgende bilder alle covid positiv
files = """18121.png
67115.jpg
35484.png
70511.png
70979.jpeg
32374.jpeg
43813.jpeg
70391.jpg
93942.jpeg
16782.jpg
69004.png
19075.jpg
93566.jpeg
91975.jpeg
80115.jpeg
64811.jpg
13916.jpeg
50862.jpeg
65417.jpeg
36118.jpeg
37763.jpeg
39115.jpg
99652.png
24466.jpeg
48527.png
19286.png
69202.jpeg
42081.jpg
82224.jpg
80404.jpeg
47981.jpeg
39162.jpg
73955.png
25308.jpeg
99471.jpeg
88355.jpeg
67605.jpeg
82283.jpg
64422.jpg
62006.jpeg
89312.jpeg
41987.png
98746.jpg
90702.jpeg
47811.jpeg
84671.png
97979.jpeg
32051.jpeg
20792.png
64938.jpeg
28014.png
47271.jpg
73377.jpeg
80089.jpg
21593.png
22165.jpg
97175.png
88851.jpg
52533.jpg
60429.jpg
27561.jpg
15643.jpeg
71407.jpg
15101.jpeg
33536.jpeg
95435.png
58635.jpeg
69209.jpeg
91988.jpg
30219.png
98491.jpeg
88101.png
64903.jpeg
94188.jpeg
34205.png
68802.jpg
37143.png
22078.jpeg
45083.jpeg
90502.jpeg
11067.jpg
44995.jpeg
14810.png
72727.jpeg
74652.png
78983.jpeg
92022.jpg
18169.jpg
25497.png
22441.png
46943.png
39130.jpeg
21405.png
91274.jpeg
47248.jpeg
90302.jpg
27392.jpeg
34308.jpeg
59832.jpeg
44196.png""".split("\n")
print(files)
from cv2 import imread, cvtColor, resize, COLOR_BGR2RGB,imwrite

imgdata = []
for f in files:
    try:

        image = imread(path.join('dataset','images', f))
        imwrite(f"heatmapexamples/{f}",image)
        image = cvtColor(image, COLOR_BGR2RGB)
        image = resize(image, IMG_DIMENSIONS)
        imgdata.append(image)
    except:
        print(f"Error encountered: {f}")

imgdata = np.array(imgdata)/255.0
data = (imgdata, None)
# Start explainer
for layer in model.layers:
    if "conv" in layer.name.lower():
        gradcam_layer=layer.name

# Start explainer
grid = explainer.explain(data, model,class_index=0,layer_name=gradcam_layer)  #
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H_%M_%S")
#saveloc=path.join(path.dirname(args['model']),'visualized')
saveloc = "heatmapexamples"
filename = f"{current_time}"
filename+=".jpg"
try:
    explainer.save(grid,f"{saveloc}", f"{filename}")
    print(f"GradCam image created and saved to {saveloc}/{filename}")
except:
    print("F")
