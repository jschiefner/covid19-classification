
import tf_explain
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
import os
import PIL

# Load pretrained model or your own
model = load_model(sys.argv[1])

#model.summary()

explainer = tf_explain.core.grad_cam.GradCAM()

#exit(0)
#files = os.listdir(sys.argv[2])
#files = files[:25]

# folgende bilder alle covid positiv
files = """58172.jpg
55115.jpg
36991.jpg
93154.jpg
54202.jpg
81806.jpg
76865.jpg
44795.jpg
99529.jpg
65696.jpg
93062.jpg
92621.jpg
34203.jpg
25075.jpg
22504.jpg
45129.jpg
63291.jpg
55993.jpg
62688.jpg
75413.jpg
93122.jpg
41524.jpg
53361.jpg
87654.jpg
20356.jpg
""".split("\n")
print(files)

imgdata = []
for f in files:
    try:
        # Load a sample image (or multiple ones)
        img = tf.keras.preprocessing.image.load_img(f"{sys.argv[2]}/{f}", target_size=(224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        imgdata.append(img)
    except:
        print(f"Error encountered: {f}")

data = (imgdata, None)
# Start explainer
grid = explainer.explain(data, model,class_index=0)  #
explainer.save(grid, "visualized", f"grad_cam_corona_0-24.png")
print(f"GradCam image created and saved: {f}")




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