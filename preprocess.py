from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import exists, join, splitext
import pandas as pd
from utils.management import check_and_create_folder
import numpy as np
from progress.bar import Bar

parser = ArgumentParser()
parser.add_argument('dataset', help='path to input folder')
parser.add_argument('-o', '--outputfolder', default='augmented_images', help='path to output folder')
parser.add_argument('-c', '--count', default=3, type=int, help='number of random augmentations for each image')
args = vars(parser.parse_args())

# Subject to change
datagen = ImageDataGenerator(rotation_range=10,
                             horizontal_flip=True,   # Das könnte blödsinnig sein
                             fill_mode='nearest',    # Constant black fill instead?
                             width_shift_range=0.1,  # Horizontal sind die Bilder größtenteils gut zentriert
                             height_shift_range=0.2, # Vertikal tendenziell etwas schlechter
                             zoom_range=0.2)

check_and_create_folder(args['outputfolder'])
imagesOutPath = join(args['outputfolder'], 'images')
check_and_create_folder(imagesOutPath)

print('Augmenting...')
metadata = pd.read_csv(join(args['dataset'], 'metadata.csv'))

outdata = pd.DataFrame(columns=metadata.columns)

with Bar('Processing images', max=len(metadata)) as bar:
    for _, row in metadata.iterrows():
        bar.next()
        img = load_img(join(args['dataset'], 'images', row.File))  # PIL image
        img = img_to_array(img)  # numpy array with shape (3, 150, 150)
        img = img.reshape((1,) + img.shape)  # numpy array with shape (1, 3, 150, 150)

        count = 0
        maxCount = args['count'] * 20 if row.Covid else args['count']
        for batch in datagen.flow(img, batch_size=1, save_to_dir=imagesOutPath,
                                  save_prefix=row.File, save_format='jpg'):
            count += 1
            if count >= maxCount:
                break  # otherwise the generator would loop indefinitely

print('Writing metadata...')
# Second pass: we don't know the augemented file names beforehand,
# so we iterate over the output folder once more to fill in metadata
for file in listdir(imagesOutPath):
    original = file.split('_')[0]  # obtain the original file name for metadata lookup
    data = metadata.loc[metadata['File'] == original]
    data.head()['File'] = file
    outdata = outdata.append(data)

outdata.index = np.arange(0, len(outdata))
outdata.to_csv(join(args['outputfolder'], "metadata.csv"))
