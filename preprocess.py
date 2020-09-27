from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import exists, join
import pandas as pd
from utils.management import check_and_create_folder

parser = ArgumentParser()
parser.add_argument('dataset', help='path to input folder')
parser.add_argument('-o', '--outputfolder', default='augmented_images', help='path to output folder')
parser.add_argument('-c', '--count', default=10, type=int, help='number of random augmentations for each image')
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
fileList = listdir(join(args['dataset'], 'images'))
for file in fileList:
    if file[-4:] in (".jpg", "jpeg", ".png"):
        img = load_img(join(args['dataset'], 'images', file))  # PIL image
        x = img_to_array(img)  # numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # numpy array with shape (1, 3, 150, 150)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=imagesOutPath,
                                  save_prefix=file, save_format='jpeg'):
            i += 1
            if i >= args['count']:
                break  # otherwise the generator would loop indefinitely

# Second pass: we don't know the augemented file names beforehand,
# so we iterate over the output folder once more to fill in metadata
print('Writing metadata...')
oldMetadata = pd.read_csv(join(args['dataset'], "metadata.csv"))
metadata = pd.DataFrame(columns=oldMetadata.columns)
for file in listdir(imagesOutPath):
    original = file.split(sep='_')[0]  # obtain the original file name for metadata lookup
    data = oldMetadata.loc[oldMetadata["File"] == original]
    data.head()['File'] = file
    metadata = metadata.append(data)

metadata.to_csv(join(args['outputfolder'], "metadata.csv"))
