# Lab Course Covid-19 Classification

This repository is part of our lab course for Pattern Recognition. The task was to identify covid-19 in x-ray images.
We use Convolutional Neural Networks in Order to solve this. Instructions on how to use all of the files are below.

## Participants

 - @cccastan
 - @TheNeedForSleep
 - @jschiefner
 
## Requirements

Everything here is implemented in python 3.7. For most of the functionality we use:

 - Keras/Tensorflow
 - tf-explain
 - OpenCV
 - Pandas
 - Scikit-Learn
 - Matplotlib

A specification of all dependencies can be found in the `Pipfile`, versions are available in `Pipfile.lock`.
We recommend using `pipenv` to set up the project. With it, you can run `pipenv install` to make all dependencies available.

## Instructions

We have combined several tools that are part of the pipeline in producing or using pretrained neural networks, train, predict and evaluate them.
Some programs require a `<dataset>` argument. For that you must pass a folder that has the following structure:

```
<dataset>
   ├── metadata.csv
   └── images
       ├── image1.png
       ├── image2.png
       ├── ...
```

The `metadata.csv` file must have a File column which contains only the filenames to the files in the `images` folder.
Apart from that, it needs to have the right columns, as you can see in the `utils/data.py` file.

The main two files are `train.py` and `predict.py`.
They enable you to use a pretrained Keras model directly and tune it to the task and then use that trained model in the next step for prediction.

#### Training

If you want to use a pretrained Keras Model just choose one from their list of applications over at https://keras.io/api/applications/.
Then, start the training program with `python train.py <dataset> -m <modelName>`, so for example `python train.py data -m VGG16` (VGG16 will also be the default).

#### Custom Model

If you want to create your own model, adjust the layers in `custom_model_creator.py` and run `python custom_model_creator.py -m <your_model_name>`.
It will be saved in the `models` folder and you can train it by running `python train.py <dataset> -m <your_model_name>`
For training there are several other optional parameters you can see by running `python train.py -h`, for example you can set how many epochs you want to train with `-e <epochs>`.
If you want to stop the training before it has reached the maximal epochs you can interrupt the program during training.
If you run it again with the same Model it will continue the training where you left off (the program saves its progress in the `models` folder).

#### Prediction

To predict with some testing images, run `python predict.py <image_folder> <outfile.csv>`.
If you want you can specify a model that should be used for prediction with `-m <model>` or multiple models to predict using an ensemble classification using `-e "<model1> <model2>"`.
Again, `predict.py` looks for those models in the `models` folder, so make sure they are correctly saved in there after training.
If you dont specify a model, or specify `-m all` all trained models from the `models` folder will be used for an ensemble prediction.

#### Data Augmentation

If you want agumentate your dataset run `python preproces.py <dataset>`. Optional parameters are available and can be seen by running `python preprocess.py -h`.

#### Training visualization

Visualizing the training progress after a training session can be done by running `python visualize.py -m <model>`.
It gives you a graph of the training loss/accuracy and the validation loss/accuracy.

#### Gradcam Heatmap Visualizations

Using `train.py` gradcam heatmaps will already be created during training. This should be enough for most cases.
If you need more sophisticated options the `tf-explain-visualize[i].py` can be used to create heatmaps for the images.
Run `python tf-explain-visualize[i].py -m models/<model_name>/epoch[j].h5 -d <dataset>`

 - `tf-explain-visualize.py`
   - Creates Heatmaps
   - Requires path to the dataset folder
   - is adjustable by following parameters:
     - `-f` filter by class (only images with covid)
	 - `-c` select class index (important locations for the decision get highlighted)
	 - `-l` limit to x amount of images (16 for 4x4)
	 - `-r` set flag to use random pictures instead of the first ones in the folder
		
 - `tf-explain-visualize2.py` 
   - Creates Heatmaps out of all images in the given folder
   - Requires a model and the image folder
   - A class index can be given

 - `tf-explain-visualize.py`
   - Creates Heatmap out of the 100 Corona positive images
   - Requires a model as parameter
   - Image path is currently hardcoded

#### Utils

Some functionality is combined into a `utils` module to keep the main files arranged more clearly.
