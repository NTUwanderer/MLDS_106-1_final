# url: https://keras.io/applications

import keras
from keras.preprocessing import image
import argparse, os
import numpy as np

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

tempImgPath = 'DeepQ-Vivepaper/data/air/img'
image_dim = (460, 612, 3)

parser = argparse.ArgumentParser()
parser.add_argument('extractor', choices = ['', ''], help = 'sport choice')

feature_extractor_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

img = image.load_img(os.path.join(tempImgPath, 'img_00000.png'), target_size=image_dim)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = feature_extractor_model.predict(x)

print ('preds shape: ', preds.shape)


