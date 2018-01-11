# url: https://keras.io/applications

import keras
from keras.preprocessing import image
import argparse, os
import numpy as np

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

#tempImgPath = 'DeepQ-Vivepaper/data/air/img'
image_dim = (460, 612, 3)
feature_size = 2048

feature_extractor_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

#img = image.load_img(os.path.join(tempImgPath, 'img_00000.png'), target_size=image_dim)
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#preds = feature_extractor_model.predict(x)
#print ('preds shape: ', preds.shape)

def getFeatures(pairs):
    temp_batch=8

    features = np.zeros((len(pairs), feature_size))
    temp_imgs = np.zeros([temp_batch] + list(image_dim))
    progress_index = 0

    count = 0
    for i in range(len(pairs)):
        print ('i: ', i)
        img = image.load_img(pairs[i][0], target_size=image_dim)
        temp_imgs[i % temp_batch] = image.img_to_array(img)

        if ((i > 0 and (i + 1) % temp_batch == 0) or i == len(pairs) - 1):
            preds = feature_extractor_model.predict(preprocess_input(temp_imgs))
            features[progress_index : i + 1] = preds[: i + 1 - progress_index]
            count += i + 1 - progress_index
            progress_index = i + 1


    print ('count: ', count)

