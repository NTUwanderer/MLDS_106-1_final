import argparse, os
"""
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64, help = 'Batch size of classifier')
parser.add_argument('-s', '--hidden_size', type=int, default=128, help = 'Hidden size of the layers')
parser.add_argument('-a', '--activation', default='selu', choices=['selu', 'relu', 'softmax', 'sigmoid'], help = 'Activation function')
parser.add_argument('-l', '--layer_size', type=int, default=3, help = 'Num of layers of the model')
parser.add_argument('-e', '--epochs', type=int, default=10, help = 'Num of epochs to train')
parser.add_argument('-p', '--split', type=float, default=0.1, help = 'Validation Split')

args = parser.parse_args()
"""
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
import numpy as np
import read_data
import preprocess_test
import pickle
import progressbar
import json

def loadLabels(pairs):
    labels = np.zeros((len(pairs), 3), dtype=int)
    bar = progressbar.ProgressBar()
    for i in bar(range(len(pairs))):
        label = 0
        json_data = json.load(open(pairs[i][1]))
        if 'L' in json_data['bbox']:
            label += 1
        if 'R' in json_data['bbox']:
            label += 2

        labels[i][label - 1] = 1

    return labels

def predictHands(pairs, filename):

    #pairs = read_data.getRealTestPairs()
    if os.path.exists(filename):
        real_features = pickle.load(open(filename, 'rb'))
    else:
        real_features = preprocess_test.getFeatures(pairs, 16)
        pickle.dump(real_features, open(filename, 'wb'))
    
    order = np.random.shuffle(np.arange(real_features.shape[0]))
    real_features = real_features[order][0]
    
    model = load_model('model/model.hdf5')
    
    preds = model.predict(real_features)
    onehot_preds = np.argmax(preds, axis=1)

    del real_features
    del preds
    del model

    return onehot_preds

def predictHands_prod(pairs):

    real_features = preprocess_test.getFeatures(pairs, 16)
    
    order = np.random.shuffle(np.arange(real_features.shape[0]))
    real_features = real_features[order][0]
    
    model = load_model('model/model.hdf5')
    
    preds = model.predict(real_features)
    onehot_preds = np.argmax(preds, axis=1)

    del real_features
    del preds
    del model

    return onehot_preds

