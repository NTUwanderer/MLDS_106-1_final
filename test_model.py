import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64, help = 'Batch size of classifier')
parser.add_argument('-s', '--hidden_size', type=int, default=128, help = 'Hidden size of the layers')
parser.add_argument('-a', '--activation', default='selu', choices=['selu', 'relu', 'softmax', 'sigmoid'], help = 'Activation function')
parser.add_argument('-l', '--layer_size', type=int, default=3, help = 'Num of layers of the model')
parser.add_argument('-e', '--epochs', type=int, default=10, help = 'Num of epochs to train')
parser.add_argument('-p', '--split', type=float, default=0.1, help = 'Validation Split')

args = parser.parse_args()

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import accuracy_score
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

#features = preprocess_test.getFeatures(read_data.getRealPairs(), 16)
synth_labels = loadLabels(read_data.getSynthPairs())
real_labels  = loadLabels(read_data.getRealPairs())

synth_features = pickle.load(open('features_s.pkl' , 'rb'))
real_features  = pickle.load(open('features_r.pkl' , 'rb'))

order = np.random.shuffle(np.arange(synth_features.shape[0]))
synth_features = synth_features[order][0]
synth_labels = synth_labels[order][0]
order = np.random.shuffle(np.arange(real_features.shape[0]))
print ('shapes: ', real_features.shape, real_labels.shape)
real_features = real_features[order][0]
real_labels = real_labels[order][0]
print ('shapes: ', real_features.shape, real_labels.shape)

model = Sequential()
model.add(Dense(args.hidden_size, activation=args.activation, kernel_initializer='normal', input_shape=(2048,)))
for i in range(args.layer_size - 1):
    model.add(Dense(args.hidden_size, activation=args.activation, kernel_initializer='normal', input_shape=(args.hidden_size,)))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(synth_features, synth_labels, batch_size=args.batch_size, epochs=args.epochs, verbose=1, validation_split=0.1)
model.fit(real_features, real_labels, batch_size=args.batch_size, epochs=args.epochs, verbose=1, validation_split=args.split)
model.save('model/model.hdf5')

preds = model.predict(real_features)
print ('preds.shape: ', preds.shape)
onehot_preds = np.argmax(preds, axis=1)
full_preds = np.zeros(real_labels.shape)
full_preds[np.arange(len(full_preds)), onehot_preds] = 1

print ('acc: ', accuracy_score(real_labels, full_preds))

counts = [0, 0, 0]
for label in onehot_preds:
    counts[label] += 1
print ('pred counts: ', counts)

preds = model.predict(synth_features)
print ('preds.shape: ', preds.shape)
onehot_preds = np.argmax(preds, axis=1)

print ('synth acc: ', accuracy_score(np.argmax(synth_labels, axis=1), onehot_preds))
counts = [0, 0, 0]
for label in np.argmax(synth_labels, axis=1):
    counts[label] += 1
print ('synth counts: ', counts)

counts = [0, 0, 0]
for label in onehot_preds:
    counts[label] += 1
print ('synth pred counts: ', counts)

