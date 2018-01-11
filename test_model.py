from keras.models import Sequential
from keras.layers import Dense, Activation
import argparse, os
import read_data
import preprocess_test

features = preprocess_test.getFeatures(read_data.getRealPairs())

print ('features: ', features[:10])

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--hidden_size', type=int, default=128, help = 'Hidden size of the layers')
parser.add_argument('-a', '--activation', choices=['selu', 'relu', 'softmax', 'sigmoid'], help = 'Activation function')
parser.add_argument('-l', '--layer_size', type=int, default=3, help = 'Num of layers of the model')

args = parser.parse_args()

model = Sequential()
model.add(Dense(args.hidden_size, activation=args.activation, kernel_initializer='normal', input_shape=(2048,)))
for i in range(args.layer_size - 1):
    model.add(Dense(args.hidden_size, activation=args.activation, kernel_initializer='normal', input_shape=(args.hidden_size,)))

model.add(Dense(3, activation='softmax'))


