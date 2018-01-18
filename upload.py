#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import judger_hand


import read_data
import predict_hands

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='full_yolo_hand_small.h5',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

def _main_(args):
 
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    print (weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    #pairs = read_data.getRealTestPairs()
    imgs = judger_hand.get_file_names()
    #imgs = ['data/DeepQ-Vivepaper/data/air/img/img_00000.png', 'data/DeepQ-Vivepaper/data/air/img/img_00001.png']
    pairs = [ [img, '']  for img in imgs ]
    hand_labels = predict_hands.predictHands_prod(pairs) # 0: left, 1: right, 2: both
    f = judger_hand.get_output_file_object()
    #f = open('test_output.txt', 'wb')
    for index, pair in enumerate(pairs):
        hand_label = hand_labels[index]
        image_path = pair[0]
        if image_path[-4:] == '.mp4':
            video_out = image_path[:-4] + '_detected' + image_path[-4:]

            video_reader = cv2.VideoCapture(image_path)

            nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

            video_writer = cv2.VideoWriter(video_out,
                                   cv2.VideoWriter_fourcc(*'MPEG'), 
                                   50.0, 
                                   (frame_w, frame_h))

            for i in tqdm(range(nb_frames)):
                _, image = video_reader.read()
                
                boxes = yolo.predict(image)
                image = draw_boxes(image, boxes, config['model']['labels'])

                video_writer.write(np.uint8(image))

            video_reader.release()
            video_writer.release()  
        else:
            #image = cv2.imread(image_path)
            #boxes = yolo.predict(image, hand_label)
            #image = draw_boxes(image, boxes, config['model']['labels'])

            image = cv2.imread(image_path)
            boxes = yolo.predict(image, hand_label)
            image = draw_boxes(image, boxes, config['model']['labels'])
            for box in boxes:
                x0 = int((box.x - box.w/3) * image.shape[1])
                x1 = int((box.x + box.w/3) * image.shape[1])
                y0 = int((box.y - box.h/3) * image.shape[0])
                y1 = int((box.y + box.h/3) * image.shape[0])
                x0 = np.clip(x0, 0, image.shape[1])
                x1 = np.clip(x1, 0, image.shape[1])
                y0 = np.clip(y0, 0, image.shape[0])
                y1 = np.clip(y1, 0, image.shape[0])
                #for box in boxes:
                #f.write(str.encode('%s %d %d %d %d %d %f\n' % (image_path, x0, y0, x1, y1, box.get_label(), box.get_score())))
                #f.write('%s %d %d %d %d %d %f\n' % (image_path, x0, y0, x1, y1, box.get_label(), box.get_score()))
                f.write((image_path + ' ' + str(x0) + ' ' + str(y0) + ' '+ str(x1) + ' ' + str(y1) + ' ' + str(box.get_label()) + ' ' +str(box.get_score()) +'\n' ).encode('ascii'))
    score, err = judger_hand.judge()
    print('score:', score)
    print('err:',err)
            #image_path = 'output3/' + image_path.split('/')[-3] + '_' + image_path.split('/')[-1]

            #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
    from keras import backend as K
    K.clear_session()
