import tensorflow as tf
import pandas as pd
import numpy as np
import skimage.io
import skimage.transform
import argparse
import time
import cv2
import sys
import re
import os

train_step = 50000
batch_size = 16

class SimGAN:

    def __init__(self):
        self.learning_rate = 0.001
        self.image_shape = [64, 64, 3]

        self.build_model()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        def conv2d_layer(input_tensor, filter_shape, strides, name, padding='SAME', activation=tf.nn.relu):
            with tf.variable_scope(name):
                filters = tf.get_variable('filters', filter_shape, dtype=tf.float32)
                bias = tf.get_variable('bias', [filter_shape[3]], dtype=tf.float32)
                conv = tf.nn.conv2d(input_tensor, filters, strides, padding)
                output = activation(tf.nn.bias_add(conv, bias))
                return output

        def dense_layer(input_tensor, input_dim, output_dim, name, activation=tf.nn.relu):
            with tf.variable_scope(name):
                W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32)
                b = tf.get_variable('bias', [output_dim], dtype=tf.float32)
                output = activation(tf.matmul(input_tensor, W))
                return output

        def max_pool_layer(input_tensor, ksize, strides, padding='SAME'):
            pool = tf.nn.max_pool(input_tensor, ksize, strides, padding)
            return pool

        def batch_norm(x, is_test, name, decay=0.9, eps=1e-5, stddev=0.02):
            normed = tf.contrib.layers.batch_norm(x,
                                                  decay=decay,
                                                  updates_collections=None,
                                                  epsilon=eps,
                                                  scale=True,
                                                  is_training=tf.logical_not(is_test),
                                                  scope=name)
            return normed

        def lrelu(x, alpha=0.2):
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

        def residual_block(x, name):
            with tf.variable_scope(name):
                conv = conv2d_layer(
                              input_tensor=x,
                              filter_shape=[7, 7, 64, 64],
                              strides=[1, 1, 1, 1],
                              activation=lrelu,
                              name='conv1')
                conv = batch_norm(conv, self.tf_is_test, 'bn_conv1')
                conv = conv2d_layer(
                              input_tensor=conv,
                              filter_shape=[7, 7, 64, 64],
                              strides=[1, 1, 1, 1],
                              activation=lrelu,
                              name='conv2')
                conv = batch_norm(conv, self.tf_is_test, 'bn_conv2')
            return tf.add(x, conv)


        ############################## Generator ###############################
        def refiner(images):
            with tf.variable_scope('refiner'):
                n_input = tf.shape(images)[0]

                conv1 = conv2d_layer(
                              input_tensor=images,
                              filter_shape=[7, 7, 3, 64],
                              strides=[1, 1, 1, 1],
                              activation=lrelu,
                              name='conv1')
                conv1 = batch_norm(conv1, self.tf_is_test, 'bn_conv1')

                res1 = residual_block(conv1, 'res1')
                res2 = residual_block(res1, 'res2')
                res3 = residual_block(res2, 'res3')
                res4 = residual_block(res3, 'res4')
                res5 = residual_block(res4, 'res5')

                refined = conv2d_layer(
                              input_tensor=res5,
                              filter_shape=[1, 1, 64, 3],
                              strides=[1, 1, 1, 1],
                              activation=tf.nn.tanh,
                              name='refined_images')
            return refined

        ############################## Discriminator ################################
        def discriminator(images, reuse):
            with tf.variable_scope('discriminator', reuse=reuse):
                conv1 = conv2d_layer(
                            input_tensor=images,
                            filter_shape=[7, 7, 3, 32],
                            strides=[1, 4, 4, 1],
                            activation=lrelu,
                            name='conv1')
                conv1 = batch_norm(conv1, self.tf_is_test, 'bn_conv1')
                conv2 = conv2d_layer(
                            input_tensor=conv1,
                            filter_shape=[5, 5, 32, 64],
                            strides=[1, 2, 2, 1],
                            activation=lrelu,
                            name='conv2')
                conv2 = batch_norm(conv2, self.tf_is_test, 'bn_conv2')
                pool1 = max_pool_layer(conv2, [1, 3, 3, 1], [1, 2, 2, 1])
                conv3 = conv2d_layer(
                            input_tensor=pool1,
                            filter_shape=[3, 3, 64, 128],
                            strides=[1, 2, 2, 1],
                            activation=lrelu,
                            name='conv3')
                conv3 = batch_norm(conv3, self.tf_is_test, 'bn_conv3')
                conv4 = conv2d_layer(
                            input_tensor=conv3,
                            filter_shape=[1, 1, 128, 256],
                            strides=[1, 1, 1, 1],
                            activation=lrelu,
                            name='conv4')
                conv4 = batch_norm(conv4, self.tf_is_test, 'bn_conv4')
                conv5 = conv2d_layer(
                            input_tensor=conv4,
                            filter_shape=[1, 1, 256, 128],
                            strides=[1, 1, 1, 1],
                            activation=lrelu,
                            name='conv5')
                conv5 = batch_norm(conv5, self.tf_is_test, 'bn_conv5')

                d = conv2d_layer(
                        input_tensor=conv5,
                        filter_shape=[1, 1, 128, 1],
                        strides=[1, 1, 1, 1],
                        activation=tf.nn.sigmoid,
                        name='d_output')
            return d   # [batch, 6, 8, 1]

        ########################### Build GAN Model #############################
        self.tf_is_test = tf.placeholder(tf.bool, name='test_mode')

        # Refiner
        self.tf_synthetic_images = tf.placeholder(tf.float32, [None, 192, 256, 3], name='synthetic_images')
        self.images_refined = refiner(self.tf_synthetic_images)

        d_synthetic = discriminator(self.images_refined, reuse=False)
        l_real = -tf.reduce_mean(tf.log(1 - d_synthetic + 1e-8)) 
        l_reg  = tf.reduce_mean(tf.abs(self.tf_synthetic_images - self.images_refined))

        self.loss_g = l_real + 1 * l_reg 

        # Discriminator
        self.tf_fake_images = tf.placeholder(tf.float32, [None, 192, 256, 3], name='fake_images')
        self.tf_real_images = tf.placeholder(tf.float32, [None, 192, 256, 3], name='real_images')

        d_fake = discriminator(self.tf_fake_images, reuse=True)
        d_real = discriminator(self.tf_real_images, reuse=True)

        self.loss_d = -tf.reduce_mean(tf.log(d_fake + 1e-8)) - tf.reduce_mean(tf.log(1 - d_real + 1e-8))

        # Optimizers
        vars_gen = tf.get_collection(tf.GraphKeys.VARIABLES, scope='refiner')
        vars_dis = tf.get_collection(tf.GraphKeys.VARIABLES, scope='discriminator')

        self.train_gen_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_g, var_list=vars_gen)
        self.train_dis_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_d, var_list=vars_dis)

    def train_refiner(self, fake_images):
        loss_g, _ = self.sess.run([self.loss_g, self.train_gen_op], 
                        feed_dict={self.tf_synthetic_images: fake_images,
                                   self.tf_is_test: False})
        return loss_g

    def train_discriminator(self, fake_images, real_images):
        loss_d, _ = self.sess.run([self.loss_d, self.train_dis_op], feed_dict={self.tf_real_images: real_images,
                                                                               self.tf_fake_images: fake_images,
                                                                               self.tf_is_test: False})
        return loss_d

    def refine_images(self, images):
        images_refined = self.sess.run(self.images_refined, feed_dict={self.tf_synthetic_images: images,
                                                                       self.tf_is_test: True})
        return images_refined

    def save_model(self, dir, global_step):
        self.saver.save(self.sess, dir, global_step=global_step)

    def load_model(self, model_path):
        self.saver.restore(self.sess, model_path)

class Dataset:

    def __init__(self, path='/run/media/snowman/Data/Users/Woody/Desktop/mlds_final/MLDS_final'):

        self.path = path
        self.batch_size = 128
        self.real_list = os.listdir(os.path.join(path, 'frame'))
        self.n_real = len(self.real_list)

        self.buffer_size = 2000
        self.buffer_index = 0
        self.buffer = np.zeros((self.buffer_size, 192, 256, 3), dtype=np.float16)
        self.full_buffer = False

    def sample_real(self, size):
        sample_ids = np.random.choice(self.n_real, size, replace=False)
        images = np.zeros((size, 192, 256, 3))
        for i, index in enumerate(sample_ids):
            img = cv2.imread(os.path.join(self.path, 'frame', self.real_list[index]))
            images[i] = cv2.resize(img, (256, 192))
        images /= 255
        return images

    def sample_fake(self, size):
        dir_id = 's0%02d' % (np.random.randint(10))
        image_list = os.listdir(os.path.join(self.path, 'synthesis', dir_id, 'img'))
        sample_ids = np.random.choice(len(image_list), size, replace=False)
        images = np.zeros((size, 192, 256, 3))
        for i, index in enumerate(sample_ids):
            img = cv2.imread(os.path.join(self.path, 'synthesis', dir_id, 'img', image_list[index]))
            images[i] = cv2.resize(img, (256, 192))
        images /= 255
        return images

    def sample_buffer(self, size):
        sample_ids = np.random.choice(self.buffer_size, size, replace=False)
        return self.buffer[sample_ids]

    def update_buffer(self, images):
        if self.full_buffer:
            replace_id = np.random.choice(self.buffer_size, len(images), replace=False)
            self.buffer[replace_id] = images
        elif self.buffer_index + len(images) > self.buffer_size:
            self.buffer[-len(images):] = images
            self.full_buffer = True
        else:
            self.buffer[self.buffer_index: self.buffer_index + len(images)] = images
            self.buffer_index += len(images)

    def save_buffer(self, path):
        np.save(path, self.buffer)

    def load_buffer(self, path):
        self.buffer = np.load(path)
        self.full_buffer = True
    '''
    def next_batch(self, batch_size, real=False):

        dir_id = 's0%02d' % (np.random.randint(10))
        image_list = os.listdir(os.path.join(self.path, 'synthesis', dir_id, 'img'))
        fake_images = np.zeros((batch_size, 192, 256, 3))
        
        if real and self.full_buffer:
            sample_ids = np.random.choice(len(image_list), batch_size / 2, replace=False)
            for i, index in enumerate(sample_ids):
                img = skimage.io.imread(os.path.join(self.path, 'synthesis', dir_id, 'img', image_list[index]))
                fake_images[i] = skimage.transform.resize(img, 192, 256)
            sample_ids = np.random.choice(buffer_size, batch_size / 2, replace=False)
            fake_images[(batch_size/2):] = self.buffer[sample_ids]
        else:
            sample_ids = np.random.choice(len(image_list), batch_size, replace=False)
            for i, index in enumerate(sample_ids):
                img = skimage.io.imread(os.path.join(self.path, 'synthesis', dir_id, 'img', image_list[index]))
                fake_images[i] = skimage.transform.resize(img, 192, 256)

        if real:
            sample_ids = np.random.choice(self.n_real, batch_size, replace=False)
            real_images = np.zeros((batch_size, 192, 256, 3))
            for i, index in enumerate(sample_ids):
                img = skimage.io.imread(os.path.join(self.path, 'frame', self.real_list[index]))
                real_images[i] = skimage.transform.resize(img, 192, 256)
            return fake_images, real_images
        else:
            return fake_images
    '''


        
def train(args):

    #data = Dataset('train_sentence.csv', 'faces')
    model = SimGAN()
    data = Dataset('data/')

    step_count = 0
    training_log = open('training_log_refine.csv', 'a')
    print('epoch,generator_loss,discriminator_loss', file=training_log)
    if args.resume:
        model.load_model('model_refine/model.ckpt-%d'%args.resume)
        data.load_buffer('model_refine/buffer.npy')
        step_count = args.resume

    while step_count < train_step:

        t = time.time()
        step_count += 1

        for k in range(2):
            fake_images_batch = data.sample_fake(batch_size)
            loss_g = model.train_refiner(fake_images_batch)

        for k in range(1):
            if data.full_buffer:
                new_images = data.sample_fake(int(batch_size / 2))
                new_images = model.refine_images(new_images)
                old_images = data.sample_buffer(int(batch_size / 2))
                fake_images_batch = np.concatenate((new_images, old_images), 0)
                data.update_buffer(new_images)
            else:
                fake_images = data.sample_fake(batch_size)
                fake_images_batch = model.refine_images(fake_images)
                data.update_buffer(fake_images_batch)

            real_images_batch = data.sample_real(batch_size)
            loss_d = model.train_discriminator(fake_images_batch, real_images_batch)

        # Progress messages
        text = "Step # %-4d    | Time used: %-4ds | Gen_loss: %-5.3f | Dis_loss: %-5.3f\n" % (
                    step_count, 
                    time.time() - t, 
                    loss_g,
                    loss_d)
        sys.stdout.write(text)
        sys.stdout.flush() 

        print("%d,%f,%f" % (step_count, loss_g, loss_d), file=training_log)

        if (step_count >= 1000) and ((step_count % 1000) == 0):
            model.save_model('model_refine/model.ckpt', global_step=step_count)
            data.save_buffer('model_refine/buffer.npy')
            images_test = data.sample_fake(5)
            images_refined = model.refine_images(images_test)
            for i, img in enumerate(images_test):
                cv2.imwrite(
                    'refined_images/stp%d_%d.jpg'%(step_count,i+1), 
                        (np.clip(np.concatenate((img, images_refined[i]), 1),0,1)*255).astype(np.uint8))

    training_log.close()

def test(args):

    model = SimGAN()

    model.load_model('model_refine/model.ckpt-1000')

    b_size = 32
    counter = 0

    for dir_id in range(5):
        data_dir = '../data/DeepQ-Synth-Hand-0%1d/data/s0%02d/'%(np.floor(dir_id/5)+1,dir_id)
        img_list = os.listdir(os.path.join(data_dir, 'img'))
        n_images = len(img_list)
        counter = 0
        print('\nRefining: %s' % data_dir)
        while counter < n_images:
            print('\r%d/%d'%(counter, n_images),end='')
            if counter + b_size > n_images:
                size = n_images - counter
            else:
                size = b_size
            images = np.zeros((size, 192, 256, 3))
            for i in range(size):
                img = cv2.imread(os.path.join(data_dir, 'img', img_list[counter + i]))
                images[i] = cv2.resize(img, (256, 192))
            images /= 255
            refined = model.refine_images(images)
            for i, rimg in enumerate(refined):
                rimg = cv2.resize(rimg, (612, 460))
                cv2.imwrite(os.path.join(data_dir, 'refine/refined_%s'%img_list[counter+i]), 
                    (np.clip(rimg, 0, 1)*255).astype(np.uint8))
            counter += size 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', help='test mode', action='store_true')
    parser.add_argument('-r', '--resume', help='resume training', type=int, default=0)
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=14)
    parser.add_argument('-f', '--text_file', help='text file', type=str)
    args = parser.parse_args()

    if args.test:
        test(args)
    else: 
        train(args)
