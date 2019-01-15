#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers as layers_lib
import os
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import slim
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
RESNET_CLASSES = 33
pickup = "ABCDEFGHIJKLMNOPQRSTUVWXYZ3456789"
identity = np.identity(RESNET_CLASSES)
model_path = "/Users/geu/modelresnet/resnetv2.ckpt"
model_path_path = '/Users/geu/modelresnet/'


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    image_resized = tf.image.rgb_to_grayscale(image_resized)

    return image_resized


def get_train_dataset(img_dir):
    files = os.listdir(img_dir)
    images = []
    labels = []
    for label in files:
        for jpgfile in os.listdir(img_dir + "/" + label):
            if jpgfile.split(".")[1] != "jpeg":
                print(jpgfile)
                exit(0)
            images.append(os.path.join(img_dir + "/" + label + "/", jpgfile))
            text = os.path.splitext(jpgfile)[0]
            labels.append(text)
    return images, labels


def cnn_model_fn():
    x = tf.placeholder(tf.string, name='x')

    input_layer = _parse_function(x)
    input_layer = tf.reshape(input_layer, [-1, 224, 224, 1])
    net, _ = resnet_v2.resnet_v2_50(input_layer)

    y1 = tf.placeholder(tf.float32, shape=[None, RESNET_CLASSES])
    y2 = tf.placeholder(tf.float32, shape=[None, RESNET_CLASSES])
    y3 = tf.placeholder(tf.float32, shape=[None, RESNET_CLASSES])
    y4 = tf.placeholder(tf.float32, shape=[None, RESNET_CLASSES])
    y5 = tf.placeholder(tf.float32, shape=[None, RESNET_CLASSES])
    y6 = tf.placeholder(tf.float32, shape=[None, RESNET_CLASSES])

    net = tf.squeeze(net, axis=[1, 2])
    letter1 = slim.fully_connected(net, num_outputs=33, activation_fn=None, scope='train')
    letter2 = slim.fully_connected(net, num_outputs=33, activation_fn=None, scope='train1')
    letter3 = slim.fully_connected(net, num_outputs=33, activation_fn=None, scope='train2')
    letter4 = slim.fully_connected(net, num_outputs=33, activation_fn=None, scope='train3')
    letter5 = slim.fully_connected(net, num_outputs=33, activation_fn=None, scope='train4')
    letter6 = slim.fully_connected(net, num_outputs=33, activation_fn=None, scope='train5')

    letter1_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y1, logits=letter1))
    letter2_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y2, logits=letter2))
    letter3_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y3, logits=letter3))
    letter4_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y4, logits=letter4))
    letter5_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y5, logits=letter5))
    letter6_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y6, logits=letter6))
    loss = letter1_cross_entropy + letter2_cross_entropy + letter3_cross_entropy + letter4_cross_entropy + letter5_cross_entropy + letter6_cross_entropy

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = slim.learning.create_train_op(loss, optimizer,
                                             summarize_gradients=True)

    predict_concat = tf.stack([tf.argmax(letter1, 1),
                               tf.argmax(letter2, 1),
                               tf.argmax(letter3, 1),
                               tf.argmax(letter4, 1),
                               tf.argmax(letter5, 1),
                               tf.argmax(letter6, 1)],
                              1)
    y_concat = tf.stack([tf.argmax(y1, 1),
                         tf.argmax(y2, 1),
                         tf.argmax(y3, 1),
                         tf.argmax(y4, 1),
                         tf.argmax(y5, 1),
                         tf.argmax(y6, 1)],
                        1)

    accuracy_internal = tf.cast(tf.equal(predict_concat, y_concat), tf.float32),
    accuracy = tf.reduce_mean(tf.reduce_min(accuracy_internal, 2))
    accuracy_letter = tf.reduce_mean(tf.reshape(accuracy_internal, [-1]))

    length = tf.constant([1, 1, 1, 1, 1, 1], tf.int64)
    predicts = tf.map_fn(lambda pos: tf.substr(pickup, pos, length), predict_concat, tf.string)
    predict_join = tf.reduce_join(predicts, axis=1)
    # tf.print(predict_join)

    initer = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(initer)
    ckpt = tf.train.get_checkpoint_state(model_path_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    # saver.restore(sess, model_path)
    # """
    image = '/Users/geu/testimg/IDO4N4.jpeg'
    label = 'IDO4N4'
    batch_y_1 = [identity[pickup.find(label[0])]]
    batch_y_2 = [identity[pickup.find(label[1])]]
    batch_y_3 = [identity[pickup.find(label[2])]]
    batch_y_4 = [identity[pickup.find(label[3])]]
    batch_y_5 = [identity[pickup.find(label[4])]]
    batch_y_6 = [identity[pickup.find(label[5])]]

    accuracy_letter_, accuracy_, predict_ = sess.run([accuracy_letter, accuracy, predict_join],
                                                     feed_dict={x: image, y1: batch_y_1, y2: batch_y_2,
                                                                y3: batch_y_3,
                                                                y4: batch_y_4, y5: batch_y_5, y6: batch_y_6})
    print(accuracy_letter_)
    print("accuracy is ====>%f" % accuracy_)
    print(predict_)


dataset_path = '/Users/zkp/Desktop/testimg'
import time


def main(unused_argv):
    # Load training and eval data

    start = time.mktime(time.localtime())
    cnn_model_fn()
    print(time.mktime(time.localtime()) - start)


if __name__ == "__main__":
    # tf.app.run()
    main(None)
