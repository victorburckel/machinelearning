# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 14:45:42 2017

@author: Victor
"""
import io
import bson
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage.data import imread
from skimage import color
from skimage import transform
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def read_train():
    """Reads the train file and yields the product_id, category_id and picture as a (180, 180, 3) ndarray"""
    with open('train.bson', 'rb') as file:
        for entry in bson.decode_file_iter(file):
            product_id = entry['_id']
            category_id = entry['category_id']
            for img in entry['imgs']:
                picture = imread(io.BytesIO(img['picture']))
                yield product_id, category_id, picture

def gray_scale(picture):
    """Gray-scales the picture"""
    return color.rgb2gray(picture)

def resize(picture, shape):
    """Resizes the picture to a shape"""
    return transform.resize(picture, shape, mode='reflect')

def display(picture):
    """Displays the picture"""
    plt.imshow(picture)

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip(*args)

def train():
    """Train function"""
    all_category_ids = pd.read_table('category_names.csv', encoding='utf-8', sep=',', usecols =['category_id'], header=0, dtype={'category_id': np.int32}).category_id

    label_encoder = LabelEncoder()
    enc = OneHotEncoder()
    enc.fit(np.asarray(label_encoder.fit_transform(all_category_ids)).reshape(-1, 1))
    
    seed = 128
    
    input_num_units = 45 * 45
    hidden_num_units = 2048
    output_num_units = len(all_category_ids)

    batch_size = 128

    learning_rate = 0.01

    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape = (batch_size, input_num_units))
        y = tf.placeholder(tf.float32, shape = (batch_size, output_num_units))

        weights = {
            'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
            'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
        }

        biases = {
            'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
            'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
        }

        hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
        hidden_layer = tf.nn.relu(hidden_layer)

        output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
    
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
    
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    transformer = lambda picture: gray_scale(resize(picture, (45, 45)))
    try:
        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()
            print('Initialized')

            start_time = time.time()
            for counter, chunk in enumerate(grouper(map(lambda entry: (entry[0], entry[1], transformer(entry[2])), read_train()), batch_size)):
                product_ids, category_ids, pictures = zip(*chunk)
                batch_x = np.asarray(pictures).reshape(batch_size, input_num_units)
                batch_y = enc.transform(np.asarray(label_encoder.transform(category_ids)).reshape(-1, 1)).toarray()

                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})

                if counter and counter % 100 == 0:
                    end_time = time.time()
                    print(f"Done {counter} blocks of size {batch_size} = {counter * batch_size} in {int(end_time - start_time)}s, avg={(end_time - start_time) / (100 * batch_size)}")
                    start_time = end_time
    except Exception as exc:
        print("Unexpected exception caught: " + str(exc))
    
def test():
    """Test function"""
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cdiscount kaggle')
    subparsers = parser.add_subparsers()
    subparsers.add_parser('train', help='train the classifier').set_defaults(func=train)
    subparsers.add_parser('test', help='test the classifier').set_defaults(func=test)
    
    try:
        args = parser.parse_args()
    except Exception as exc:
        print(str(exc))
        parser.print_help()
        
    args.func()
    
