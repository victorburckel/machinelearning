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
import pickle
from random import shuffle
from skimage.data import imread
from skimage import color
from skimage import transform
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import struct
import os
from bson.errors import InvalidBSON

def generate_offsets(bson_file):
    """Generates an array with the offset for each document in the bson file"""
    offsets = []
    offset = 0
    with open(bson_file, 'rb') as f:
        while True:
            size_data = f.read(4)
            if len(size_data) == 0:
                break
            elif len(size_data) != 4:
                raise InvalidBSON("cut off in middle of objsize")
            obj_size = struct.Struct("<i").unpack(size_data)[0] - 4
            offset+= obj_size + 4
            offsets.append(offset)
            f.seek(obj_size, os.SEEK_CUR)
    return offsets

def maybe_pickle_offsets(pickle_file, bson_file, force=False):
    if os.path.exists(pickle_file) and not force:
        # You may override by setting force=True.
        print(f"{pickle_file} already present - Skipping pickling.")
    else:
        offsets = generate_offsets(bson_file)
        with open(pickle_file, 'wb') as f:
            pickle.dump(offsets, f, pickle.HIGHEST_PROTOCOL)

def count_train_documents():
    """Counts how many documents provided BSON file contains"""
    with open('train.bson', 'rb') as file:
        cnt = 0
        while True:
            # Read size of next object.
            size_data = file.read(4)
            if len(size_data) == 0:
                break  # Finished with file normaly.
            elif len(size_data) != 4:
                raise InvalidBSON("cut off in middle of objsize")
            obj_size = struct.Struct("<i").unpack(size_data)[0] - 4
            # Skip the next obj_size bytes
            file.seek(obj_size, os.SEEK_CUR)
            cnt += 1
        return cnt

def read_train():
    """Reads the train file and yields the product_id, category_id and picture as a (180, 180, 3) ndarray"""
    with open('train.bson', 'rb') as file:
        for entry in bson.decode_file_iter(file):
            product_id = entry['_id']
            category_id = entry['category_id']
            for img in entry['imgs']:
                picture = imread(io.BytesIO(img['picture']))
                yield product_id, category_id, picture

def decode_file_iter_random_access(file_obj, offsets, codec_options=bson.DEFAULT_CODEC_OPTIONS):
    for offset in offsets:
        file_obj.seek(offset, os.SEEK_SET)
        size_data = file_obj.read(4)
        obj_size = struct.Struct("<i").unpack(size_data)[0] - 4
        elements = size_data + file_obj.read(obj_size)
        yield  bson._bson_to_dict(elements, codec_options)

def read_train_random_access(offsets):
    with open('train.bson', 'rb') as file:
        for entry in decode_file_iter_random_access(file, offsets):
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
    maybe_pickle_offsets('train.offsets.pickle', 'train.bson')

    with open('train.offsets.pickle', 'rb') as f:
        offsets = pickle.load(f)

    shuffle(offsets)

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

        saver = tf.train.Saver(max_to_keep=5)

    try:
        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()
            transformer = lambda picture: gray_scale(resize(picture, (45, 45)))

            print('Initialized')

            start_time = time.time()
            for counter, chunk in enumerate(grouper(map(lambda entry: (entry[0], entry[1], transformer(entry[2])), read_train_random_access(offsets)), batch_size)):
                product_ids, category_ids, pictures = zip(*chunk)
                batch_x = np.asarray(pictures).reshape(batch_size, input_num_units)
                batch_y = enc.transform(np.asarray(label_encoder.transform(category_ids)).reshape(-1, 1)).toarray()

                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})

                if counter and counter % 100 == 0:
                    end_time = time.time()
                    print(f"Done {counter} blocks of size {batch_size} = {counter * batch_size} in {int(end_time - start_time)}s, avg={(end_time - start_time) / (100 * batch_size)}")
                    print(f"Cost at iteration {counter}: {c}")
                    start_time = end_time

                if counter and counter % 1000 == 0:
                    save_path = saver.save(sess, "./model.ckpt", global_step=counter)

            save_path = saver.save(sess, "./model.ckpt")
            print(f"Model saved in file: {save_path}")

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
    
