import tensorflow.compat.v1 as tf
import tensorflow as tf2
# helps us to represent our data as lists easily and quickly
# import numpy as np
# framework for defining a neural network as a set of Sequential layers
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, BatchNormalization, Dropout
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

tf.disable_v2_behavior()

def get_logits(features, weights, biases):
# network's forward pass, multiply inputs with weight
    return tf.add(tf.matmul(features, weights), biases)

# model.compile(loss=loss, optimizer=optimizer, 
#               metrics=['acc'])

# Batch of input and target output (1x1 matrices)
x = tf.placeholder(tf.float32, shape=[None, 1, 28 * 28], name='input')
y = tf.placeholder(tf.float32, shape=[None, 1, 1], name='target')

n_inputs = 28 * 28
n_classes = 10

weights = tf.Variable(tf.random.normal([n_inputs, n_classes]), dtype='float32', name='weights')
biases = tf.Variable(tf.random.normal([n_classes]), dtype='float32', name='biases')

y_ = tf.identity(tf.layers.dense(x, 1), name='output')

# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, name='loss')
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# Trivial linear model
# y_ = tf.identity(tf.layers.dense(x, 1), name='output')

logits = get_logits(x, weights, biases)

# Optimize loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')

# outputs = model(x, name='output')

# digits = [tf.Variable(d) for d in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

# y_ = model(x)

# train_op = optimizer.minimize(lambda : loss(y_, y), var_list = [y, y_], name='train')

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# optimizer = tf.train.SGD(learning_rate=0.001)
train_op = optimizer.minimize(loss, name='train')

init = tf.global_variables_initializer()

# tf.train.Saver.__init__ adds operations to the graph to save
# and restore variables.
saver_def = tf.train.Saver().as_saver_def()

print('Run this operation to initialize variables     : ', init.name)
print('Run this operation for a train step            : ', train_op.name)
print('Feed this tensor to set the checkpoint filename: ', saver_def.filename_tensor_name)
print('Run this operation to save a checkpoint        : ', saver_def.save_tensor_name)
print('Run this operation to restore a checkpoint     : ', saver_def.restore_op_name)

# Write the graph out to a file.
with open('frozen_models/graph_v1.pb', 'wb') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())