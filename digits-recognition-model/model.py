import tensorflow.compat.v1 as tf
import tensorflow as tf2
# helps us to represent our data as lists easily and quickly
# import numpy as np
# framework for defining a neural network as a set of Sequential layers
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# The LOSS function measures the guessed answers against the known correct 
# answers and measures how well or how badly it did
# then uses the OPTIMIZER function to make another guess. Based on how the 
# loss function went, it will try to minimize the loss.




# g = tf.Graph()
# with g.as_default():
#     model = tf.keras.models.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#     model.compile(optimizer='sgd', loss='mean_squared_error')

#     # train_op
#     # train_op = optimizer.minimize(loss, name='train')

#     ops = g.get_operations()

#     for op in ops:
#         print(op.name, op.type)

#     # Save graph
#     tf.io.write_graph(g.as_graph_def(), 'frozen_models', 'filename.pb', as_text=False)

# exit()


# model = tf.keras.models.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# xs = tf.constant([[1.0]])
# ys = tf.constant([[3.0]])

# model.train_on_batch(xs, ys)

# full_model = tf.function(lambda x: model(x))

# full_model = full_model.get_concrete_function(
#     tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input")
#     )

# # Get frozen ConcreteFunction
# frozen_func = convert_variables_to_constants_v2(full_model)
# frozen_func.graph.as_graph_def()

# graph = model.train_function.get_concrete_function(iter([(xs, ys)])).graph

# layers = [op for op in frozen_func.graph.get_operations()]
# print("-" * 60)
# print("Frozen model layers: ")
# for layer in layers:
#     print(layer.name, layer.type)
    
# print("-" * 60)
# print("Frozen model inputs: ")
# print(frozen_func.inputs)
# print("Frozen model outputs: ")
# print(frozen_func.outputs)

# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                   logdir="./frozen_models",
#                   name="graph.pb",
#                   as_text=False)

tf.disable_v2_behavior()

# Batch of input and target output (1x1 matrices)
x = tf.placeholder(tf.float32, shape=[None, 1, 1], name='input')
y = tf.placeholder(tf.float32, shape=[None, 1, 1], name='target')

# Trivial linear model
y_ = tf.identity(tf.layers.dense(x, 1), name='output')

# Optimize loss
loss = tf.reduce_mean(tf.square(y_ - y), name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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