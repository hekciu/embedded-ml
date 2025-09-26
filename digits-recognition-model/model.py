import tensorflow as tf
# helps us to represent our data as lists easily and quickly
import numpy as np
# framework for defining a neural network as a set of Sequential layers
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# The LOSS function measures the guessed answers against the known correct 
# answers and measures how well or how badly it did
# then uses the OPTIMIZER function to make another guess. Based on how the 
# loss function went, it will try to minimize the loss.

g = tf.Graph()
with g.as_default():
    model = tf.keras.models.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

    model.compile(optimizer='sgd', loss='mean_squared_error')

    train_op = optimizer.minimize(loss, name='train')

    init = tf.global_variables_initializer()


    ops = g.get_operations()

    for op in ops:
        print(op.name, op.type)

    # Save graph
    tf.io.write_graph(g.as_graph_def(), 'frozen_models', 'filename.pb', as_text=False)

exit()



full_model = tf.function(lambda x: model(x))

full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input"))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.outputs[0].shape, model.outputs[0].dtype, name="output"))
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="graph.pb",
                  as_text=False)
