import tensorflow as tf
import numpy as np

import random


"""
- Example of layers:
[<tensorflow.python.keras.layers.convolutional.Conv2D object at 0x15f0ce9d0>, <tensorflow.python.keras.layers.core.Flatten object at 0x15f0ced50>, <tensorflow.python.keras.layers.core.Dense object at 0x15f0ce690>, <tensorflow.python.keras.layers.core.Dense object at 0x15f0ed450>]

- Refer to tf.keras.Sequential: https://www.tensorflow.org/guide/keras/overview?hl=ko
"""


def construct_model_by_layers(layers):
    model = tf.keras.Sequential()
    for l in layers:
        model.add(l)

    model.build(input_shape=layers[0].input_shape)
    return model


def split_model(model):
    def _get_splittable_indices(model):
        return list(range(len(model.layers)))[1:]

    layers = model.layers
    splitted_models = []

    # For now, just choose random split point
    # i = random.choice(_get_splittable_indices(model))

    # Limitation of using checkpoints: should shift to *.pb format
    # For now, just fix the split point
    i = 1

    splitted_models.append(construct_model_by_layers(layers[:i]))
    splitted_models.append(construct_model_by_layers(layers[i:]))

    return splitted_models
