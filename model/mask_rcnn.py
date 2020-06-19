import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, backend
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, initializers, regularizers, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add

import os
import matplotlib.pyplot as plt
import numpy as np
import math
'''
def crop_and_resize(image, boxes, box_indices, crop_size, **kwargs):
    return tensorflow.cast(
        tensorflow.image.crop_and_resize(
            image=image,
            boxes=tensorflow.cast(boxes, tensorflow.float32),
            box_indices=box_indices,
            crop_size=crop_size,
            **kwargs
        ),
        image.dtype
    )

class RoiAlign(keras.layers.Layer):
    def __init__(self, crop_size=(14, 14), parallel_iterations=32, **kwargs):
        self.crop_size = crop_size
        self.parallel_iterations = parallel_iterations

        super(RoiAlign, self).__init__(**kwargs)

    def map_to_level(self, boxes, canonical_size=224, canonical_level=1, min_level=0, max_level=4):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        w = x2 - x1
        h = y2 - y1

        size = keras.backend.sqrt(w * h)

        levels = tf.floor(canonical_level + tf.log2(size / canonical_size + keras.backend.epsilon()))
        levels = keras.backend.clip(levels, min_level, max_level)

        return levels

    def call(self, inputs, **kwargs):
        image_shape = keras.backend.cast(inputs[0], keras.backend.floatx())
        boxes       = keras.backend.stop_gradient(inputs[1])
        scores      = keras.backend.stop_gradient(inputs[2])
        fpn         = [keras.backend.stop_gradient(i) for i in inputs[3:]]

        def _roi_align(args):
            boxes  = args[0]
            scores = args[1]
            fpn    = args[2]

            # compute from which level to get features from
            target_levels = self.map_to_level(boxes)

            # process each pyramid independently
            rois           = []
            ordered_indices = []
            for i in range(len(fpn)):
                # select the boxes and classification from this pyramid level
                indices = tf.where(keras.backend.equal(target_levels, i))
                ordered_indices.append(indices)

                level_boxes = tf.gather_nd(boxes, indices)
                fpn_shape   = keras.backend.cast(keras.backend.shape(fpn[i]), dtype=keras.backend.floatx())

                # convert to expected format for crop_and_resize
                x1 = level_boxes[:, 0]
                y1 = level_boxes[:, 1]
                x2 = level_boxes[:, 2]
                y2 = level_boxes[:, 3]
                level_boxes = keras.backend.stack([
                    (y1 / image_shape[1] * fpn_shape[0]) / (fpn_shape[0] - 1),
                    (x1 / image_shape[2] * fpn_shape[1]) / (fpn_shape[1] - 1),
                    (y2 / image_shape[1] * fpn_shape[0] - 1) / (fpn_shape[0] - 1),
                    (x2 / image_shape[2] * fpn_shape[1] - 1) / (fpn_shape[1] - 1),
                ], axis=1)

                # append the rois to the list of rois
                rois.append(crop_and_resize(
                    keras.backend.expand_dims(fpn[i], axis=0),
                    level_boxes,
                    tf.zeros((keras.backend.shape(level_boxes)[0],), dtype='int32'),  # TODO: Remove this workaround (https://github.com/tensorflow/tensorflow/issues/33787).
                    self.crop_size
                ))

            # concatenate rois to one blob
            rois = keras.backend.concatenate(rois, axis=0)

            # reorder rois back to original order
            indices = keras.backend.concatenate(ordered_indices, axis=0)
            rois    = tf.scatter_nd(indices, rois, keras.backend.cast(keras.backend.shape(rois), 'int64'))

            return rois

        roi_batch = tf.map_fn(
            _roi_align,
            elems=[boxes, scores, fpn],
            dtype=keras.backend.floatx(),
            parallel_iterations=self.parallel_iterations
        )

        return roi_batch

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], None, self.crop_size[0], self.crop_size[1], input_shape[3][-1])

    def get_config(self):
        config = super(RoiAlign, self).get_config()
        config.update({
            'crop_size' : self.crop_size,
        })

        return config
'''
model = Sequential()

# 0~6
#ROI Align
#model.add(RoiAlign())
model.add(Input(shape=(224, 224, 3), dtype='float32', name='input'))
# conv1_pad
model.add(ZeroPadding2D(padding=(3, 3)))
# conv1
model.add(Conv2D(64, (7, 7), strides=(2, 2), padding='valid'))
# bn_conv1
model.add(BatchNormalization())
# activation_1
model.add(Activation('relu'))
# pool1_pad
model.add(ZeroPadding2D(padding=(1, 1)))
# max_pooling
model.add(MaxPooling2D((3, 3), 2))

# 7~17
class resnet_block1(tf.keras.layers.Layer):
    def __init__(self):
        super(resnet_block1, self).__init__()
        self.res2a_branch2a = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')
        self.bn2a_branch2a = BatchNormalization()
        self.activation_2 = Activation('relu')
        self.res2a_branch2b = Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn2a_branch2b = BatchNormalization()
        self.activation_3 = Activation('relu')
        self.res2a_branch2c = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.res2a_branch1 = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn2a_branch2c = BatchNormalization()
        self.bn2a_branch1 = BatchNormalization()
        self.add_1 = Add()
    
    def call(self, inputs):
        x = res2a_branch2a(inputs)
        x = bn2a_branch2a(x)
        x = activation_2(x)
        x = res2a_branch2b(x)
        x = bn2a_branch2b(x)
        x = activation_3(x)
        x = res2a_branch2c(x)
        x = bn2a_branch2c(x)

        y = res2a_branch1(inputs)
        y = bn2a_branch1(y)

        z = add_1([x, y])
        return z

model.add(resnet_block1())
#18
# activation_4
model.add(Activation('relu'))

#19~27
class resnet_block2(tf.keras.layers.Layer):
    def __init__(self):
        super(resnet_block2, self).__init__()
        self.res2b_branch2a = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')
        self.bn2b_branch2a = BatchNormalization()
        self.activation_5 = Activation('relu')
        self.res2b_branch2b = Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn2b_branch2b = BatchNormalization()
        self.activation_6 = Activation('relu')
        self.res2b_branch2c = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn2b_branch2c = BatchNormalization()
        self.add_2 = Add()
    
    def call(self, inputs):
        x = res2b_branch2a(inputs)
        x = bn2b_branch2a(x)
        x = activation_5(x)
        x = res2b_branch2b(x)
        x = bn2b_branch2b(x)
        x = activation_6(x)
        x = res2b_branch2c(x)
        x = bn2b_branch2c(x)

        y = add_2([x, inputs])
        return y

model.add(resnet_block2())
#28
# activation_7
model.add(Activation('relu'))
#29~37
model.add(resnet_block2())
#38
# activation_10
model.add(Activation('relu'))

#39~49
class resnet_block3(tf.keras.layers.Layer):
    def __init__(self):
        super(resnet_block3, self).__init__()
        self.res3a_branch2a = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')
        self.bn3a_branch2a = BatchNormalization()
        self.activation_11 = Activation('relu')
        self.res3a_branch2b = Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.bn3a_branch2b = BatchNormalization()
        self.activation_12 = Activation('relu')
        self.res3a_branch2c = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.res3a_branch1 = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.bn3a_branch2c = BatchNormalization()
        self.bn3a_branch1 = BatchNormalization()
        self.add_4 = Add()
    
    def call(self, inputs):
        x = res3a_branch2a(inputs)
        x = bn3a_branch2a(x)
        x = activation_11(x)
        x = res3a_branch2b(x)
        x = bn3a_branch2b(x)
        x = activation_12(x)
        x = res3a_branch2c(x)
        x = bn3a_branch2c(x)

        y = res3a_branch1(inputs)
        y = bn3a_branch1(y)

        z = add_4([x, y])
        return z

model.add(resnet_block3())
#50
# activation_13
model.add(Activation('relu'))

#51~59
class resnet_block4(tf.keras.layers.Layer):
    def __init__(self):
        super(resnet_block4, self).__init__()
        self.res3b_branch2a = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')
        self.bn3b_branch2a = BatchNormalization()
        self.activation_14 = Activation('relu')
        self.res3b_branch2b = Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.bn3b_branch2b = BatchNormalization()
        self.activation_15 = Activation('relu')
        self.res3b_branch2c = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.bn3b_branch2c = BatchNormalization()
        self.add_5 = Add()
    
    def call(self, inputs):
        x = res3b_branch2a(inputs)
        x = bn3b_branch2a(x)
        x = activation_14(x)
        x = res3b_branch2b(x)
        x = bn3b_branch2b(x)
        x = activation_15(x)
        x = res3b_branch2c(x)
        x = bn3b_branch2c(x)

        y = add_5([x, inputs])
        return y

model.add(resnet_block4())
#60
# activation_16
model.add(Activation('relu'))
#61~69
model.add(resnet_block4())
#70
# activation_19
model.add(Activation('relu'))
#71~79
model.add(resnet_block4())
#80
# activation_22
model.add(Activation('relu'))

#81~91
class resnet_block5(tf.keras.layers.Layer):
    def __init__(self):
        super(resnet_block5, self).__init__()
        self.res4a_branch2a = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')
        self.bn4a_branch2a = BatchNormalization()
        self.activation_23 = Activation('relu')
        self.res4a_branch2b = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn4a_branch2b = BatchNormalization()
        self.activation_24 = Activation('relu')
        self.res4a_branch2c = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')
        self.res4a_branch1 = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')
        self.bn4a_branch2c = BatchNormalization()
        self.bn4a_branch1 = BatchNormalization()
        self.add_8 = Add()
    
    def call(self, inputs):
        x = res4a_branch2a(inputs)
        x = bn4a_branch2a(x)
        x = activation_23(x)
        x = res4a_branch2b(x)
        x = bn4a_branch2b(x)
        x = activation_24(x)
        x = res4a_branch2c(x)
        x = bn4a_branch2c(x)
        
        y = res4a_branch1(inputs)
        y = bn4a_branch1(y)

        z = add_8([x, y])
        return z

model.add(resnet_block5())
#92
# activation_25
model.add(Activation('relu'))

#93~101
class resnet_block6(tf.keras.layers.Layer):
    def __init__(self):
        super(resnet_block6, self).__init__()
        self.res4b_branch2a = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')
        self.bn4b_branch2a = BatchNormalization()
        self.activation_26 = Activation('relu')
        self.res4b_branch2b = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn4b_branch2b = BatchNormalization()
        self.activation_27 = Activation('relu')
        self.res4b_branch2c = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')
        self.bn4b_branch2c = BatchNormalization()
        self.add_9 = Add()
    
    def call(self, inputs):
        x = res4b_branch2a(inputs)
        x = bn4b_branch2a(x)
        x = activation_26(x)
        x = res4b_branch2b(x)
        x = bn4b_branch2b(x)
        x = activation_27(x)
        x = res4b_branch2c(x)
        x = bn4b_branch2c(x)

        y = add_9([x, inputs])
        return y

model.add(resnet_block6())
#102
# activation_28
model.add(Activation('relu'))
#103~111
model.add(resnet_block6())
#112
# activation_31
model.add(Activation('relu'))
#113~121
model.add(resnet_block6())
#122
# activation_34
model.add(Activation('relu'))
#123~131
model.add(resnet_block6())
#131
# activation_37
model.add(Activation('relu'))
#133~141
model.add(resnet_block6())
#142
# activation_40
model.add(Activation('relu'))

#143~154
class resnet_block7(tf.keras.layers.Layer):
    def __init__(self):
        super(resnet_block7, self).__init__()
        self.res5a_branch2a = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')
        self.bn5a_branch2a = BatchNormalization()
        self.activation_41 = Activation('relu')
        self.res5a_branch2b = Conv2D(512, (3, 3), strides=(1, 1), padding='same')
        self.bn5a_branch2b = BatchNormalization()
        self.activation_42 = Activation('relu')
        self.res5a_branch2c = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')
        self.res5a_branch1 = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')
        self.bn5a_branch2c = BatchNormalization()
        self.bn5a_branch1 = BatchNormalization()
        self.add_14 = Add()
    
    def call(self, inputs):
        x = res5a_branch2a(inputs)
        x = bn5a_branch2a(x)
        x = activation_41(x)
        x = res5a_branch2b(x)
        x = bn5a_branch2b(x)
        x = activation_42(x)
        x = res5a_branch2c(x)
        x = bn5a_branch2c(x)

        y = res5a_branch1(inputs)
        y = bn5a_branch1(y)

        z = add_14([x, y])
        return z

model.add(resnet_block7())
#154
# activation_43
model.add(Activation('relu'))

#155~163
class resnet_block8(tf.keras.layers.Layer):
    def __init__(self):
        super(resnet_block8, self).__init__()
        self.res5b_branch2a = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')
        self.bn5b_branch2a = BatchNormalization()
        self.activation_44 = Activation('relu')
        self.res5b_branch2b = Conv2D(512, (3, 3), strides=(1, 1), padding='same')
        self.bn5b_branch2b = BatchNormalization()
        self.activation_45 = Activation('relu')
        self.res5b_branch2c = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')
        self.bn5b_branch2c = BatchNormalization()
        self.add_15 = Add()
    
    def call(self, inputs):
        x = res5b_branch2a(inputs)
        x = bn5b_branch2a(x)
        x = activation_44(x)
        x = res5b_branch2b(x)
        x = bn5b_branch2b(x)
        x = activation_45(x)
        x = res5b_branch2c(x)
        x = bn5b_branch2c(x)

        y = add_15([x, inputs])
        return y

model.add(resnet_block8())
#164
# activation_46
model.add(Activation('relu'))
#165~173
model.add(resnet_block8())
#174~176
# activation_49
model.add(Activation('relu'))
#
model.add(GlobalAveragePooling2D())
model.add(Dense(2048, activation='softmax'))
model.compile()