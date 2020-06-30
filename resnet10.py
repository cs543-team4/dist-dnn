import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, Add, Input, Lambda, Dense, Layer
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend
from tensorflow.python.keras  import initializers

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
L2_WEIGHT_DECAY = 2e-4

class ConversionLayer(Layer):
    def __init__(self, **kwargs):
        super(ConversionLayer, self).__init__(**kwargs)
        self.layer1 = ZeroPadding2D(padding=(1, 1))
        self.layer2 = Conv2D(16, (3, 3),
                                strides=(1, 1),
                                padding='valid', use_bias=False,
                                kernel_initializer='he_normal')
        self.layer3 = BatchNormalization()
        self.layer4 = Activation('relu')
    def call(self, input_tensor):
        x = self.layer1(input_tensor)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def get_config(self):
        config = super(ConversionLayer, self).get_config()
        return config

class IdentityBuildingBlock(Layer):
        def __init__(self, kernel_size, filters, stage, strides, block, training=None, **kwargs):
            super(IdentityBuildingBlock, self).__init__(**kwargs)
            self._stage = stage
            self._block = block
            self.filters1, self.filters2 = filters
            self._kernel_size = kernel_size
            self._filters = filters
            self._strides = strides
            self._training = training
            self.layer1 = Conv2D(self.filters1, self._kernel_size,
                                padding='same', use_bias=False,
                                kernel_initializer='he_normal')
            self.layer2 = BatchNormalization()
            self.layer3 = Activation('relu')

            self.layer4 = Conv2D(self.filters2, self._kernel_size,
                                padding='same', use_bias=False,
                                kernel_initializer='he_normal')
            self.layer5 = BatchNormalization()

            self.layer6 = Add()
            self.layer7 = Activation('relu')

        def call(self, input_tensor):
            x = self.layer1(input_tensor)
            x = self.layer2(x, training=self._training)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x, training=self._training)
            x = self.layer6([x, input_tensor])
            x = self.layer7(x)
            return x
        
        def get_config(self):
            config = super(IdentityBuildingBlock, self).get_config()
            config.update({"kernel_size": self._kernel_size})
            config.update({"filters": [self.filters1, self.filters2]})
            config.update({"stage": self._stage})
            config.update({"strides": self._strides})
            config.update({"block": self._block})
            config.update({"training": self._training})
            return config
    
class ConvBuildingBlock(Layer):
    def __init__(self, kernel_size, filters, stage, block, strides=(2, 2), training=None, **kwargs):
        super(ConvBuildingBlock, self).__init__(**kwargs)
        self.filters1, self.filters2 = filters
        self._stage = stage
        self._block = block
        self._kernel_size = kernel_size
        self._strides = strides
        self._training = training
        self.layer1 = Conv2D(self.filters1, self._kernel_size,
                                strides=self._strides,
                                padding='same', use_bias=False,
                                kernel_initializer='he_normal')
        self.layer2 = BatchNormalization()
        self.layer3 = Activation('relu')

        self.layer4 = Conv2D(self.filters2, self._kernel_size,
                                padding='same', use_bias=False,
                                kernel_initializer='he_normal')
        self.layer5 = BatchNormalization()

        self.layer6 = Conv2D(self.filters2, (1, 1), strides=self._strides,
                                    use_bias=False,
                                    kernel_initializer='he_normal')
        self.layer7 = BatchNormalization()

        self.layer8 = Add()
        self.layer9 = Activation('relu')

    def call(self, input_tensor):
        x = self.layer1(input_tensor)
        x = self.layer2(x, training=self._training)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x, training=self._training)
        shortcut = self.layer6(input_tensor)
        shortcut = self.layer7(shortcut, training=self._training)
        x = self.layer8([x, shortcut])
        x = self.layer9(x)
        return x
    
    def get_config(self):
            config = super(ConvBuildingBlock, self).get_config()
            config.update({"kernel_size": self._kernel_size})
            config.update({"filters": [self.filters1, self.filters2]})
            config.update({"stage": self._stage})
            config.update({"block": self._block})
            config.update({"strides": self._strides})
            config.update({"training": self._training})
            return config

def create_model():
    model = tf.keras.Sequential()

    model.add(Input(shape=(32, 32, 3)))
            
    model.add(ConversionLayer())

    model.add(ConvBuildingBlock(3, [16, 16], 2, 'block_0', (1, 1), True))
    for i in range(2):
        model.add(IdentityBuildingBlock(3, [16, 16], 2, (1, 1), 'block_%d' % (i + 1), True))
    
    model.add(ConvBuildingBlock(3, [32, 32], 3, 'block_0', (2, 2), True))
    for i in range(2):
        model.add(IdentityBuildingBlock(3, [32, 32], 3, (2, 2), 'block_%d' % (i + 1), True))
    
    model.add(ConvBuildingBlock(3, [64, 64], 4, 'block_0', (2, 2), True))
    for i in range(2):
        model.add(IdentityBuildingBlock(3, [64, 64], 4, (2, 2), 'block_%d' % (i + 1), True))
    
    model.add(Lambda(lambda x: backend.mean(x, [1, 2])))
    model.add(Dense(100,
                   activation='softmax',
                   kernel_initializer=initializers.RandomNormal(stddev=0.01)))
    
    model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['categorical_accuracy'])

    return model

def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.
  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).
  This method converts the label to one hot to fit the loss function.
  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: Data type to use for input images.
  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.io.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(tf.float32))

  image = tf.cast(image, dtype)

  # TODO(haoyuzhang,hongkuny,tobyboyd): Remove or replace the use of V1 API
  label = tf.sparse.to_dense(label)
  return image, label



cifar100 = tf.keras.datasets.cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# new axis for channel dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
y_train = tf.one_hot(tf.reshape(y_train, [50000]), 100)
y_test = tf.one_hot(tf.reshape(y_test, [10000]), 100)

model = create_model()

if __name__ == '__main__':
    model.fit(x_train, y_train, epochs=90,
              validation_data=(x_test, y_test))

    model.save('full_model.h5')

    custom_objects = {"ConversionLayer": ConversionLayer,
        "IdentityBuildingBlock": IdentityBuildingBlock,
        "ConvBuildingBlock": ConvBuildingBlock }

    model.summary()
