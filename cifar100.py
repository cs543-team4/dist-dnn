import tensorflow as tf

cifar100 = tf.keras.datasets.cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# new axis for channel dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


def get_test_ds(batch_size=32):
    return tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(batch_size)
