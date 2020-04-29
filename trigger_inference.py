from __future__ import print_function

import zlib

import grpc
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

import tensor_pb2
import tensor_pb2_grpc

BATCH_SIZE = 32

mnist = tf.keras.datasets.mnist

_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test[..., tf.newaxis]

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)


# gRPC client (request next inference with intermediate result)

def request(data):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051', options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ('grpc.max_message_length', 50 * 1024 * 1024),
        ('grpc.max_metadata_size', 16 * 1024 * 1024)
    ]) as channel:
        stub = tensor_pb2_grpc.TransmitterStub(channel)
        response = stub.SendTensor(tensor_pb2.SerializedTensor(data=data))
    print("Transmitter client received: " + response.message)


def compress(data):
    compressed_data = zlib.compress(data, 9)
    return compressed_data


def serialize(tensor):
    serialized_string = tf.io.serialize_tensor(tensor)
    serialized_string = tf.io.encode_base64(serialized_string)
    return serialized_string.numpy()


if __name__ == '__main__':
    sub_model = tf.keras.Sequential()
    sub_model.add(Conv2D(32, 3, activation='relu'))
    sub_model.load_weights('./split_models/model_0')
    sub_model.build((32, 28, 28, 1))
    sub_model.summary()

    images, _ = list(test_ds)[0]

    request(compress(serialize(sub_model(images))))
