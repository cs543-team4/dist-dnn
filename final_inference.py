import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D

import numpy as np

from concurrent import futures
import logging

import grpc
import tensor_pb2
import tensor_pb2_grpc

BATCH_SIZE = 32

mnist = tf.keras.datasets.mnist

_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test[..., tf.newaxis]

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# gRPC server
# TODO: extend to multiple devices

def parse(message):
    encoded_tensor = tf.convert_to_tensor(message)
    return tf.io.parse_tensor(tf.io.decode_base64(encoded_tensor), tf.float32)

def decompress(data)
    return zlib.decompress(data)

class Transmitter(tensor_pb2_grpc.TransmitterServicer):
    def send_tensor(self, request, context):
        parsed = decompress(parse(request.data))
        process_data(parsed)
        return tensor_pb2.Reply(message='Received Serialized Tensor')


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100), options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ('grpc.max_message_length', 50 * 1024 * 1024),
        ('grpc.max_metadata_size', 16 * 1024 * 1024)
    ])
    tensor_pb2_grpc.add_transmitter_servicer_to_server(Transmitter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


def process_data(input_data):
    submodel = tf.keras.Sequential()
    submodel.add(Flatten())
    submodel.add(Dense(128, activation='relu'))
    submodel.add(Dense(10, activation='softmax'))
    submodel.load_weights('./splitted_models/model_1')
    submodel.build((32, 26, 26, 32))
    submodel.summary()

    predictions = submodel(input_data)

    _, labels = list(test_ds)[0]

    t_loss = loss_object(labels, predictions)
    logging.info('predictions: {}'.format([np.argmax(p) for p in predictions]))
    logging.info('labels: {}'.format(labels))

    test_loss(t_loss)
    test_accuracy(labels, predictions)

    logging.info('test loss: {}, test accuracy: {}'.format(test_loss.result(), test_accuracy.result()))


if __name__ == '__main__':
    logging.basicConfig(filename='./rpc_server.log', level=logging.INFO)
    serve()
