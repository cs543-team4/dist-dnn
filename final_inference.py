import logging
from concurrent import futures

import grpc
import numpy as np
import tensorflow as tf

import inference_service_pb2
import inference_service_pb2_grpc

mnist = tf.keras.datasets.mnist

_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test[..., tf.newaxis]

test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(32)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


# gRPC server
# TODO: extend to multiple devices

def parse(message):
    encoded_tensor = tf.convert_to_tensor(message)
    return tf.io.parse_tensor(tf.io.decode_base64(encoded_tensor), tf.float32)


class InferenceService(inference_service_pb2_grpc.InferenceServiceServicer):
    def ProcessTensor(self, request, context):
        parsed = parse(request.data)
        process_data(parsed)
        return inference_service_pb2.Reply(message='Received Serialized Tensor')


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100), options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ('grpc.max_message_length', 50 * 1024 * 1024),
        ('grpc.max_metadata_size', 16 * 1024 * 1024)
    ])
    inference_service_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


def process_data(input_data):
    sub_model = tf.keras.models.load_model('./split_models/model_1.h5')
    input_shape = list(input_data.shape[1:])
    input_shape.insert(0, None)
    sub_model.build(input_shape=input_shape)

    predictions = sub_model(input_data)

    sub_model.summary()

    _, labels = list(test_ds)[0]

    t_loss = loss_object(labels, predictions)
    logging.info('predictions: {}'.format([np.argmax(p) for p in predictions]))
    logging.info('labels: {}'.format(labels))

    test_loss(t_loss)
    test_accuracy(labels, predictions)

    logging.info('test loss: {}, test accuracy: {}'.format(
        test_loss.result(), test_accuracy.result()))


if __name__ == '__main__':
    logging.basicConfig(filename='./rpc_server.log', level=logging.INFO)
    serve()
