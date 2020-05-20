# Run as both RPC server + client

import logging
import argparse
from concurrent import futures

import grpc
import numpy as np
import tensorflow as tf

import inference_service_pb2
import inference_service_pb2_grpc

import mnist

"""
Test dataset for verification
"""


test_ds = mnist.get_test_ds()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


def serve(connected_servers, model_index):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100), options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ('grpc.max_message_length', 50 * 1024 * 1024),
        ('grpc.max_metadata_size', 16 * 1024 * 1024)
    ])
    inference_service_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceService(connected_servers, model_index), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


class InferenceService(inference_service_pb2_grpc.InferenceServiceServicer):
    def __init__(self, connected_servers, model_index):
        self.model = SubModel(model_index)
        self.connected_servers = connected_servers

    def _choose_next_server(self, connected_servers):
        # TODO: distribution algorithm - choose next inference server w.r.t. current network state and server availability
        return connected_servers[0]

    def ProcessTensor(self, request, context):
        parsed = parse(request.data)
        result = self.model.process_data(parsed)
        if len(self.connected_servers) == 0:
            accuracy = self.model.validate_predictions()
            return inference_service_pb2.Reply(message='Received serialized tensor (current accuracy {}%)'.format(accuracy * 100))
        else:
            next_server_address = self._choose_next_server(
                self.connected_servers)
            request(serialize(result), next_server_address)


def parse(message):
    encoded_tensor = tf.convert_to_tensor(message)
    return tf.io.parse_tensor(tf.io.decode_base64(encoded_tensor), tf.float32)


def serialize(tensor):
    serialized_string = tf.io.serialize_tensor(tensor)
    serialized_string = tf.io.encode_base64(serialized_string)
    return serialized_string.numpy()


class SubModel():
    def __init__(self, model_index=0):
        self.predictions = []
        self.pred_index = 0
        self.model_index = model_index

    def process_data(self, input_data):
        sub_model = tf.keras.models.load_model(
            './split_models/model_{}.h5'.format(self.model_index))
        input_shape = list(input_data.shape[1:])
        input_shape.insert(0, None)
        sub_model.build(input_shape=input_shape)
        sub_model.summary()

        result = sub_model(input_data)
        self.predictions.append(result)
        self.pred_index += 1

        return result

    def validate_predictions(self):
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')
        for i, (_, labels) in enumerate(test_ds):
            if (i >= self.pred_index):
                break

            t_loss = loss_object(labels, self.predictions[i])
            logging.info('predictions: {}'.format(
                [np.argmax(p) for p in self.predictions[i]]))
            logging.info('labels: {}'.format(labels))

            test_loss(t_loss)
            test_accuracy(labels, self.predictions)

        logging.info('test loss: {}, test accuracy: {}'.format(
            test_loss.result(), test_accuracy.result()))

        return test_accuracy.result()


def request(data, server_address='localhost'):
    with grpc.insecure_channel('{}:50051'.format(server_address), options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ('grpc.max_message_length', 50 * 1024 * 1024),
        ('grpc.max_metadata_size', 16 * 1024 * 1024)
    ]) as channel:
        stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
        response = stub.ProcessTensor(
            inference_service_pb2.SerializedTensor(data=data))
    print("Response from next inference server: ", response.message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Daemon')
    parser.add_argument('--model_index', type=int,
                        default=0, help='split model index')
    parser.add_argument('--connected_server', '-s', type=str, action='append')
    parser.add_argument('--device', action='store_true')
    parser.add_argument('--log_filepath', type=str,
                        default='./inference_result.log')

    args = parser.parse_args()
    if args.connected_server == None:
        args.connected_server = []

    print('Model Index: ', args.model_index)
    print('Connected Servers: ', args.connected_server)

    if args.device:
        sub_model = tf.keras.models.load_model('./split_models/model_0.h5')
        sub_model.summary()

        images, _ = list(test_ds)[0]

        request(serialize(sub_model(images)))

    logging.basicConfig(
        filename=args.log_filepath, level=logging.INFO)
    serve(args.connected_server, args.model_index)
