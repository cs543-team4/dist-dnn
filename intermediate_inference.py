import logging
import zlib
from concurrent import futures

import grpc
import tensorflow as tf


# both roles of gRPC server and client
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


def recv_input_data():
    tensor = None
    return tensor


def send_output_data(data):
    pass


def compress(data):
    compressed_data = zlib.compress(data, 9)
    return compressed_data


def decompress(data):
    return zlib.decompress(data)


def process_data():
    compressed_input_data = recv_input_data()
    input_data = decompress(compressed_input_data)

    sub_model = tf.keras.Sequential()
    sub_model.load_weights('./checkpoints/split_model')

    send_output_data(compress(sub_model(input_data)))


if __name__ == '__main__':
    logging.basicConfig()
    serve()
