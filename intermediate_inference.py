import logging
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


def process_data():
    input_data = recv_input_data()

    submodel = tf.keras.Sequential()
    submodel.load_weights('./checkpoints/splitted_model')

    send_output_data(submodel(input_data))


if __name__ == '__main__':
    logging.basicConfig()
    serve()
