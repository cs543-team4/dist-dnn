import tensorflow as tf

from concurrent import futures
import logging

import grpc
import helloworld_pb2
import helloworld_pb2_grpc

# both roles of gRPC server and client

class Greeter(helloworld_pb2_grpc.GreeterServicer):

    def SayHello(self, request, context):
        return helloworld_pb2.HelloReply(message='Hello, %s!' % request.name)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
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
