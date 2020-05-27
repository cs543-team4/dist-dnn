import grpc
import tensorflow as tf

import inference_service_pb2
import inference_service_pb2_grpc
import mnist

test_ds = mnist.get_test_ds()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


# TODO: split algorithm based on the time
def get_best_split_point(server_times):
    return [[0, 1], [2, 3]]


def run_split():
    server_times = []
    # TODO: how can we serialize the data?
    for server in connected_server:
        server_time = request_test(1, server)
        server_times.append(server_time)

    best_point = get_best_split_point(server_times)

    for level, server in enumerate(connected_server):
        start, end = best_point[level]
        request_split(start, end, server)
        print("Level [{}] Layers from {} to {}".format(level, start, end))
        print("Server Address: {}".format)


def request_test(data, server_address='localhost'):
    with grpc.insecure_channel('{}:50051'.format(server_address), options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ('grpc.max_message_length', 50 * 1024 * 1024),
        ('grpc.max_metadata_size', 16 * 1024 * 1024)
    ]) as channel:
        stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
        response = stub.test_process(inference_service_pb2.testData(data=data))
        return response.time


def request_split(start, end, server_address='localhost'):
    with grpc.insecure_channel('{}:50051'.format(server_address), options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ('grpc.max_message_length', 50 * 1024 * 1024),
        ('grpc.max_metadata_size', 16 * 1024 * 1024)
    ]) as channel:
        stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
        response = stub.split_model(inference_service_pb2.slicingData(start=start, end=end))
        return response.message


if __name__ == '__main__':
    connected_server = []
    run_split()
