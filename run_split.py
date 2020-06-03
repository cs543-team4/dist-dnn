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


def get_best_split_point(server_times):
    number_of_layers = len(server_times[0][0])
    split_points = []
    for server_pipeline in server_times:
        number_of_servers = len(server_pipeline)
        time_per_layers = [[0] * number_of_servers for _ in range(number_of_layers)]
        for i, server_time in enumerate(server_pipeline):
            for j, time_per_layer in enumerate(server_time):
                time_per_layers[j][i] = time_per_layer

        sum_of_time = [[0] * number_of_servers for _ in range(number_of_layers)]
        points = [[-1] * number_of_servers for _ in range(number_of_layers)]

        for i in range(number_of_layers):
            for j in range(number_of_servers):
                min_time = time_per_layers[i][j] + sum_of_time[i-1][j]
                for k in range(0, j):
                    new_time = time_per_layers[i][j] + sum_of_time[i - 1][k]
                    if new_time < min_time:
                        points[i][j] = k
                        min_time = new_time
                sum_of_time[i][j] = min_time

        split_point_for_set = [[0, -1] for _ in range(number_of_servers)]

        j = number_of_servers - 1
        time = sum_of_time[number_of_layers - 1][j]
        for i in range(number_of_servers):
            if time > sum_of_time[number_of_layers - 1][i]:
                j = i
                time = sum_of_time[number_of_layers - 1][i]

        split_point_for_set[j][1] = number_of_layers - 1
        for i in range(number_of_layers - 1, -1, -1):
            if j == 0:
                break
            if points[i][j] != -1:
                split_point_for_set[j][0] = i
                j = points[i][j]
                split_point_for_set[j][1] = i - 1

        split_points.append(split_point_for_set)

    return split_points


def run_split():
    server_times = []
    # TODO: how can we serialize the data?
    for server_pipeline in connected_server:
        pipeline_times = []
        for server in server_pipeline:
            server_time = request_test(1, server)
            pipeline_times.append(server_time)
        server_times.append(pipeline_times)

    best_point = get_best_split_point(server_times)

    for pipeline, server_pipeline in enumerate(connected_server):
        for level, server in enumerate(server_pipeline):
            start, end = best_point[pipeline][level]
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
