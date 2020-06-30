import grpc
import tensorflow as tf

from tensor_utils import serialize, parse

import argparse
import inference_service_pb2
import inference_service_pb2_grpc
import cifar100

NUM_OF_LAYERS = 12
DEFAULT_POINT = [3, 7]

test_ds = cifar100.get_test_ds()
sample_images, _ = list(test_ds)[0]

loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(
    name='test_accuracy')

connected_server = [('localhost', 50051), ('localhost', 50052), ('localhost', 50053)]

def get_best_split_point(server_times):
    number_of_layers = len(server_times[0][0])
    split_points = []
    for server_pipeline in server_times:
        number_of_servers = len(server_pipeline)
        time_per_layers = [[0] * number_of_servers for _ in range(number_of_layers)]
        print('length of server pipeline: ', len(server_pipeline))
        for i, server_time in enumerate(server_pipeline):
            print('length of measured time sent from server: ', len(server_time))
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


def run_split(best_point=None):
    # server_times = []
    # for server_pipeline in connected_server:
    #     pipeline_times = []
    #     for server, port in server_pipeline:
    #         server_time = request_test(serialize(tf.cast(sample_images, dtype=tf.float32, name=None)), server, port)
    #         while len(server_time) != NUM_OF_LAYERS:    # request again: # of layers are hardcoded
    #             server_time = request_test(serialize(tf.cast(sample_images, dtype=tf.float32, name=None)), server, port)
    #
    #         pipeline_times.append(server_time)
    #     server_times.append(pipeline_times)
    #     print(pipeline_times)
    #
    # best_point = get_best_split_point(server_times)

    if best_point is None:
        best_point = DEFAULT_POINT

    def parse_best_point(best_point):
        ans = [(0, best_point)]

        for i in range(1, best_point):
            ans.append((best_point[i - 1] + 1, best_point[i]))
        ans.append((best_point[-1] + 1, NUM_OF_LAYERS - 1))

        return ans

    best_point = parse_best_point(best_point)

    for level, (server, port) in enumerate(connected_server):
        start, end = best_point[level]
        request_split(start, end, server, port)
        print("Level [{}] Layers from {} to {}".format(level, start, end))
        print("Server Address: {}".format)

def request_test(data, server_address='localhost', port=50051):
    with grpc.insecure_channel('{}:{}'.format(server_address, port), options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ('grpc.max_message_length', 50 * 1024 * 1024),
        ('grpc.max_metadata_size', 16 * 1024 * 1024)
    ]) as channel:
        stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
        response = stub.test_process(inference_service_pb2.testData(data=data))
        return response.time


def request_split(start, end, server_address='localhost', port=50051):
    with grpc.insecure_channel('{}:{}'.format(server_address, port), options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ('grpc.max_message_length', 50 * 1024 * 1024),
        ('grpc.max_metadata_size', 16 * 1024 * 1024)
    ]) as channel:
        stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
        try:
            response = stub.split_model(inference_service_pb2.slicingData(start=start, end=end))
            message = response.message
        except Exception as e:
            print(e)
            print('Unknown Exception occured - split failed')
            message = None 

        return message


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get best split point')
    parser.add_argument('points', metavar='N', type=int, nargs='*',
                        help='list of the index of split point')

    args = parser.parse_args()
    run_split(args.points)
