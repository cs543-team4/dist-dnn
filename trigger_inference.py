from __future__ import print_function

import grpc
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

import inference_service_pb2
import inference_service_pb2_grpc

BATCH_SIZE = 32

mnist = tf.keras.datasets.mnist

_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test[..., tf.newaxis]

test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)


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
        stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
        response = stub.ProcessTensor(
            inference_service_pb2.SerializedTensor(data=data))
    print("InferenceService client received: " + response.message)


def serialize(tensor):
    serialized_string = tf.io.serialize_tensor(tensor)
    serialized_string = tf.io.encode_base64(serialized_string)
    return serialized_string.numpy()


if __name__ == '__main__':
    sub_model = tf.keras.models.load_model('./split_models/model_0.h5')
    sub_model.summary()

    images, _ = list(test_ds)[0]

    request(serialize(sub_model(images)))
