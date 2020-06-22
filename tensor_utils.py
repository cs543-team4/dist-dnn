import tensorflow as tf

def parse(message):
    encoded_tensor = tf.convert_to_tensor(message)
    return tf.io.parse_tensor(tf.io.decode_base64(encoded_tensor), tf.float32)


def serialize(tensor):
    serialized_string = tf.io.serialize_tensor(tensor)
    serialized_string = tf.io.encode_base64(serialized_string)
    return serialized_string.numpy()
