from train_and_save import create_model, checkpoint_path
from split import split_model

import tensorflow as tf

BATCH_SIZE = 32

mnist = tf.keras.datasets.mnist

_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test[..., tf.newaxis]

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def test_step(models, images, labels):
    intermediate_pred = images
    for model in models:
        # TODO: this step should be done in separate machines
        intermediate_pred = model(intermediate_pred)

    predictions = intermediate_pred

    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def run_split():
    model = create_model()
    model.load_weights(checkpoint_path)
    model.build_graph((32, 28, 28, 1))
    model.summary()

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    for images, labels in test_ds:
        s_models = split_model(model)
        for i in range(len(s_models)):
            s_models[i].save_weights('./splitted_models/model_{}'.format(i))

        test_step(split_model(model), images, labels)

    model.summary()
    print('test_loss: {}, test_accuracy: {}'.format(test_loss.result(), test_accuracy.result()))


if __name__ == '__main__':
    run_split()
