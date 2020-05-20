import tensorflow as tf

from split import split_model
from train_and_save import create_model

BATCH_SIZE = 32

mnist = tf.keras.datasets.mnist

_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test[..., tf.newaxis]

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


def run_split():
    model = tf.keras.models.load_model('full_model.h5')
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(BATCH_SIZE)

    models = [model]

    @tf.function
    def test_step(images, labels):
        intermediate_prediction = images
        for model in models:
            # TODO: this step should be done in separate machines
            intermediate_prediction = model(intermediate_prediction)

        predictions = intermediate_prediction

        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    models = split_model(model)
    for i in range(len(models)):
        models[i].summary()
        models[i].save('./split_models/model_{}.h5'.format(i))

    for images, labels in test_ds:
        test_step(images, labels)

    print('test_loss: {}, test_accuracy: {}'.format(
        test_loss.result(), test_accuracy.result()))


if __name__ == '__main__':
    run_split()
