import tensorflow as tf

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


test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)


if __name__ == '__main__':
    sub_model1 = tf.keras.Sequential()
    sub_model1 = tf.keras.models.load_model('./split_models/model_0.h5')
    sub_model1.summary()

    sub_model2 = tf.keras.models.load_model('./split_models/model_1.h5')
    sub_model2.summary()

    models = [sub_model1, sub_model2]

    images, labels = list(test_ds)[0]

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

    for images, labels in test_ds:
        test_step(images, labels)

    print('test_loss: {}, test_accuracy: {}'.format(
        test_loss.result(), test_accuracy.result()))
