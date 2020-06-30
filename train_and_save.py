import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPooling2D


def create_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, 3, input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(BatchNormalization())    
    model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512,activation="relu"))
        
    model.add(Dense(10,activation="softmax"))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ============================================= #


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# new axis for channel dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = create_model()

if __name__ == '__main__':
    model.fit(x_train, y_train, epochs=1,
              validation_data=(x_test, y_test))

    model.save('full_model.h5')
    model.summary()
