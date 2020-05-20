import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

import mnist


def create_model():
    model = tf.keras.Sequential()
    # 64개의 유닛을 가진 완전 연결 층을 모델에 추가합니다:
    model.add(Conv2D(32, 3, input_shape=(28, 28, 1), activation='relu'))
    # 또 하나를 추가합니다:
    model.add(Flatten())
    # 10개의 출력 유닛을 가진 소프트맥스 층을 추가합니다:
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
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
