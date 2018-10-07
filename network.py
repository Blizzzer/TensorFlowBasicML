import tensorflow as tf
from tensorflow import keras


class MyNetwork:
    def __init__(self):
        self.train_images = 0
        self.train_labels = 0
        self.test_images = 0
        self.test_labels = 0

        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def load_data(self):
        (self.train_images, self.train_labels), (
            self.test_images, self.test_labels) = keras.datasets.fashion_mnist.load_data()

        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def learn(self):
        self.model.fit(self.train_images, self.train_labels, epochs=5)

    def evaluete(self):
        return self.model.evaluate(self.test_images, self.test_labels)

    def predict(self):
        return self.model.predict(self.test_images)
