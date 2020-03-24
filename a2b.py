#!/usr/bin/python3
# sjain106@asu.edu 12152185643

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model
import datetime


# Get file writers for publishing tensorboard logs and graphs
def get_tensorboard_log_writers():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    return train_summary_writer, test_summary_writer


# Convolution Neural Network Model class
class ConvNN(Model):

    def __init__(self):

        self.batch_size = 256
        self.learning_rate = 0.001
        self.classes = 10
        self.drop_prob = 0.20

        super(ConvNN, self).__init__()

        self.build_model()

        self.define_loss_optimizer_metrics()


    # Build model for CNN
    def build_model(self):

        self.conv1 = Conv2D(
            filters=32, kernel_size=5, strides=[1, 1], padding='SAME', activation='relu', name='conv1'
        )  # no of filters = 32, filter size = 3

        self.maxpool1 = MaxPool2D(
            pool_size=[1, 1], strides=[1, 1], padding='VALID', data_format=None, name='pool1'
        )

        self.conv2 = Conv2D(
            filters=64, kernel_size=5, strides=[1, 1], padding='SAME', activation='relu', name='conv2'
        )  # no of filters = 64, filter size = 3

        self.maxpool2 = MaxPool2D(
            pool_size=[2, 2], strides=[2, 2], padding='VALID', data_format=None, name='pool2'
        )

        self.flatten = Flatten()

        self.dense1 = Dense(256, activation='relu')

        self.dropout = Dropout(rate=self.drop_prob)

        self.dense2 = Dense(self.classes)

    # Run model build
    def call(self, input_data):
        x = self.conv1(input_data)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

    # Set loss function, optimizer function, metrics for calculating loss and accuracy
    def define_loss_optimizer_metrics(self):
        # Loss Function
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)  # loss function = sparsely categorical CrossEntropy

        # Optimizing function
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)  # optimizing function = AdamOptimizer

        # Train Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        # Test Metrics
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

    # Get MNIST data
    def get_data(self):
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # get decimal values in tensor
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Add a channels dimension
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(self.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch_size)

        return train_dataset, test_dataset

    # Predict -> Calculate Loss -> Optimize Loss on training data, Check accuracy
    @tf.function
    def train_step(self, images, labels):

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self(images, training=True)
            loss = self.loss_func(labels, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    # Predict -> Calculate test Loss -> Check accuracy
    @tf.function
    def test_step(self, images, labels):
        predictions = self(images, training=False)
        testing_loss = self.loss_func(labels, predictions)

        self.test_loss(testing_loss)
        self.test_accuracy(labels, predictions)

    # Train model
    def train(self, epochs=5):

        train_summary_writer, test_summary_writer = get_tensorboard_log_writers()

        train_dataset, test_dataset = self.get_data()


        for epoch in range(epochs):

            start_time = datetime.datetime.now()

            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            # Train Model
            for images, labels in train_dataset:
                self.train_step(images, labels)

            # Log Training Results
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)

            # Test Model
            for test_images, test_labels in test_dataset:
                self.test_step(test_images, test_labels)

            # Log Testing results
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)

            # Print Results
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result() * 100,
                                  self.test_loss.result(),
                                  self.test_accuracy.result() * 100))

            time_taken = datetime.datetime.now() - start_time

            print("Time taken for epoch " + str(epoch) + ": " + str(time_taken.seconds) + " secs")




# main program

if __name__ == '__main__':
    model = ConvNN()
    model.train()