import numpy as np
import tensorflow as tf
import utils
import tensorflow.keras.datasets as datasets
from sklearn.preprocessing import MinMaxScaler

padding = "SAME"  # @param ['SAME', 'VALID' ]


class RCNN:
    def __init__(self, num_classes, num_layers, learning_rate, dropout_rate, leaky_relu_alpha, output_layer_1, output_layer_2):

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.leaky_relu_alpha = leaky_relu_alpha

        self.initializer = tf.initializers.glorot_uniform()
        self.optimizer = tf.optimizers.Adam(learning_rate)

        self.output_layer_1 = output_layer_1
        self.output_layer_2 = output_layer_2

    def conv2d(self, inputs, filters, stride_size):
        out = tf.nn.conv2d(inputs, filters, strides=[
                           1, stride_size, stride_size, 1], padding=padding)
        return tf.nn.leaky_relu(out, alpha=self.leaky_relu_alpha)

    def maxpool(self, inputs, pool_size, stride_size):
        return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], padding='SAME', strides=[1, stride_size, stride_size, 1])

    def loss(self, pred, target):
        return tf.losses.categorical_crossentropy(target, pred)

    def get_weight(self, shape, name):
        return tf.Variable(self.initializer(shape), name=name, trainable=True, dtype=tf.float32)

    # def train_step(self, model, inputs, outputs):
    #     with tf.GradientTape() as tape:
    #         current_loss = self.loss(model(inputs), outputs)
    #     grads = tape.gradient(current_loss, self.weights)
    #     self.optimizer.apply_gradients(zip(grads, self.weights))
    #     print(tf.reduce_mean(current_loss))

    # Implementation of the first architecture, rCNN. Composed  by two convolutions of 8x8, and a final of 1x1. With 2 poolings of 2x2 after
    # the first two convolutions
    def model_rcnn1(self, inpt, out):

        self.input = tf.Variable(initial_value=inpt, dtype=tf.float32, shape=[
            None, None,  None, 3 + self.num_classes])
        self.output = tf.Variable(out, tf.int32, [None, None, None])

        w_conv1 = tf.Variable(tf.random.truncated_normal(
            [8, 8, 3 + self.num_classes, self.output_layer_1], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=self.output_layer_1))

        w_conv2 = tf.Variable(tf.random.truncated_normal(
            [8, 8, self.output_layer_1, self.output_layer_2], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=self.output_layer_2))

        w_conv3 = tf.Variable(tf.random.truncated_normal(
            [1, 1, self.output_layer_2, self.num_classes], stddev=0.1))
        b_conv3 = tf.Variable(tf.constant(0.1, shape=self.num_classes))

        current_input = self.input
        current_output = self.output

        self.errors = []
        self.logits = []

        for n_layer in range(self.num_layers):

            current_output = tf.strided_slice(current_output, [0, 0, 0], [
                                              0, 0, 0], strides=[1, 4, 4], end_mask=7)
            # current_output = tf.strided_slice(current_output, [0, 0, 0], [0, 0, 0], strides=[1, 2, 2], end_mask=7)

            h_conv1 = self.conv2d(current_input, w_conv1, 1) + b_conv1
            h_pool1 = self.maxpool(h_conv1, 2, 2)

            tanh1 = tf.tan(h_pool1)

            h_conv2 = self.conv2d(tanh1, w_conv2, 1) + b_conv2
            h_pool2 = self.maxpool(h_conv2, 2, 2)
            tanh2 = tf.tanh(h_pool2)

            logits = self.conv2d(tanh2, w_conv3, 1) + b_conv3
            predictions = tf.nn.softmax(
                logits,
                axis=None,
                name=None)

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                current_output, logits)
            error_for_all_pixel = tf.reduce_mean(cross_entropy)
            error_for_image = tf.reduce_mean(error_for_all_pixel)

            self.logits.append(logits)
            self.errors.append(error_for_image)

            # extracts RGB channels from input image. Only keeps every other pixel, since convolution scales down the
            #  output. The shape of this should have the same height and width and the logits.
            rgb = tf.strided_slice(current_input, [0, 0, 0, 0], [
                                   0, 0, 0, 3], strides=[1, 4, 4, 1], end_mask=7)

            current_input = tf.concat(values=[rgb, predictions], axis=3)

        self.loss = tf.add_n(self.errors)
        print(self.loss)
        self.train_step = self.optimizer.minimize(self.loss, self.logits)

    def train(self, dataset, n_epochs):
        for e in range(n_epochs):
            for features in list(dataset)[0:10]:
                image, label = features['image'], features['label']
                self.train_step(self.model, image, tf.one_hot(label, depth=3))
