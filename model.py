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

        # shapes = [
        #     [3, 3, 3, 16],
        #     [3, 3, 16, 16],
        #     [3, 3, 16, 32],
        #     [3, 3, 32, 32],
        #     [3, 3, 32, 64],
        #     [3, 3, 64, 64],
        #     [3, 3, 64, 128],
        #     [3, 3, 128, 128],
        #     [3, 3, 128, 256],
        #     [3, 3, 256, 256],
        #     [3, 3, 256, 512],
        #     [3, 3, 512, 512],
        #     [8192, 3600],
        #     [3600, 2400],
        #     [2400, 1600],
        #     [1600, 800],
        #     [800, 64],
        #     [64, num_classes],
        # ]
        # weights = []
        # for i in range(len(shapes)):
        #     weights.append(self.get_weight(shapes[i], 'weight{}'.format(i)))
        # self.weights = weights

    def conv2d(self, inputs, filters, stride_size):
        out = tf.nn.conv2d(inputs, filters, strides=[
                           1, stride_size, stride_size, 1], padding=padding)
        return tf.nn.leaky_relu(out, alpha=self.leaky_relu_alpha)

    def maxpool(self, inputs, pool_size, stride_size):
        return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], padding='VALID', strides=[1, stride_size, stride_size, 1])

    def loss(self, pred, target):
        return tf.losses.categorical_crossentropy(target, pred)

    def get_weight(self, shape, name):
        return tf.Variable(self.initializer(shape), name=name, trainable=True, dtype=tf.float32)

    def train_step(self, model, inputs, outputs):
        with tf.GradientTape() as tape:
            current_loss = self.loss(model(inputs), outputs)
        grads = tape.gradient(current_loss, self.weights)
        self.optimizer.apply_gradients(zip(grads, self.weights))
        print(tf.reduce_mean(current_loss))

    def model(self, x):

        self.input = tf.placeholder(dtype=tf.float32, shape=[
                                    None, None, None, 3 + self.num_classes])
        self.output = tf.placeholder(tf.int32, [None, None, None])

        w_conv1 = tf.Variable(tf.truncated_normal(
            [8, 8, 3 + self.num_classes, self.output_layer_1], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=self.output_layer_1))

        w_conv2 = tf.Variable(tf.truncated_normal(
            [8, 8, self.output_layer_1, self.output_layer_2], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=self.output_layer_2))

        w_conv3 = tf.Variable(tf.truncated_normal(
            [1, 1, 3 + self.output_layer_2, self.num_classes], stddev=0.1))
        b_conv3 = tf.Variable(tf.constant(0.1, shape=self.num_classes))

        current_input = self.input
        current_output = self.output

        for n_layer in range(self.num_layers):
            h_conv1 = self.conv2d(current_input, self.w_conv1, 1) + b_conv1
            h_pool1 = self.maxpool(h_conv1, 2, 2)

        return []

    def train(self, dataset, n_epochs):
        for e in range(n_epochs):
            for features in list(dataset)[0:10]:
                image, label = features['image'], features['label']
                self.train_step(self.model, image, tf.one_hot(label, depth=3))
