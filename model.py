import numpy as np
import tensorflow as tf
import utils
import tensorflow.keras.datasets as datasets
from sklearn.preprocessing import MinMaxScaler

padding = "SAME"  # @param ['SAME', 'VALID' ]


class RCNN:
    def __init__(self, num_classes, num_layers, learning_rate,dropout_rate,leaky_relu_alpha):
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.leaky_relu_alpha = leaky_relu_alpha

        self.initializer = tf.initializers.glorot_uniform()
        self.optimizer = tf.optimizers.Adam(learning_rate)

        shapes = [
            [3, 3, 3, 16],
            [3, 3, 16, 16],
            [3, 3, 16, 32],
            [3, 3, 32, 32],
            [3, 3, 32, 64],
            [3, 3, 64, 64],
            [3, 3, 64, 128],
            [3, 3, 128, 128],
            [3, 3, 128, 256],
            [3, 3, 256, 256],
            [3, 3, 256, 512],
            [3, 3, 512, 512],
            [8192, 3600],
            [3600, 2400],
            [2400, 1600],
            [1600, 800],
            [800, 64],
            [64, num_classes],
        ]
        weights = []
        for i in range(len(shapes)):
            weights.append(self.get_weight(shapes[i], 'weight{}'.format(i)))
        self.weights = weights
        


    def conv2d(self, inputs, filters, stride_size):
        out = tf.nn.conv2d(inputs, filters, strides=[
            1, stride_size, stride_size, 1], padding=padding)
        return tf.nn.leaky_relu(out, alpha=self.leaky_relu_alpha)

    def maxpool(self, inputs, pool_size, stride_size):
        return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], padding='VALID', strides=[1, stride_size, stride_size, 1])

    def dense(self, inputs, weights):
        x = tf.nn.leaky_relu(tf.matmul(inputs, weights),
                             alpha=self.leaky_relu_alpha)
        return tf.nn.dropout(x, rate=self.dropout_rate)

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


        self.input = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3 + self.num_classes])
        self.output = tf.placeholder(tf.int32, [None, None, None])
        
        x = tf.cast(x, dtype=tf.float32)
        c1 = self.conv2d(x, self.weights[0], stride_size=1)
        c1 = self.conv2d(c1, self.weights[1], stride_size=1)
        p1 = self.maxpool(c1, pool_size=2, stride_size=2)

        c2 = self.conv2d(p1, self.weights[2], stride_size=1)
        c2 = self.conv2d(c2, self.weights[3], stride_size=1)
        p2 = self.maxpool(c2, pool_size=2, stride_size=2)

        c3 = self.conv2d(p2, self.weights[4], stride_size=1)
        c3 = self.conv2d(c3, self.weights[5], stride_size=1)
        p3 = self.maxpool(c3, pool_size=2, stride_size=2)

        c4 = self.conv2d(p3, self.weights[6], stride_size=1)
        c4 = self.conv2d(c4, self.weights[7], stride_size=1)
        p4 = self.maxpool(c4, pool_size=2, stride_size=2)

        c5 = self.conv2d(p4, self.weights[8], stride_size=1)
        c5 = self.conv2d(c5, self.weights[9], stride_size=1)
        p5 = self.maxpool(c5, pool_size=2, stride_size=2)

        c6 = self.conv2d(p5, self.weights[10], stride_size=1)
        c6 = self.conv2d(c6, self.weights[11], stride_size=1)
        p6 = self.maxpool(c6, pool_size=2, stride_size=2)

        flatten = tf.reshape(p6, shape=(tf.shape(p6)[0], -1))

        d1 = self.dense(flatten, self.weights[12])
        d2 = self.dense(d1, self.weights[13])
        d3 = self.dense(d2, self.weights[14])
        d4 = self.dense(d3, self.weights[15])
        d5 = self.dense(d4, self.weights[16])
        logits = tf.matmul(d5, self.weights[17])

        return tf.nn.softmax(logits)

    def train(self, dataset, n_epochs):
        for e in range(n_epochs):
            for features in list(dataset)[0:10]:
                image, label = features['image'], features['label']
                self.train_step(self.model, image, tf.one_hot(label, depth=3))