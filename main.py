import numpy as np
import tensorflow as tf
import utils
import tensorflow.keras.datasets as datasets
from sklearn.preprocessing import MinMaxScaler
import tensorflow_datasets as tfds


padding = "SAME"  # @param ['SAME', 'VALID' ]

batch_size = 32  # @param {type: "number"}
learning_rate = 0.001  # @param {type: "number"}


dataset_name = 'horses_or_humans'  # @param {type: "string"}

dataset = tfds.load(name=dataset_name, split=tfds.Split.TRAIN)
dataset = dataset.shuffle(1024).batch(batch_size)

leaky_relu_alpha = 0.2
dropout_rate = 0.3


def conv2d(inputs, filters, stride_size):
    out = tf.nn.conv2d(inputs, filters, strides=[
                       1, stride_size, stride_size, 1], padding=padding)
    return tf.nn.leaky_relu(out, alpha=leaky_relu_alpha)


def maxpool(inputs, pool_size, stride_size):
    return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], padding='VALID', strides=[1, stride_size, stride_size, 1])


def dense(inputs, weights):
    x = tf.nn.leaky_relu(tf.matmul(inputs, weights), alpha=leaky_relu_alpha)
    return tf.nn.dropout(x, rate=dropout_rate)


# 5) Initializing CNN weights
# We initialize the weights for our CNN. The shapes need to calculated but the `tf.nn.conv2d` expects the filters to have a shape of `[ kernel_size , kernel_size , in_dims , out_dims ]`.
# We use the `glorot_uniform` initializer for our weights.

output_classes = 3
initializer = tf.initializers.glorot_uniform()


def get_weight(shape, name):
    return tf.Variable(initializer(shape), name=name, trainable=True, dtype=tf.float32)


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
    [64, output_classes],
]

weights = []
for i in range(len(shapes)):
    weights.append(get_weight(shapes[i], 'weight{}'.format(i)))


def model(x):
    x = tf.cast(x, dtype=tf.float32)
    c1 = conv2d(x, weights[0], stride_size=1)
    c1 = conv2d(c1, weights[1], stride_size=1)
    p1 = maxpool(c1, pool_size=2, stride_size=2)

    c2 = conv2d(p1, weights[2], stride_size=1)
    c2 = conv2d(c2, weights[3], stride_size=1)
    p2 = maxpool(c2, pool_size=2, stride_size=2)

    c3 = conv2d(p2, weights[4], stride_size=1)
    c3 = conv2d(c3, weights[5], stride_size=1)
    p3 = maxpool(c3, pool_size=2, stride_size=2)

    c4 = conv2d(p3, weights[6], stride_size=1)
    c4 = conv2d(c4, weights[7], stride_size=1)
    p4 = maxpool(c4, pool_size=2, stride_size=2)

    c5 = conv2d(p4, weights[8], stride_size=1)
    c5 = conv2d(c5, weights[9], stride_size=1)
    p5 = maxpool(c5, pool_size=2, stride_size=2)

    c6 = conv2d(p5, weights[10], stride_size=1)
    c6 = conv2d(c6, weights[11], stride_size=1)
    p6 = maxpool(c6, pool_size=2, stride_size=2)

    flatten = tf.reshape(p6, shape=(tf.shape(p6)[0], -1))

    d1 = dense(flatten, weights[12])
    d2 = dense(d1, weights[13])
    d3 = dense(d2, weights[14])
    d4 = dense(d3, weights[15])
    d5 = dense(d4, weights[16])
    logits = tf.matmul(d5, weights[17])

    return tf.nn.softmax(logits)


def loss(pred, target):
    return tf.losses.categorical_crossentropy(target, pred)


optimizer = tf.optimizers.Adam(learning_rate)


def train_step(model, inputs, outputs):
    with tf.GradientTape() as tape:
        current_loss = loss(model(inputs), outputs)
    grads = tape.gradient(current_loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    print(tf.reduce_mean(current_loss))


num_epochs = 2  # @param {type: "number"}

for e in range(num_epochs):
    for features in dataset:
        image, label = features['image'], features['label']
        train_step(model, image, tf.one_hot(label, depth=3))

# def cnn_model_fn(features, labels, mode):
#     input_layer = tf.reshape(tensor=features["x"], shape=[-1, 28, 28, 1])
#     conv1 = tf.layers.conv2d(
#         inputs=input_layer,
#         filters=14,
#         kernel_size=[5, 5],
#         padding="same",
#         activation=tf.nn.relu)

#     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#     conv2 = tf.layers.conv2d(
#         inputs=pool1,
#         filters=36,
#         kernel_size=[5, 5],
#         padding="same",
#         activation=tf.nn.relu)
#     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#     pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 36])

#     dense = tf.layers.dense(inputs=pool2_flat, units=7 *
#                             7 * 36, activation=tf.nn.relu)
#     dropout = tf.layers.dropout(
#         inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
#     logits = tf.layers.dense(inputs=dropout, units=10)
#     # You can create a dictionary containing the classes and the probability of each class.
#     predictions = {
#         # Generate predictions
#         "classes": tf.argmax(input=logits, axis=1),
#         "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

#     # Calculate Loss
#     loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

#     # Configure the Training Op (for TRAIN mode)
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#         train_op = optimizer.minimize(
#             loss=loss,
#             global_step=tf.train.get_global_step())
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

#     # Add evaluation metrics Evaluation mode
#     eval_metric_ops = {
#         "accuracy": tf.metrics.accuracy(
#             labels=labels, predictions=predictions["classes"])}
#     return tf.estimator.EstimatorSpec(
#         mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
# batch_size = len(x_train)

# mnist_classifier = tf.estimator.Estimator(
#     model_fn=cnn_model_fn, model_dir="train/mnist_convnet_model")

# # Set up logging for predictions
# tensors_to_log = {"probabilities": "softmax_tensor"}
# logging_hook = tf.train.LoggingTensorHook(
#     tensors=tensors_to_log, every_n_iter=50)


# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": x_train},
#     y=y_train,
#     batch_size=100,
#     num_epochs=None,
#     shuffle=True)


# mnist_classifier.train(
#     input_fn=train_input_fn,
#     steps=1600,
#     hooks=[logging_hook])


# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": x_test},
#     y=y_test,
#     num_epochs=1,
#     shuffle=False)
# eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
# print(eval_results)
