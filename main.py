import numpy as np
import utils
import model
import tensorflow as tf
import tensorflow_datasets as tfds

dataset = utils.read_dataset('iccv09Data')

num_epochs = 2
learning_rate = 0.001  # @param {type: "number"}
leaky_relu_alpha = 0.2
dropout_rate = 0.3
output_classes = 8
convolution_output_1 = 25
convolution_output_2 = 50

RCNN_model = model.RCNN(output_classes, 2, learning_rate, dropout_rate,
                        leaky_relu_alpha, convolution_output_1, convolution_output_2)

print(RCNN_model)


def loss(predictions, logits, output):
    errors = []
    out =tf.Variable(output, dtype=tf.int32, shape=[None, None, None])
    for index, prediction in enumerate(predictions):
        out = tf.strided_slice(out, [0, 0, 0], [
        0, 0, 0], strides=[1, 4, 4], end_mask=7)
        print(out.shape, logits[index].shape)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            out, logits[index])
        error_for_all_pixel = tf.reduce_mean(cross_entropy)
        error_for_image = tf.reduce_mean(error_for_all_pixel)
        errors.append(error_for_image)
    return tf.add_n(errors)


for n in range(num_epochs):
    count = 0
    for image, labels, image_id in dataset:
        if(count == 0):
            with tf.GradientTape() as tape:
                h, w = labels.shape
                input_image = np.append(image, np.zeros(
                    shape=[h, w, RCNN_model.num_classes], dtype=np.float32), axis=2)
                RCNN_model([input_image])
                current_loss = loss(RCNN_model.predictions,
                                    RCNN_model.logits, [labels])
            gradients = tape.gradient(loss, RCNN_model.trainable_variables)
            RCNN_model.optimizer.apply_gradients(
                zip(gradients, RCNN_model.trainable_variables))
            count += 1
        else:
            break
