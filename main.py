import numpy as np
import utils
import model
import tensorflow as tf
import tensorflow_datasets as tfds

dataset = utils.read_dataset('iccv09Data')

learning_rate = 0.001  # @param {type: "number"}
leaky_relu_alpha = 0.2
dropout_rate = 0.3
output_classes = 3
convolution_output_1 = 25
convolution_output_2 = 50

RCNN_model = model.RCNN(output_classes, 2, learning_rate, dropout_rate,
                        leaky_relu_alpha, convolution_output_1, convolution_output_2)


for image, labels, image_id in dataset[0:20]:
    print(labels.shape, image_id)
