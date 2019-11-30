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

RCNN_model = model.RCNN(output_classes,2,learning_rate,dropout_rate,leaky_relu_alpha)
# RCNN_model.train(dataset,2)


for image, labels, image_id  in dataset[0:20]:
    print(labels.shape,image_id)