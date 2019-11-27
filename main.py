import numpy as np
import utils
import model
import tensorflow as tf
import tensorflow_datasets as tfds


padding = "SAME"  # @param ['SAME', 'VALID' ]

batch_size = 32  # @param {type: "number"}
learning_rate = 0.001  # @param {type: "number"}


dataset_name = 'horses_or_humans'  # @param {type: "string"}

dataset = tfds.load(name=dataset_name, split=tfds.Split.TRAIN)
dataset = dataset.shuffle(1024).batch(batch_size)

leaky_relu_alpha = 0.2
dropout_rate = 0.3
output_classes = 3

RCNN_model = model.RCNN(output_classes,2,learning_rate,dropout_rate,leaky_relu_alpha)
RCNN_model.train(dataset,2)


