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


for n in range(num_epochs):
    count = 0
    for image, labels, image_id in dataset:
        if(count == 0):
            print(labels.shape, image_id)
            h, w = labels.shape
            input_image = np.append(image, np.zeros(
                shape=[h, w, RCNN_model.num_classes], dtype=np.float32), axis=2)
            # feed_dict = {RCNN_model.input: [input_image], RCNN_model.output: [labels]}

            RCNN_model.model_rcnn1([input_image],[labels])
            count += 1
        else:
            break
