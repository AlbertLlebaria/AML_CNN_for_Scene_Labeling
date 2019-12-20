import numpy as np
import utils
import model
import tensorflow as tf
import tensorflow_datasets as tfds
import plac
import time
import os
import math

learning_rate = 0.001  # @param {type: "number"}
leaky_relu_alpha = 0.2
dropout_rate = 0.3
output_classes = 34
convolution_output_1 = 25
convolution_output_2 = 50


def loss(predictions, logits, output, stride_size):
    errors = []
    out = tf.Variable(output, dtype=tf.int32, shape=[None, None, None])
    for index, prediction in enumerate(predictions):
        out = tf.strided_slice(out, [0, 0, 0], [
            0, 0, 0], strides=[1, stride_size, stride_size], end_mask=7)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            out, logits[index])
        error_for_all_pixel = tf.reduce_mean(cross_entropy)
        error_for_image = tf.reduce_mean(error_for_all_pixel)
        errors.append(error_for_image)
    return errors[0]


def train_step(model, image, labels):
    h, w = labels.shape
    input_image = np.append(image, np.zeros(
        shape=[h, w, model.num_classes], dtype=np.float32), axis=2)
    with tf.GradientTape() as tape:
        model([input_image])
        current_loss = loss(model.predictions,
                            model.logits, [labels], 4 if model.model_v == 1 else 2)
    gradients = tape.gradient(
        current_loss, model.trainable_variables)
    model.optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))

    return current_loss


def train(model, ckpt, manager, dataset_dir, n_epoch):

    for n in range(n_epoch):
        dataset = utils.read_dataset(dataset_dir)
        print(f"Epoch: {n}")
        for image, labels, image_id in dataset:
            current_loss = train_step(model, image, labels)
            ckpt.step.assign_add(1)
            if int(ckpt.step) % 10 == 0:
                save_path = manager.save()
                print("Checkpoint stored at step {}: {}".format(
                    int(ckpt.step), save_path))
                print("loss {:1.2f}".format(current_loss.numpy()))


def test_model(dataset_dir, model: model.RCNN, output_dir, category_colors):
    """

    """
    total_accuracy = 0
    class_correct_counts = np.zeros(model.num_classes)
    class_total_counts = np.zeros(model.num_classes)
    i = 0

    for image, labels, img_id in utils.read_dataset(dataset_dir):
        i += 1
        start_time = time.time()
        accuracy = 0.0
        h, w = labels.shape
        input_image = np.append(image, np.zeros(
            shape=[h, w, model.num_classes], dtype=np.float32), axis=2)

        model([input_image])
        logits1, logits2 = model.logits

        logits = logits1 if model.num_layers == 1 else logits2
        stride = 16 if model.model_v == 1 else 4

        predicted_labels = np.argmax(logits, axis=3)

        true_labels = labels[::stride, ::stride]

        correct_labels = np.equal(predicted_labels, true_labels)
        accuracy = np.mean(correct_labels)
        total_accuracy += accuracy

        for c in range(model.num_classes):
            current_class_labels = np.equal(true_labels, c)
            class_total_counts[c] += np.sum(current_class_labels)
            class_correct_counts[c] += np.sum(
                np.equal(true_labels, c) * correct_labels)

        print("Image #%d: %s: Accuracy: %f (time: %.1fs)" % (
            i, img_id, accuracy, time.time() - start_time))

        for layer_num in [1, 2]:
            output_filename = os.path.join(
                output_dir, img_id + '_test_%d.png' % layer_num)
            utils.save_labels_array(predicted_labels.astype(
                np.uint8), output_filename, colors=category_colors)

    print("%d Images, Total Accuracy: %f" % (i, total_accuracy / i))
    print("Per Class correct counts:", class_correct_counts)
    print("Per Class totals:", class_total_counts)
    print("Per Class accuracy:", class_correct_counts / class_total_counts)


@plac.annotations(
    model_dir=("Model name where is stored or loaded.", "option", "m", str),
    model_v=("Model name where is stored or loaded.", "option", "v", int),
    dataset_dir=(
        "Directory where the data set is stored and saved.", "option", "d", str),
    isTraining=("Flag to either train the model or test the model.",
                "option", "t", bool),
    n_epoch=("Epoch number", "option", "e", int),
    out_dir=("Output directory for predictionsr", "option", "o", str),
)
def main(model_dir='./tf_ckpts', model_v=1, dataset_dir='dataset', isTraining=False, n_epoch=30, out_dir='predictions'):
    RCNN_model = model.RCNN(output_classes, 2, learning_rate, dropout_rate,
                            leaky_relu_alpha, convolution_output_1, convolution_output_2, model_v)

    ckpt = tf.train.Checkpoint(step=tf.Variable(
        1), optimizer=RCNN_model.optimizer, net=RCNN_model)
    manager = tf.train.CheckpointManager(ckpt, f'./{model_dir}', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restaured since {}".format(manager.latest_checkpoint))
    else:
        print("Initializng model from 0.")

    if(isTraining):
        train(RCNN_model, ckpt, manager, dataset_dir, n_epoch)
    else:
        category_colors, category_names, names_to_ids = utils.read_object_classes(
            'stanford_bground_categories.txt')
        test_model(dataset_dir, RCNN_model, out_dir, category_colors)


if __name__ == "__main__":
    plac.call(main)
