import numpy as np
import utils
import model
import tensorflow as tf
import tensorflow_datasets as tfds
import plac


learning_rate = 0.001  # @param {type: "number"}
leaky_relu_alpha = 0.2
dropout_rate = 0.3
output_classes = 8
convolution_output_1 = 25
convolution_output_2 = 50


def loss(predictions, logits, output):
    errors = []
    out = tf.Variable(output, dtype=tf.int32, shape=[None, None, None])
    for index, prediction in enumerate(predictions):
        out = tf.strided_slice(out, [0, 0, 0], [
            0, 0, 0], strides=[1, 4, 4], end_mask=7)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            out, logits[index])
        error_for_all_pixel = tf.reduce_mean(cross_entropy)
        error_for_image = tf.reduce_mean(error_for_all_pixel)
        errors.append(error_for_image)
    return tf.add_n(errors)


def train_step(model, image, labels):
    h, w = labels.shape
    input_image = np.append(image, np.zeros(
        shape=[h, w, model.num_classes], dtype=np.float32), axis=2)
    with tf.GradientTape() as tape:
        model([input_image])
        current_loss = loss(model.predictions,
                            model.logits, [labels])
    gradients = tape.gradient(
        current_loss, model.trainable_variables)
    model.optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))

    return current_loss


@plac.annotations(
    model_dir=("Model name where is stored or loaded.", "option", "m", str),
    dataset=("Model name where is stored or loaded.", "option", "d", bool),
    train=("Model name. Defaults to blank 'en' model.", "option", "t", str),
    n_epoch=("Epoch number", "option", "e", int),
)
def main(model_dir='./tf_ckpts', dataset='dataset', train=False, n_epoch=30):
    dataset = utils.read_dataset('iccv09Data')
    RCNN_model = model.RCNN(output_classes, 2, learning_rate, dropout_rate,
                            leaky_relu_alpha, convolution_output_1, convolution_output_2)

    ckpt = tf.train.Checkpoint(step=tf.Variable(
        1), optimizer=RCNN_model.optimizer, net=RCNN_model)
    manager = tf.train.CheckpointManager(ckpt, f'./{model_dir}', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restaured since {}".format(manager.latest_checkpoint))
    else:
        print("Initializng model from 0.")

    for n in range(n_epoch):
        print(f"Epoch: {n}")
        for image, labels, image_id in dataset:
            current_loss = train_step(RCNN_model, image,labels)
            ckpt.step.assign_add(1)
            if int(ckpt.step) % 10 == 0:
                save_path = manager.save()
                print("Checkpoint stored at step {}: {}".format(
                    int(ckpt.step), save_path))
                print("loss {:1.2f}".format(current_loss.numpy()))


if __name__ == "__main__":
    plac.call(main)
