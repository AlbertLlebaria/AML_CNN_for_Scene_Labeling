import sys
import numpy as np
from PIL import Image
import os
import os
from os.path import isfile
import glob


def image_to_np_array(img_filename, float_cols=True):
    """
    Reads an image into a numpy array, with shape [height x width x 3]
    Each pixel is represented by 3 RGB values, either as floats in [0, 1] or as ints in [0, 255]
    :param img_filename: The filename of the image to load
    :param float_cols: Whether to load colors as floats in [0, 1] or as ints in [0, 255]
    :return: A numpy array containing the image data
    """
    img = Image.open(img_filename)
    img.load()
    if float_cols:
        data = np.asarray(img, dtype="float32") / 255.0
    else:
        data = np.asarray(img, dtype="uint8")
    return data


def labels_to_np_array(lab_filename):
    """
    Reads an image of category labels as a numpy array of category IDs.
    # corresponds to label ID.
    NOTE: The image data must already be in a color pallette such that color
    The "Playing for Data" dataset is configured in this way (http://download.visinf.tu-darmstadt.de/data/from_games/)
    :param lab_filename: The filename of the label image to load
    :return: A numpy array containing the label ID for each pixel
    """
    img = Image.open(lab_filename)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data

def text_labels_to_np_array(lab_filename):
    label_file = open(lab_filename, 'r')
    labels = []

    for line in  label_file.readlines():
        labels.append(list(map(lambda n: max(0, int(n)), line.split())))
    return np.array(labels, dtype=np.int8)

def read_dataset(data_dir):
    labels_dir = os.path.join(data_dir, 'labels')
    images_dir = os.path.join(data_dir, 'images')

    labels = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if
              isfile(os.path.join(labels_dir, f)) and not f.startswith('.')]
    files = filter(lambda f: f.endswith('regions.txt'), labels)
    labels = sorted(files)

    images = [f for f in glob.glob(f'{images_dir}/*.jpg')]
    images = sorted(images)
    train_files = zip(labels, images)
    for label_f, image_f in train_files:
        # if os.path.basename(label_f).split('.')[0] != os.path.basename(image_f).split('.')[0]:
        #     # print ("UNEQUAL IMAGE NAMES!", label_f, image_f)
        img_id = os.path.basename(label_f).split('.')[0]
        image = image_to_np_array(image_f)
        labels = text_labels_to_np_array(label_f)
        yield image, labels, img_id


