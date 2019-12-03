import sys
import numpy as np
from PIL import Image
import os
from shutil import copyfile
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
    :param lab_filename: The filename of the label image to load
    :return: A numpy array containing the label ID for each pixel
    """
    img = Image.open(lab_filename)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data


def text_labels_to_np_array(lab_filename):
    """
    Reads an label file conaining the labels per pixels of an image as a numpy array.
    :param lab_filename: The filename of the label file to load
    """
    label_file = open(lab_filename, 'r')
    labels = []

    for line in label_file.readlines():
        labels.append(list(map(lambda n: max(0, int(n)), line.split())))
    return np.array(labels, dtype=np.int8)


def read_dataset(data_dir):
    """
    Reads a directory containg two folders, one called images and the other labels.
    Returns a generator that yields the image as a numpy array, the labels of that image and the image ID.
    It is important that label files and image failes that match have the same name. Also labels must contain .region.txt
    """
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
        img_id = os.path.basename(label_f).split('.')[0]
        image = image_to_np_array(image_f)
        labels = text_labels_to_np_array(label_f)
        yield image, labels, img_id


def split_dataset(data_dir, train_percentage, out_dir):
    """
    Splits a dataset containg images and labels into train and test data by giving a percentage for amount of the training data.
    """
    labels_dir = os.path.join(data_dir, 'labels')
    images_dir = os.path.join(data_dir, 'images')
    labels = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if
              isfile(os.path.join(labels_dir, f)) and not f.startswith('.')]
    files = filter(lambda f: f.endswith('regions.txt'), labels)
    labels = sorted(files)

    images = [f for f in glob.glob(f'{images_dir}/*.jpg')]
    images = sorted(images)
    train_files = zip(labels, images)

    train_images_len = int(len(images)*0.8)
    count = 0
    for label_f, image_f in train_files:
        if os.path.basename(label_f).split('.')[0] != os.path.basename(image_f).split('.')[0]:
            print("UNEQUAL IMAGE NAMES!", label_f, image_f)
        else:
            img_id = image_f.split('/')[1::]
            img_id = '/'.join(img_id)
            label_id = label_f.split('/')[1::]
            label_id = '/'.join(label_id)
            if count < train_images_len:
                copyfile(image_f, f"./{out_dir}/train/{img_id}")
                copyfile(label_f, f"./{out_dir}/train/{label_id}")
                count += 1
            else:
                copyfile(image_f, f"./{out_dir}/test/{img_id}")
                copyfile(label_f, f"./{out_dir}/test/{label_id}")
                count += 1



# split('iccv09Data', 0.8,'dataset')
