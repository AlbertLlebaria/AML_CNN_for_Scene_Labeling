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
    for label_f, image_f in list(train_files):
        img_id = os.path.basename(label_f).split('.')[0]
        image = image_to_np_array(image_f)
        labels = text_labels_to_np_array(label_f)
        yield image, labels, img_id


def split_dataset(data_dir, train_percentage, out_dir):
    """
    Splits a dataset containg images and labels into train and test data by giving a percentage for amount of the training data.
    Make sure the outdirctory contaings the subfolders: train and test, with images and labels subfolders on each one too.
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



def read_object_classes(classes_map_filename):
    """
    Reads an index of object classes and their corresponding names and colors.
    Each line of the file has 5 elements: R,G,B values as floats, an integer ID, and a name as a string.
    :param classes_map_filename: The filename storing the index
    :return: a tuple of 4 items:
        1. an array of ID -> category color as RGB tuple (in [0, 255])
        2. a dictionary of category color (as an RGB tuple) -> ID
        3. an array of ID -> category name
        2. a dictionary of category name -> ID
    """
    # TODO handle different potential formats better
    format_description = "Each line should contain 5 elements: (float R, float G, float B, int ID, str Name)."
    ids = set()
    ids_to_cols = {}
    ids_to_names = {}
    names_to_ids = {}
    with open(classes_map_filename, 'r') as classes_file:
        for line in classes_file:
            try:
                vals = line.split()
                if len(vals) == 0:
                    continue
                elif len(vals) == 2:
                    has_cols = False
                    category_num = int(vals[0])
                    category_name = vals[1]
                elif len(vals) == 5:
                    has_cols = True
                    rgb = tuple([int(255 * float(s)) for s in vals[:3]])
                    category_num = int(vals[3])
                    category_name = vals[4]
                else:
                    raise ValueError("Category map must have either 2 or 5 columns")

                # check for duplicate categories
                if category_num in ids:
                    sys.stderr.write("A category with this number (%d) already exists.\n" % category_num)
                    continue
                if category_name in names_to_ids:
                    sys.stderr.write("A category with this name (%s) already exists.\n" % category_name)
                    continue

                ids.add(category_num)
                ids_to_names[category_num] = category_name
                names_to_ids[category_name] = category_num
                if has_cols:
                    ids_to_cols[category_num] = rgb

            except (ValueError, IndexError) as e:
                sys.stderr.write("%s %s\n" % (format_description, e))
                continue

    max_id = max(ids)
    category_colors = [None] * (max_id + 1)
    category_names = [None] * (max_id + 1)
    for cat_id in ids:
        category_names[cat_id] = ids_to_names[cat_id]
        if has_cols:
            category_colors[cat_id] = ids_to_cols[cat_id]

    return category_colors, category_names, names_to_ids

def save_labels_array(labels, output_filename, colors):
    """
    Saves a numpy array of labels to an paletted image.
    :param colors: An array of colors for each index. Should correspond to label ID's in 'labels'
    :param labels: A 2D array of labels
    :param output_filename: The filename of the image to output
    """
    img = Image.fromarray(obj=labels, mode="P")
    # palette is a flattened array of r,g,b values, repreesnting the colors in the palette in order.
    palette = []
    for c in colors:
        palette.extend(c)
    img.putpalette(palette)
    img.save(output_filename)