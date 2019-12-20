import sys
import numpy as np
from PIL import Image
import os
from shutil import copyfile
from os.path import isfile
import glob
from scipy.io import loadmat
from random import shuffle


def matToTxtLabels(labels_dir):
    for f in os.listdir(labels_dir):
        print(f)
        filename = f.split('.')[0]
        print(filename)
        x = loadmat(labels_dir + f)
        label = x['S']
        np.savetxt(labels_dir + filename + '.regions.txt', label, delimiter=' ', fmt="%d")


def image_to_np_array(img_filename, float_cols=True):
    img = Image.open(img_filename)
    img.load()
    if float_cols:
        data = np.asarray(img, dtype="float32") / 255.0
    else:
        data = np.asarray(img, dtype="uint8")
    return data


def labels_to_np_array(lab_filename):
    img = Image.open(lab_filename)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data


def text_labels_to_np_array(lab_filename):
    label_file = open(lab_filename, 'r')
    labels = []

    for line in label_file.readlines():
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
    for label_f, image_f in list(train_files):
        img_id = os.path.basename(label_f).split('.')[0]
        image = image_to_np_array(image_f)
        labels = text_labels_to_np_array(label_f)
        yield image, labels, img_id


def split_sift_dataset(data_dir, out_dir):
    labels_dir = os.path.join(data_dir, 'labels')
    images_dir = os.path.join(data_dir, 'images')
    print(labels_dir, images_dir)

    images = [f for f in glob.glob(f'{images_dir}/*.jpg')]
    images_names = [f.split('/')[2].split('.')[0] for f in glob.glob(f'{images_dir}/*.jpg')]
    labels = [labels_dir + '/' + im + '.mat' for im in images_names]

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
            print(count)
            copyfile(image_f, f"./{out_dir}/train/{img_id}")
            copyfile(label_f, f"./{out_dir}/train/{label_id}")
            count += 1
        else:
            copyfile(image_f, f"./{out_dir}/test/{img_id}")
            copyfile(label_f, f"./{out_dir}/test/{label_id}")
            count += 1


def split_dataset(data_dir, train_percentage, out_dir):
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

    img = Image.fromarray(obj=labels[0], mode="P")
    palette = []
    for c in colors:
        palette.extend(c)
    img.putpalette(palette)
    img.save(output_filename)
