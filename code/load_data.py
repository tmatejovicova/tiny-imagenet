from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from keras.utils import np_utils
import constants

def build_label_dicts():
    label_list = []
    label_dict, class_dict = {}, {}
    with open(constants.IMAGE_DIR + 'wnids.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            id = line.strip()
            label_dict[id] = i
            label_list.append(id)
    with open(constants.IMAGE_DIR + 'words.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            id, desc = line.split('\t')
            id = id.strip()
            if id in label_dict:
                class_dict[label_dict[id]] = desc.strip()
    return label_list, label_dict, class_dict

def get_generators(label_list, label_dict, class_dict, batch_size, augment):
    x_val, y_val = get_val_data(label_dict, class_dict)
    tg = get_train_generator(batch_size, label_list, augment)
    vg = get_val_generator(x_val, y_val, batch_size)
    return tg, vg

def get_val_data(label_dict, class_dict):
    x = []
    y = []
    # numbw = 0
    with open(constants.IMAGE_DIR + 'val/val_annotations.txt') as f:
        for i, line in enumerate(f.readlines()):
            segments = line.split('\t')
            img_name = segments[0]
            img_id = segments[1]
            img = Image.open(constants.IMAGE_DIR + 'val/images/' + img_name)
            if(np.array(img).shape==(64,64)): # Image is greyscale
                # print(class_dict[label_dict[img_id]])
                # print(img.shape)
                # numbw += 1
                img = np.stack((img,)*3, axis=-1)
            x.append(np.array(img))
            y.append(label_dict[img_id])
    y = np_utils.to_categorical(np.asarray(y), num_classes = 200)
    return np.asarray(x), np.asarray(y)

def get_train_generator(batch_size, label_list, augment):
    if augment:
        train_datagen = ImageDataGenerator(
            rescale = 1. / 255,
            rotation_range = 20,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            brightness_range = [0.5, 1.5],
            horizontal_flip = True)
    else:
        train_datagen = ImageDataGenerator(rescale = 1. / 255)
    train_generator = train_datagen.flow_from_directory(
        constants.IMAGE_DIR + 'train',
        target_size = (constants.IMG_WIDTH, constants.IMG_WIDTH),
        color_mode = 'rgb',
        batch_size = batch_size,
        class_mode = 'categorical',
        classes = label_list)
    return train_generator

def get_val_generator(x, y, batch_size):
    val_datagen = ImageDataGenerator(rescale = 1. / 255)
    val_generator = val_datagen.flow(
        x = x,
        y = y,
        batch_size = batch_size)
    return val_generator
