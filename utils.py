import tensorflow as tf
from tensorflow.keras.utils import normalize, to_categorical
import tensorflow.keras.backend as K
import os
import cv2
import numpy as np
from sklearn.utils import class_weight
from model import multi_unet_model


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, lst, batch_size, n_classes, dirs, shuffle):
        self.lst = lst
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.dirs = dirs
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.lst) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.lst))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = range(0, len(self.lst), self.batch_size)
        batch_names = self.lst[indexes[index] : indexes[index] + self.batch_size]

        return self.__data_generation(batch_names)

    def __data_generation(self, batch_names):

        train_images, train_labels = [], []

        for im_name in batch_names:
            img = cv2.imread(os.path.join(self.dirs["im_dir"], im_name))
            label = cv2.imread(os.path.join(self.dirs["label_dir"], im_name), 0)
            train_images.append(img)
            train_labels.append(label)

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        train_images = normalize(train_images, axis=1)
        train_labels_cat = to_categorical(train_labels, num_classes=self.n_classes)

        return train_images, train_labels_cat


def weightedLoss(originalLossFunc, weightsList):
    def lossFunc(true, pred):
        axis = -1
        classSelectors = K.argmax(true, axis=axis)
        classSelectors = [K.equal(i, tf.cast(classSelectors, tf.int32)) for i in range(len(weightsList))]
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        weights = [sel * w for sel,w in zip(classSelectors, weightsList)]

        weightMultiplier = weights[0]

        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]

        loss = originalLossFunc(true,pred)
        loss = loss * weightMultiplier

        return loss
    return lossFunc


def compute_class_weights(data_y):
    labels = np.array(data_y)
    labels_reshaped = labels.ravel()

    class_weights = class_weight.compute_class_weight('balanced',classes = np.unique(labels_reshaped),y = labels_reshaped)
    print("Class weights are...:", class_weights)
    return class_weights


def get_model(n_classes, img_height, img_width, img_channels):
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=img_height, IMG_WIDTH=img_width, IMG_CHANNELS=img_channels)


def read_images(dirs, names, n_classes, compute_cl_weights):

    x = []
    y = []
    i = 0

    for name in names:
        img = cv2.imread(os.path.join(dirs['im_dir'], name)) / 255.0
        label = cv2.imread(os.path.join(dirs['label_dir'], name), 0)

        if np.sum(label) != 0:
            x.append(img)
            y.append(label)
            i += 1

    print(f'total data : {i}')

    x = np.asarray(x)
    y = np.asarray(y)

    y_cat = to_categorical(y, num_classes=n_classes)

    if compute_cl_weights:
        class_weights = compute_class_weights(y)
        return x, y_cat, class_weights
    else:
        return x, y_cat
