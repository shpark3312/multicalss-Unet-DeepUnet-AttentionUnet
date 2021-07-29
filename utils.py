import tensorflow as tf
from tensorflow.keras.utils import normalize, to_categorical
import tensorflow.keras.backend as K
import os
import cv2
import numpy as np
from sklearn.utils import class_weight
import sys
from model import *
import matplotlib.pyplot as plt
from osgeo import gdal

def to_onehot(y, num_classes, mask, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    if mask:
        categorical = categorical[:,:,:,1:]

    return categorical


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, lst, batch_size, n_classes, dirs, mask, shuffle):
        self.lst = lst
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.dirs = dirs
        self.mask = mask
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.lst) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.lst))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_names = [self.lst[k] for k in indexes]

        return self.__data_generation(batch_names)

    def __data_generation(self, batch_names):

        train_images, train_labels = [], []

        for im_name in batch_names:
            if os.path.splitext(im_name)[-1].lower() == 'tif':
                img = gdal.Open(os.path.join(self.dirs['im_dir'], im_name))
                img = img.ReadAsArray()
                img = img/255.0
                img = np.transpose(img, (1, 2, 0))[...,:4]

                label = gdal.Open(os.path.join(self.dirs['label_dir'], im_name))
                label = label.ReadAsArray()

            else:
                img = cv2.imread(os.path.join(self.dirs["im_dir"], im_name))
                label = cv2.imread(os.path.join(self.dirs["label_dir"], im_name), 0)


            train_images.append(img)
            train_labels.append(label)

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        train_images = normalize(train_images, axis=1)

        if self.mask:
            train_labels_cat = to_onehot(train_labels, self.n_classes+1, self.mask)
        else:
            train_labels_cat = to_onehot(train_labels, self.n_classes, self.mask)
        # train_labels_cat = to_categorical(train_labels, num_classes=self.n_classes)

        return train_images, train_labels_cat


def weightedLoss(originalLossFunc, weightsList, mask):
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
        loss *= weightMultiplier

        if mask:
            label_mask = np.where(np.argmax(true.numpy(), axis = 3) == 0, 0, 1)
            loss *= label_mask
        return loss
    return lossFunc


def compute_class_weights(data_y):
    labels = np.array(data_y)
    labels_reshaped = labels.ravel()

    class_weights = class_weight.compute_class_weight('balanced',classes = np.unique(labels_reshaped),y = labels_reshaped)
    print("Class weights are...:", class_weights)
    return class_weights

def read_images(dirs, names, n_classes, compute_cl_weights):

    x = []
    y = []
    i = 0

    for im_name in names:
        if os.path.splitext(im_name)[-1].lower() == 'tif':
            img = gdal.Open(os.path.join(dirs['im_dir'], im_name))
            img = img.ReadAsArray()
            img = img/255.0
            img = np.transpose(img, (1, 2, 0))[...,:4]

            label = gdal.Open(os.path.join(dirs['label_dir'], im_name))
            label = label.ReadAsArray()

        else:
            img = cv2.imread(os.path.join(dirs["im_dir"], im_name))
            label = cv2.imread(os.path.join(dirs["label_dir"], im_name), 0)


        if np.sum(label) != 0:
            x.append(img)
            y.append(label)
            # plt.imshow(label)
            # plt.show()
            i += 1

    print(f'total data : {i}')

    x = np.asarray(x)
    y = np.asarray(y)

    x = normalize(x, axis=1)
    y_cat = to_categorical(y, num_classes=n_classes)

    if compute_cl_weights:
        class_weights = compute_class_weights(y)
        return x, y_cat, class_weights
    else:
        return x, y_cat

def get_apri_from_cm(cm, n_classes, mask):
    # precision_by_classes, recall_by_classes, IoU_by_classes, accuracy_by_classes, f1_by_classes = [], [], [], [], []

    if mask:
        cm = cm[1:,1:]
        n_classes -= 1

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    IoU = TP/(TP+FN+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    f1 = 2*(recall * precision) / (recall + precision)

    for i in range(n_classes):
        if mask:

            print(f"Class {i+1} | IoU = {IoU[i]:.3f}, Precision = {precision[i]:.3f}, recall = {recall[i]:.3f}, accuracy = {accuracy[i]:.3f}, f1 = {f1[i]:.3f}")
        else:
            print(f"Class {i} | IoU = {IoU[i]:.3f}, Precision = {precision[i]:.3f}, recall = {recall[i]:.3f}, accuracy = {accuracy[i]:.3f}, f1 = {f1[i]:.3f}")

    return precision, recall, IoU, accuracy, f1




def load_model(model_type, n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS):
    if model_type == 'unet':
        model = get_unet_model(n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS)
    elif model_type == 'dunet':
        model = get_dunet_model(n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS).build()
    elif model_type == 'aunet':
        model = get_aunet_model(n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS).build()
    else:
        print('Possible model types are unet, dunet, aunet')
        sys.exit()

    return model
