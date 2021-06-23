import tensorflow as tf
from tensorflow.keras.utils import normalize, to_categorical
import tensorflow.keras.backend as K
import os
import cv2
import numpy as np
from sklearn.utils import class_weight
from model import multi_unet_model

# def DataGenerator(lst, n_classes, dirs):
#     for im_name in lst:
#         img = cv2.imread(os.path.join(dirs["im_dir"], im_name)) / 255.0
#         label = cv2.imread(os.path.join(dirs["label_dir"], im_name), 0)
#         img = np.array(img)
#         label = np.array(label)
#         # train_images = normalize(train_images, axis=1)
#         label = to_categorical(label, num_classes=n_classes)

#         yield img, label

# class DataGenerator():
#     def __init__(self, lst, batch_size, n_classes, dirs):
#         self.lst = lst
#         self.batch_size = batch_size
#         self.n_classes = n_classes
#         self.dirs = dirs

#     def __call__(self):

#         for im_name in self.lst:
#             img = cv2.imread(os.path.join(self.dirs["im_dir"], im_name)) / 255.0
#             label = cv2.imread(os.path.join(self.dirs["label_dir"], im_name), 0)
#             img = np.array(img)
#             label = np.array(label)
#             # train_images = normalize(train_images, axis=1)
#             label = to_categorical(label, num_classes=self.n_classes)

#             yield img, label




class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, lst, batch_size, n_classes, dirs, shuffle):
        self.lst = lst
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.dirs = dirs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.lst) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.lst))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # indexes = range(0, len(self.lst), self.batch_size)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_names = [self.lst[k] for k in indexes]

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
            label_mask = np.where(true == 0, 0, 1)
            loss *= label_mask
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

def get_apci_from_cm(cm, n_classes, mask):
    precision_by_classes, recall_by_classes, IoU_by_classes, accuracy_by_classes = [], [], [], []

    FN, FP, TP, TN = 0, 0, 0, 0

    if mask:
        for i in range(1, n_classes):
            for j in range(1, n_classes):
                if i == j:
                    continue
                FN += cm[i,j]
                FP += cm[j,i]
                
            TP = cm[i,i]
            TN = np.trace(cm) - TP - cm[0,0]

            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            IoU = TP/(TP+FN+FP)
            accuracy = (TP+TN)/(TP+TN+FP+FN)

            precision_by_classes.append(precision)
            recall_by_classes.append(recall)
            IoU_by_classes.append(IoU)
            accuracy_by_classes.append(accuracy)

            print(f"Class {i} | IoU = {IoU:.3f}, Precision = {precision:.3f}, recall = {recall:.3f}, accuracy = {accuracy:.3f}")
    else:
        for i in range(n_classes):
            for j in range(n_classes):
                if i == j:
                    continue
                FN += cm[i,j]
                FP += cm[j,i]
            TP = cm[i,i]
            TN = np.trace(cm) - TP

            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            IoU = TP/(TP+FN+FP)
            accuracy = (TP+TN)/(TP+TN+FP+FN)

            precision_by_classes.append(precision)
            recall_by_classes.append(recall)
            IoU_by_classes.append(IoU)
            accuracy_by_classes.append(accuracy)

            print(f"Class {i} | IoU = {IoU:.3f}, Precision = {precision:.3f}, recall = {recall:.3f}, accuracy = {accuracy:.3f}")
