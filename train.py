from model import multi_unet_model
import tensorflow as tf
from tensorflow.keras.utils import normalize, to_categorical
import tensorflow.keras.backend as K
import os
import cv2
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import matplotlib

import random


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

def train(parser_args):

    os.environ["CUDA_VISIBLE_DEVICES"] = parser_args.gpu
    SIZE_X = parser_args.img_size[0]
    SIZE_Y = parser_args.img_size[1]
    IMG_CHANNELS = parser_args.img_size[2]
    n_classes = parser_args.class_num
    batch_size = parser_args.batch_size
    epochs = parser_args.epochs
    get_class_weights = parser_args.class_weights

    # dirs = {'im_dir' : './datasets/20210607_cropped/images/', 'label_dir': "./datasets/20210607_cropped/labels/"}
    dirs = {'im_dir' : '/mnt/aiteam_data/shpark/kriso/drone_seg/datasets/20210607_cropped/images', 'label_dir': "/mnt/aiteam_data/shpark/kriso/drone_seg/datasets/20210607_cropped/labels"}
    dirs = {'im_dir' : parser_args.img_dir, 'label_dir': parser_args.label_dir}

    im_names = [f for f in os.listdir(dirs['im_dir']) if f[-4:] == ".png"]

    class_weights = {0:1.47170879e-01, 1:1.04067896e+02, 2:5.23742906e+01, 3:1.56621161e+02, 4:1.47537983e+01, 5:1.99087760e+03, 6:9.82233808e+00}
    class_weights = [1.47170879e-01, 1.04067896e+02, 5.23742906e+01, 1.56621161e+02, 1.47537983e+01, 1.99087760e+03, 9.82233808e+00]

    if get_class_weights:
        data_X, data_Y, class_weights = read_images(dirs, im_names, n_classes, compute_cl_weights=get_class_weights)
    else:
        data_X, data_Y = read_images(dirs, im_names, n_classes, compute_cl_weights=get_class_weights)

    train_data_X, val_data_X, train_data_Y, val_data_Y = train_test_split(data_X, data_Y, test_size = 0.2, random_state = 0)

    print(f'train images = {len(train_data_X)}, val images = {len(val_data_X)}')

    tot_batch_num_train = int(np.ceil(len(train_data_X) / batch_size))
    tot_batch_num_val = int(np.ceil(len(val_data_X) / batch_size))

    model = get_model(n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS)

    if get_class_weights:
        model.compile(loss= weightedLoss(tf.keras.losses.categorical_crossentropy, class_weights), optimizer='adam', metrics = ['accuracy'])
    else:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # train_gen = DataGenerator(train_names, batch_size, n_classes, dirs, shuffle = True)
    # val_gen = DataGenerator(val_names, batch_size, n_classes, dirs, shuffle = False)


    history = model.fit(train_data_X,
                        train_data_Y,
                        verbose=1,
                        epochs=epochs,
                        batch_size = 8,
                        # steps_per_epoch=tot_batch_num_train,
                        # validation_steps=tot_batch_num_val,
                        validation_data=(val_data_X, val_data_Y),
                        # class_weight =class_weights,
                        shuffle=False)



    model.save('test.hdf5')

# if __name__ == '__main__':
