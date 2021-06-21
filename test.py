from model import multi_unet_model

import tensorflow as tf
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.metrics import MeanIoU
import tensorflow.keras.backend as K

import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from train import get_model
import matplotlib
import random

def test(parser_args):

    os.environ["CUDA_VISIBLE_DEVICES"] = parser_args.gpu
    model_path = parser_args.model_path
    SIZE_X = parser_args.img_size[0]
    SIZE_Y = parser_args.img_size[1]
    IMG_CHANNELS = parser_args.img_size[2]
    n_classes = parser_args.class_num

    viridis = matplotlib.cm.get_cmap('viridis', 256)
    COLORS = viridis(np.linspace(0, 1, n_classes))[...,:3]

    # COLORS = [(matplotlib.colors.hsv_to_rgb([x, 1, 1]) * 255).astype(int) for x in np.linspace(0, 1, n_classes, endpoint=False)]

    # dirs = {'im_dir' : './datasets/20210607_cropped/images/', 'label_dir': "./datasets/20210607_cropped/labels/"}
    # dirs = {'im_dir' : '/mnt/aiteam_data/shpark/kriso/drone_seg/datasets/20210607_cropped/images', 'label_dir': "/mnt/aiteam_data/shpark/kriso/drone_seg/datasets/20210607_cropped/labels"}
    dirs = {'im_dir' : parser_args.img_dir, 'label_dir': parser_args.label_dir}




    model = get_model(n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS)
    # model.load_weights('./src/test_20210618.hdf5')
    model.load_weights(model_path)


    im_names = [f for f in os.listdir(dirs['im_dir']) if f[-4:] == ".png"]

    for n in im_names:
        label = cv2.imread(os.path.join(dirs["label_dir"], n), 0)
        if np.sum(label) == 0:
            continue

        # if n in ['DJI_0001_02m_15s_0_2.png', 'DJI_0001_02m_15s_0_3.png', 'DJI_0001_03m_00s_7_2.png', 'DJI_0001_04m_03s_3_2.png', 'DJI_0002_05m_09s_5_2.png', 'DJI_0002_05m_18s_6_4.png']:

        X_test = cv2.imread(os.path.join(dirs['im_dir'], n)) / 255.0
        plt.subplot(221)
        plt.title('test_image')
        plt.imshow(X_test)

        test_img_input=np.expand_dims(X_test, 0)

        prediction = (model.predict(test_img_input))
        y_pred_argmax=np.argmax(prediction, axis=3)
        res_mask = np.zeros((SIZE_X, SIZE_Y, 3))
        label_color = np.zeros((SIZE_X, SIZE_Y, 3))

        for i in range(SIZE_X):
            for j in range(SIZE_Y):
                res_mask[i,j] = COLORS[y_pred_argmax[0,i,j]]
                label_color[i,j] = COLORS[label[i,j]]

        plt.subplot(222)
        plt.title('prediction')
        plt.imshow(res_mask)

        plt.subplot(223)
        plt.title('GT label')
        plt.imshow(label_color)
        plt.show()




# if __name__ == '__main__':

    # test_img_number = 0

    # test_img = X_test[test_img_number]
    # test_img_input=np.expand_dims(test_img, 0)
    # prediction = (model.predict(test_img_input))
    # print(prediction.shape)
    # predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    # plt.imshow(predicted_img)
    # plt.show()
    # plt.subplot(221)
    # plt.imshow(X_test[0])

    # plt.subplot(222)
    # plt.imshow(X_test[1])

    # #IOU
    # y_pred=model.predict(X_test)
    # y_pred_argmax=np.argmax(y_pred, axis=3)


    # plt.subplot(223)
    # plt.imshow(y_pred_argmax[0, ...])
    # plt.subplot(224)
    # plt.imshow(y_pred_argmax[1, ...])
    # plt.show()
