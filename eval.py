
from inspect import TPFLAGS_IS_ABSTRACT
from tensorflow.keras.metrics import MeanIoU, Precision, Recall
import os
import numpy as np
from matplotlib import pyplot as plt
from train import get_model
import matplotlib
from utils import *




def eval(parser_args):

    model_path = parser_args.model_path
    SIZE_X = parser_args.img_size[0]
    SIZE_Y = parser_args.img_size[1]
    IMG_CHANNELS = parser_args.img_size[2]
    n_classes = parser_args.class_num
    # save_dir = parser_args.save_dir

    plot_img = False

    viridis = matplotlib.cm.get_cmap('viridis', 256)
    COLORS = viridis(np.linspace(0, 1, n_classes))[...,:3]

    im_dir = parser_args.img_dir

    dirs = {'im_dir' : parser_args.img_dir, 'label_dir': parser_args.label_dir}
    im_names = [f for f in os.listdir(dirs['im_dir']) if f[-4:] == ".png"]
    # data_X, data_Y = read_images(dirs, im_names[0:10:len(im_names)], n_classes, compute_cl_weights=False)
    data_X, data_Y = read_images(dirs, im_names, n_classes, compute_cl_weights=False)

    model = get_model(n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS)
    model.load_weights(model_path)

    y_pred = model.predict(data_X)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    data_Y = np.argmax(data_Y, axis=3)


    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(data_Y, y_pred_argmax)

    print("Mean IoU       =", IOU_keras.result().numpy())

    #To calculate I0U for each class...
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)
    precision_by_classes, recall_by_classes, IoU_by_classes, accuracy_by_classes = [], [], [], []

    FN, FP, TP, TN = 0, 0, 0, 0

    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                continue
            FN += values[i,j]
            FP += values[j,i]
        TP = values[i,i]
        TN = np.trace(values) - TP

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        IoU = TP/(TP+FN+FP)
        accuracy = (TP+TN)/(TP+TN+FP+FN)

        precision_by_classes.append(precision)
        recall_by_classes.append(recall)
        IoU_by_classes.append(IoU)
        accuracy_by_classes.append(accuracy)

        print(f"Class {i} | IoU = {IoU:.3f}, Precision = {precision:.3f}, recall = {recall:.3f}, accuracy = {accuracy:.3f}")
