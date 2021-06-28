from tensorflow.keras.metrics import MeanIoU
import os
import numpy as np
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
    mask = parser_args.mask
    model_type = parser_args.model_type

    plot_img = False

    viridis = matplotlib.cm.get_cmap('viridis', 256)
    COLORS = viridis(np.linspace(0, 1, n_classes))[...,:3]

    dirs = {'im_dir' : parser_args.img_dir, 'label_dir': parser_args.label_dir}
    im_names = [f for f in os.listdir(dirs['im_dir']) if f[-4:] == ".png"]
    data_X, data_Y = read_images(dirs, im_names, n_classes, compute_cl_weights=False)

    model = load_model(model_type, n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS)
    model.load_weights(model_path)

    y_pred = model.predict(data_X)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    data_Y = np.argmax(data_Y, axis=3)

    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(data_Y, y_pred_argmax)

    cm = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(cm)

    get_apri_from_cm(cm, n_classes, mask)
