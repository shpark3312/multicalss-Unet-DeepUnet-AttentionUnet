# from tensorflow.keras.metrics import MeanIoU
import os
import numpy as np
import matplotlib
from utils import load_model, read_images, get_apri_from_cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

def eval(parser_args):

    model_path = parser_args.model_path
    SIZE_X = parser_args.img_size[0]
    SIZE_Y = parser_args.img_size[1]
    IMG_CHANNELS = parser_args.img_size[2]
    n_classes = parser_args.class_num
    # save_dir = parser_args.save_dir
    mask = parser_args.mask
    model_type = parser_args.model_type

    dirs = {'im_dir' : parser_args.img_dir, 'label_dir': parser_args.label_dir}
    im_names = [f for f in os.listdir(dirs['im_dir']) if f[-4:] == ".png"]

    _, val_names = train_test_split(im_names, test_size = 0.2, random_state = 0)

    data_X, data_Y = read_images(dirs, val_names, n_classes, compute_cl_weights=False)

    model = load_model(model_type, n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS)
    model.load_weights(model_path)

    y_pred = model.predict(data_X)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    data_Y = np.argmax(data_Y, axis=3)

    cm = confusion_matrix(data_Y.ravel(), y_pred_argmax.ravel())

    print('confusion metrics = \n', cm)

    precision, recall, IoU, accuracy, f1 = get_apri_from_cm(cm, n_classes, mask)

    if mask:
        n_classes -= 1

    f = open(f'results/{os.path.splitext(os.path.split(model_path)[-1])[0]}.csv', 'w')
    f.write("Class,IoU,Precision,recall,accuracy,f1\n")

    for i in range(n_classes):
        if mask:
            f.write(f"{i+1},{IoU[i]:.3f},{precision[i]:.3f},{recall[i]:.3f},{accuracy[i]:.3f},{f1[i]:.3f}\n")
        else:
            f.write(f"{i},{IoU[i]:.3f},{precision[i]:.3f},{recall[i]:.3f},{accuracy[i]:.3f},{f1[i]:.3f}\n")
    return
