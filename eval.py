
from inspect import TPFLAGS_IS_ABSTRACT
from tensorflow.keras.metrics import MeanIoU, Precision, Recall
import os
import numpy as np
from matplotlib import pyplot as plt
from train import get_model
import matplotlib
from utils import *




def eval(parser_args):

    os.environ["CUDA_VISIBLE_DEVICES"] = parser_args.gpu
    model_path = parser_args.model_path
    SIZE_X = parser_args.img_size[0]
    SIZE_Y = parser_args.img_size[1]
    IMG_CHANNELS = parser_args.img_size[2]
    n_classes = parser_args.class_num
    save_dir = parser_args.save_dir

    plot_img = False

    viridis = matplotlib.cm.get_cmap('viridis', 256)
    COLORS = viridis(np.linspace(0, 1, n_classes))[...,:3]

    im_dir = parser_args.img_dir

    dirs = {'im_dir' : parser_args.img_dir, 'label_dir': parser_args.label_dir}
    im_names = [f for f in os.listdir(dirs['im_dir']) if f[-4:] == ".png"]
    data_X, data_Y = read_images(dirs, im_names, n_classes, compute_cl_weights=False)

    model = get_model(n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS)
    model.load_weights(model_path)


    _, acc = model.evaluate(data_X, data_Y)
    print("Accuracy is = ", (acc * 100.0), "%")

    y_pred = model.predict(data_X)
    y_pred_argmax = np.argmax(y_pred, axis=3)

    #Using built in keras function

    IOU_keras = MeanIoU(num_classes=n_classes)
    precision_keras = Precision()
    recall_keras = Recall()

    IOU_keras.update_state(data_Y[:,:,:,0], y_pred_argmax)
    precision_keras.update_state(data_Y[:,:,:,0], y_pred_argmax)
    recall_keras.update_state(data_Y[:,:,:,0], y_pred_argmax)

    print("Mean IoU       =", IOU_keras.result().numpy())
    print("Mean precision =", precision_keras.result().numpy())
    print("Mean recall    =", recall_keras.result().numpy())

    IOU_keras.


    #To calculate I0U for each class...
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)
    precision_by_classes, recall_by_classes, IoU_by_classes, accuracy_by_classes = []

    FN, FP, TP, TN = 0, 0, 0, 0

    for i in n_classes:
        for j in n_classes:
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
        recall_by_classes.append(precision)
        IoU_by_classes.append(precision)
        accuracy_by_classes.append(precision)




        Iou = values[i,i]/(values[i,i] + values[i,1] + values[i,2] + values[i,3] + values[1,i]+ values[2,i]+ values[3,i])
        precision = values[i,0]/(values[i,0] + values[i,1])
        IoU_by_classes.append(Iou)
        print(f"Iou for class {i} is : {Iou}")

    class1_IoU = values[i,i]/(values[i,i] + values[i,1] + values[i,2] + values[i,3] + values[1,i]+ values[2,i]+ values[3,i])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
    class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])
