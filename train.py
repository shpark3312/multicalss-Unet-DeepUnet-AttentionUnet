
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import matplotlib.pyplot as plt
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Precision, Recall
from datetime import datetime
import tensorflow as tf

def train(parser_args):

    SIZE_X = parser_args.img_size[0]
    SIZE_Y = parser_args.img_size[1]
    IMG_CHANNELS = parser_args.img_size[2]
    n_classes = parser_args.class_num
    batch_size = parser_args.batch_size
    epochs = parser_args.epochs
    get_class_weights = parser_args.class_weights
    mask = parser_args.mask
    model_dir = parser_args.model_dir
    model_type = parser_args.model_type

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    todays_date = datetime.now()
    dirs = {'im_dir' : parser_args.img_dir, 'label_dir': parser_args.label_dir}

    im_names = [f for f in os.listdir(dirs['im_dir']) if f[-4:] == ".png"]

    # class_weights = {0:1.47170879e-01, 1:1.04067896e+02, 2:5.23742906e+01, 3:1.56621161e+02, 4:1.47537983e+01, 5:1.99087760e+03, 6:9.82233808e+00}
    # class_weights = [1.47170879e-01, 1.04067896e+02, 5.23742906e+01, 1.56621161e+02, 1.47537983e+01, 1.99087760e+03, 9.82233808e+00]

    if get_class_weights:
        _, _, class_weights = read_images(dirs, im_names, n_classes, compute_cl_weights = get_class_weights)
    else:
        class_weights = [1 for i in range(n_classes)]

    train_names, val_names = train_test_split(im_names, test_size = 0.2, random_state = 0)

    tot_batch_num_train = int(np.floor(len(train_names) / batch_size))
    tot_batch_num_val = int(np.floor(len(val_names) / batch_size))

    print(f'train batches = {tot_batch_num_train}, validation batches = {tot_batch_num_val}')

    model = load_model(model_type, n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS)
    tf.config.experimental_run_functions_eagerly(True)
    # model.compile(loss = weightedLoss(categorical_crossentropy, class_weights, mask), run_eagerly=True, optimizer='adam', metrics = ['accuracy'])

    metric_per_class = [Recall(class_id=i) for i in range(n_classes)] +  [Precision(class_id=i) for i in range(n_classes)]
    model.compile(loss = weightedLoss(categorical_crossentropy, class_weights, mask), run_eagerly=True, optimizer='adam', metrics=metric_per_class)

    model.summary()

    train_gen = DataGenerator(train_names, batch_size, n_classes, dirs, shuffle = False)
    val_gen = DataGenerator(val_names, batch_size, n_classes, dirs, shuffle = False)

    history = model.fit(train_gen,
                        verbose=1,
                        epochs=epochs,
                        steps_per_epoch=tot_batch_num_train,
                        validation_steps=tot_batch_num_val,
                        validation_data=val_gen,
                        shuffle=False)
    if get_class_weights:
        model_filename = f'{todays_date.year}{todays_date.month:02}{todays_date.day:02}_{todays_date.hour:02}_{todays_date.minute:02}_{model_type}_classWeights.hdf5'
    else:
        model_filename = f'{todays_date.year}{todays_date.month:02}{todays_date.day:02}_{todays_date.hour:02}_{todays_date.minute:02}_{model_type}.hdf5'

    model.save(os.path.join(model_dir, model_filename))

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'y', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # acc = history.history['acc']
    # val_acc = history.history['val_acc']

    # plt.plot(epochs, acc, 'y', label='Training Accuracy')
    # plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    # plt.title('Training and validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

# if __name__ == '__main__':
