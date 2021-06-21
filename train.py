
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import matplotlib.pyplot as plt


def train(parser_args):

    SIZE_X = parser_args.img_size[0]
    SIZE_Y = parser_args.img_size[1]
    IMG_CHANNELS = parser_args.img_size[2]
    n_classes = parser_args.class_num
    batch_size = parser_args.batch_size
    epochs = parser_args.epochs
    get_class_weights = parser_args.class_weights

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

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# if __name__ == '__main__':
