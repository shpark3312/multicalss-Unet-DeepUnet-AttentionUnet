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
import random




if __name__ == '__main__':

    model = get_model(n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS)
    model.load_weights('test.hdf5')

    _, acc = model.evaluate(X_test, y_test_cat)
    print("Accuracy is = ", (acc * 100.0), "%")


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


    ##################################

    ##################################################

    #Using built in keras function

    n_classes = 4
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())


    #To calculate I0U for each class...
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)
    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
    class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

    print("IoU for class1 is: ", class1_IoU)
    print("IoU for class2 is: ", class2_IoU)
    print("IoU for class3 is: ", class3_IoU)
    print("IoU for class4 is: ", class4_IoU)

    plt.imshow(train_images[0, :,:,0], cmap='gray')
    plt.imshow(train_masks[0], cmap='gray')
    #######################################################################
    #Predict on a few images
    #model = get_model()
    #model.load_weights('???.hdf5')

    test_img_number = random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth=y_test[test_img_number]
    test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]


    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    plt.show()

    ####################################################################

    Predict on large image

    Apply a trained model on large image



    large_image = cv2.imread('large_images/large_image.tif', 0)
    #This will split the image into small images of shape [3,3]
    patches = patchify(large_image, (128, 128), step=128)  #Step=256 for 256 patches means no overlap

    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            print(i,j)

            single_patch = patches[i,j,:,:]
            single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
            single_patch_input=np.expand_dims(single_patch_norm, 0)
            single_patch_prediction = (model.predict(single_patch_input))
            single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]

            predicted_patches.append(single_patch_predicted_img)

    predicted_patches = np.array(predicted_patches)

    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 128,128) )

    reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
    plt.imshow(reconstructed_image, cmap='gray')
    #plt.imsave('data/results/segm.jpg', reconstructed_image, cmap='gray')

    plt.hist(reconstructed_image.flatten())  #Threshold everything above 0

    # final_prediction = (reconstructed_image > 0.01).astype(np.uint8)
    # plt.imshow(final_prediction)

    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('Large Image')
    plt.imshow(large_image, cmap='gray')
    plt.subplot(222)
    plt.title('Prediction of large Image')
    plt.imshow(reconstructed_image, cmap='jet')
    plt.show()
