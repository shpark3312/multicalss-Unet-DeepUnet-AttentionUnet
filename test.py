import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from train import get_model
import matplotlib

def test(parser_args):

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

    model = get_model(n_classes, SIZE_X, SIZE_Y, IMG_CHANNELS)
    model.load_weights(model_path)

    im_names = [f for f in os.listdir(im_dir) if f[-4:] == ".png"]

    for n in im_names:

        save_path = os.path.join(save_dir, f'{n[:-4]}_mask.{n[-3:]}')

        X_test = cv2.imread(os.path.join(im_dir, n)) / 255.0

        test_img_input=np.expand_dims(X_test, 0)

        prediction = (model.predict(test_img_input))
        y_pred_argmax=np.argmax(prediction, axis=3)
        res_mask = np.zeros((SIZE_X, SIZE_Y, 3))
        # label_color = np.zeros((SIZE_X, SIZE_Y, 3))

        for i in range(SIZE_X):
            for j in range(SIZE_Y):
                res_mask[i,j] = COLORS[y_pred_argmax[0,i,j]]
                # label_color[i,j] = COLORS[label[i,j]]

        cv2.imwrite(save_path, res_mask)

        if plot_img:
            plt.subplot(221)
            plt.title('test_image')
            plt.imshow(X_test)

            plt.subplot(222)
            plt.title('prediction')
            plt.imshow(res_mask)

            # plt.subplot(223)
            # plt.title('GT label')
            # plt.imshow(label_color)
            # plt.show()




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
