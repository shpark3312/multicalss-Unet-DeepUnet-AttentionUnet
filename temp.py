import matplotlib
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


if __name__=='__main__':

    n_classes = 7
    im_dir = './datasets/20210630_cropped_filtered/labels'
    im_save_dir = './datasets/20210630_cropped_filtered/labels_color'

    viridis = matplotlib.cm.get_cmap('viridis', 256)
    COLORS = viridis(np.linspace(0, 1, n_classes))[...,:3]


    im_names = [f for f in os.listdir(im_dir) if f[-4:] == ".png"]

    for n in im_names:
        img = cv2.imread(os.path.join(im_dir, n), 0)

        label = np.zeros((*img.shape, 3))

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                label[i,j] = COLORS[img[i,j]]

        plt.imsave(os.path.join(im_save_dir, n), label)

        # plt.imshow(label)
        # plt.show()
                # label_color[i,j] = COLORS[label[i,j]]
