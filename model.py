from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Activation
import tensorflow as tf


def get_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    c0 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c0 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c0)
    p0 = MaxPooling2D((2, 2))(c0)

    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p0)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)


    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c0], axis=3)
    c8 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c8)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


class get_dunet_model():
    def __init__ (self, n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
        self.n_classes = n_classes
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        self.filter_num = 64
        # self.filter_num = 128

    def encoder_layer(self, x):
        p = Activation('relu')(x)
        c = Conv2D(self.filter_num, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p)
        c1 = Conv2D(self.filter_num/2, (3, 3), kernel_initializer='he_normal', padding='same')(c)
        c1 += p
        p1 = MaxPooling2D((2, 2))(c1)

        return c1, p1

    def decoder_layer(self, x, concat, add = None):
        if add is not None:
            x += add

        u1 = UpSampling2D((2,2))(x)
        u = Activation('relu')(u1)
        u = concatenate([concat, u], axis = 3)
        c = Conv2D(self.filter_num, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u)
        c = Conv2D(self.filter_num/2, (2, 2), kernel_initializer='he_normal', padding='same')(c)

        return c, u1


    def build(self):
        inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        s = inputs

        c = Conv2D(self.filter_num, (3, 3), kernel_initializer='he_normal', padding='same')(s)
        c = Activation('relu')(c)
        c = Conv2D(self.filter_num, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
        c0 = Conv2D(self.filter_num/2, (2, 2), kernel_initializer='he_normal', padding='same')(c)
        p0 = MaxPooling2D(pool_size = (2, 2))(c0)

        c1, p1 = self.encoder_layer(p0)
        c2, p2 = self.encoder_layer(p1)
        c3, p3 = self.encoder_layer(p2)
        c4, p4 = self.encoder_layer(p3)
        c5, p5 = self.encoder_layer(p4)
        c6, p6 = self.encoder_layer(p5)

        p6 = Activation('relu')(p6)

        u1, c = self.decoder_layer(p6, c6, add = None)
        u2, c = self.decoder_layer(c, c5, add = u1)
        u3, c = self.decoder_layer(c, c4, add = u2)
        u4, c = self.decoder_layer(c, c3, add = u3)
        u5, c = self.decoder_layer(c, c2, add = u4)
        u6, c = self.decoder_layer(c, c1, add = u5)
        _, c = self.decoder_layer(c, c0, add = u6)

        outputs = Conv2D(self.n_classes, (1, 1), activation='softmax')(c)

        model = Model(inputs=[inputs], outputs=[outputs])

        return model

class get_aunet_model():
    def __init__ (self, n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
        self.n_classes = n_classes
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNELS = IMG_CHANNELS
        self.filter_num = 16
        # self.filter_num = 32
        self.W_g5 = Conv2D(self.filter_num*4, 1, padding='same', use_bias=True)
        self.W_x5 = Conv2D(self.filter_num*4, 1, padding='same', use_bias=False)
        self.psi5 = Conv2D(1, 1, activation='sigmoid', padding='same')
        self.W_g6 = Conv2D(self.filter_num*2, 1, padding='same', use_bias=True)
        self.W_x6 = Conv2D(self.filter_num*2, 1, padding='same', use_bias=False)
        self.psi6 = Conv2D(1, 1, activation='sigmoid', padding='same')
        self.W_g7 = Conv2D(self.filter_num, 1, padding='same', use_bias=True)
        self.W_x7 = Conv2D(self.filter_num, 1, padding='same', use_bias=False)
        self.psi7 = Conv2D(1, 1, activation='sigmoid', padding='same')

    def build(self):

        inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
        s = inputs

        x = Conv2D(self.filter_num, 3, activation='relu', padding='same')(s)
        x1 = Conv2D(self.filter_num, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2,2))(x1)

        x = Conv2D(self.filter_num*2, 3, activation='relu', padding='same')(x)
        x2= Conv2D(self.filter_num*2, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2,2))(x2)

        x = Conv2D(self.filter_num*4, 3, activation='relu', padding='same')(x)
        x3 = Conv2D(self.filter_num*4, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2,2))(x3)

        x = Conv2D(self.filter_num*8, 3, activation='relu', padding='same')(x)
        x = Conv2D(self.filter_num*8, 3, activation='relu', padding='same')(x)

        x = UpSampling2D(size=(2,2))(x)

        x3 = self.attention_gate(x, x3, self.W_g5, self.W_x5, self.psi5)
        x = tf.concat([x3, x], axis=3)
        x = Conv2D(self.filter_num*4, 3, activation='relu', padding='same')(x)
        x = Conv2D(self.filter_num*4, 3, activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2,2))(x)

        x2 = self.attention_gate(x, x2, self.W_g6, self.W_x6, self.psi6)
        x = tf.concat([x2, x], axis=3)
        x = Conv2D(self.filter_num*2, 3, activation='relu', padding='same')(x)
        x = Conv2D(self.filter_num*2, 3, activation='relu', padding='same')(x)

        x = UpSampling2D(size=(2,2))(x)
        x1 = self.attention_gate(x, x1, self.W_g7, self.W_x7, self.psi7)
        x = tf.concat([x1, x], axis=3)

        x = Conv2D(self.filter_num, 3, activation='relu', padding='same')(x)
        x = Conv2D(self.filter_num, 3, activation='relu', padding='same')(x)

        outputs = Conv2D(self.n_classes, (1, 1), activation='softmax')(x)

        model = Model(inputs=[inputs], outputs=[outputs])

        return model

    def attention_gate(self, g, x, W_g, W_x, psi):
        return x*psi(tf.nn.relu(W_g(g)+W_x(x)))


if __name__ == '__main__':
    model = get_aunet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1).build()

    model.summary()
