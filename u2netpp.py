from keras.layers import (Conv2D, Conv2DTranspose, concatenate, Input, BatchNormalization, Activation,
                          AvgPool2D)
from keras.models import Model
from keras.utils.vis_utils import plot_model

IMG_SHAPE = (128, 128, 1)

smooth = 1.
dropout_rate = 0.5
act = "relu"


def U2NetPP():
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3

    def conv_batch_norm_relu_block(input_tensor, filter_, kernel_size=3):
        x = Conv2D(filter_, (kernel_size, kernel_size), padding='same')(input_tensor)
        x = BatchNormalization(axis=2)(x)
        x = Activation('relu')(x)

        return x

    img_input = Input(shape=IMG_SHAPE, name='input_image')

    conv1_1 = conv_batch_norm_relu_block(img_input, filter_=nb_filter[0])
    pool1 = AvgPool2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = conv_batch_norm_relu_block(pool1, filter_=nb_filter[1])
    pool2 = AvgPool2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = conv_batch_norm_relu_block(conv1_2, filter_=nb_filter[0])

    conv3_1 = conv_batch_norm_relu_block(pool2, filter_=nb_filter[2])
    pool3 = AvgPool2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = conv_batch_norm_relu_block(conv2_2, filter_=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = conv_batch_norm_relu_block(conv1_3, filter_=nb_filter[0])

    conv4_1 = conv_batch_norm_relu_block(pool3, filter_=nb_filter[3])
    pool4 = AvgPool2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = conv_batch_norm_relu_block(conv3_2, filter_=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = conv_batch_norm_relu_block(conv2_3, filter_=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = conv_batch_norm_relu_block(conv1_4, filter_=nb_filter[0])

    conv5_1 = conv_batch_norm_relu_block(pool4, filter_=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = conv_batch_norm_relu_block(conv4_2, filter_=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = conv_batch_norm_relu_block(conv3_3, filter_=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = conv_batch_norm_relu_block(conv2_4, filter_=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = conv_batch_norm_relu_block(conv1_5, filter_=nb_filter[0])

    output = Conv2D(1, (1, 1), activation='sigmoid', name='output', padding='same')(conv1_5)

    model = Model(img_input, output)
    plot_model(model, to_file="u2net_model.png", show_shapes=True)
    return model
