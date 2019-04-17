'''Trains a simple ternarize CNN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to % test accuracy after 20 epochs using tensorflow backend
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
import keras.backend as K

K.set_image_data_format('channels_first')

from ternary_ops import ternarize
from ternary_layers import TernaryDense, TernaryConv2D


def ternary_tanh(x):
    x = K.clip(x, -1, 1)
    return ternarize(x)

H = 1.
kernel_lr_multiplier = 'Glorot'

# nn
batch_size = 50
epochs = 20
nb_channel = 1
img_rows = 28
img_cols = 28
nb_filters = 32
nb_conv = 3
nb_pool = 2
nb_hid = 128
nb_classes = 10
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
p1 = 0.25
p2 = 0.5

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 1, 28, 28)
X_test = X_test.reshape(10000, 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to ternary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1

##############################################################################################################
model = Sequential()
# conv1
model.add(TernaryConv2D(32, kernel_size=kernel_size, input_shape=(channels, img_rows, img_cols),
                       data_format='channels_first',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       padding='same', use_bias=use_bias, name='conv1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
model.add(Activation(ternary_tanh, name='act1'))
##############################################################################################################

# conv_dw_2_1
model.add(TernaryConv2D(32, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_dw_2_1'))
#model.add(MaxPooling2D(pool_size=pool_size, name='pool2', data_format='channels_first'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_2_1'))
model.add(Activation(ternary_tanh, name='conv_dw_act_2_1'))

# conv_1x1_2_1
model.add(TernaryConv2D(32, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_2_1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv1x1_bn_2_1'))
model.add(Activation(ternary_tanh, name='conv1x1_act_2_1'))

# conv_dw_2_2
model.add(TernaryConv2D(64, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first', strides=2,
                       padding='same', use_bias=use_bias, name='conv_dw_2_2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_2_2'))
model.add(Activation(ternary_tanh, name='conv_dw_act_2_2'))

# conv_1x1_2_2
model.add(TernaryConv2D(64, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_2_2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_2_2'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_2_2'))
##############################################################################################################

# conv_dw_3_1
model.add(TernaryConv2D(128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_dw_3_1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_3_1'))
model.add(Activation(ternary_tanh, name='conv_dw_act_3_1'))

# conv_1x1_3_1
model.add(TernaryConv2D(128, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_3_1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_3_1'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_3_1'))

# conv_dw_3_2
model.add(TernaryConv2D(128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first', strides=2,
                       padding='same', use_bias=use_bias, name='conv_dw_3_2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_3_2'))
model.add(Activation(ternary_tanh, name='conv_dw_act_3_2'))

# conv_1x1_3_2
model.add(TernaryConv2D(128, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_3_2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_3_2'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_3_2'))
##############################################################################################################

# conv_dw_4_1
model.add(TernaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_dw_4_1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_4_1'))
model.add(Activation(ternary_tanh, name='conv_dw_act_4_1'))

# conv_1x1_4_1
model.add(TernaryConv2D(256, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_4_1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_4_1'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_4_1'))

# conv_dw_4_2
model.add(TernaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first', strides=2,
                       padding='same', use_bias=use_bias, name='conv_dw_4_2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_4_2'))
model.add(Activation(ternary_tanh, name='conv_dw_act_4_2'))

# conv_1x1_4_2
model.add(TernaryConv2D(256, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_4_2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_4_2'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_4_2'))
##############################################################################################################

# conv_dw_5_1
model.add(TernaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_dw_5_1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_5_1'))
model.add(Activation(ternary_tanh, name='conv_dw_act_5_1'))

# conv_1x1_5_1
model.add(TernaryConv2D(512, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_5_1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_5_1'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_5_1'))

# conv_dw_5_2
model.add(TernaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_dw_5_2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_5_2'))
model.add(Activation(ternary_tanh, name='conv_dw_act_5_2'))

# conv_1x1_5_2
model.add(TernaryConv2D(512, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_5_2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_5_2'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_5_2'))

# conv_dw_5_3
model.add(TernaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_dw_5_3'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_5_3'))
model.add(Activation(ternary_tanh, name='conv_dw_act_5_3'))

# conv_1x1_5_3
model.add(TernaryConv2D(512, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_5_3'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_5_3'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_5_3'))

# conv_dw_5_4
model.add(TernaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_dw_5_4'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_5_4'))
model.add(Activation(ternary_tanh, name='conv_dw_act_5_4'))

# conv_1x1_5_4
model.add(TernaryConv2D(512, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_5_4'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_5_4'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_5_4'))

# conv_dw_5_5
model.add(TernaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_dw_5_5'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_5_5'))
model.add(Activation(ternary_tanh, name='conv_dw_act_5_5'))

# conv_1x1_5_5
model.add(TernaryConv2D(512, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_5_5'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_5_5'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_5_5'))
##############################################################################################################

# conv_dw_6_1
model.add(TernaryConv2D(512, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',strides=2,
                       padding='same', use_bias=use_bias, name='conv_dw_6_1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_6_1'))
model.add(Activation(ternary_tanh, name='conv_dw_act_6_1'))

# conv_1x1_6_1
model.add(TernaryConv2D(512, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_6_1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_6_1'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_6_1'))
##############################################################################################################

# conv_dw_7_2
model.add(TernaryConv2D(1024, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_dw_7_2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_dw_bn_7_2'))
model.add(Activation(ternary_tanh, name='conv_dw_act_7_2'))

# conv_1x1_7_2
model.add(TernaryConv2D(1024, kernel_size=mini_kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first',
                       padding='same', use_bias=use_bias, name='conv_1x1_7_2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='conv_1x1_bn_7_2'))
model.add(Activation(ternary_tanh, name='conv_1x1_act_7_2'))
##############################################################################################################

#model.add(MaxPooling2D(pool_size=pool_size, name='pool7', data_format='channels_first'))
model.add(Flatten())

##############################################################################################################
# dense1
model.add(TernaryDense(1024, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense5'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn5'))
model.add(Activation(ternary_tanh, name='act5'))
# dense2
model.add(TernaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))
##############################################################################################################

opt = Adam(lr=lr_start)
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
model.summary()

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])
score = model.evaluate(X_test, Y_test, verbose=0)
model.save('cifar10_ter_mobilenet.h5')
print('Test score:', score[0])
print('Test accuracy:', score[1])
