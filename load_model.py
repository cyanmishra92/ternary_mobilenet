
'''Trains a simple ternarize CNN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to % test accuracy after 20 epochs using tensorflow backend
'''

from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
import keras.backend as K

from keras.models import load_model
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils import model_to_dot

from keras.utils.vis_utils import plot_model

from keras.utils import CustomObjectScope


K.set_image_data_format('channels_first')

from ternary_ops import ternarize
from ternary_layers import TernaryDense, TernaryConv2D, Clip, ternary_tanh
#from cifar10_mini_mobilenet import ternary_tanh

############## PARAMS ##############
batch_size = 50
epochs = 20
channels = 3
nb_channel = 1
img_rows = 32
img_cols = 32
nb_filters = 32
kernel_size = (3,3)
mini_kernel_size = (1,1)
nb_conv = 3
nb_pool = 2
nb_hid = 128
nb_classes = 10
classes = 10
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
####################################


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.reshape(50000, 3, 32, 32)
X_test = X_test.reshape(10000, 3, 32, 32)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to ternary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1

print('loading model...')

print('loading Full model with classes ...')
model = load_model('cifar10_ter_mini_mobilenet.h5', custom_objects={'TernaryConv2D': TernaryConv2D, 'Clip': Clip, 'ternary_tanh': ternary_tanh, 'TernaryDense': TernaryDense})
print("Model loaded")
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

################# TEST SCORE #################
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
##############################################
