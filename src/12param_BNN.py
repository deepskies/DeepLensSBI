# Import basic packages
import numpy as np
import scipy as sp
import pandas as pd
import pylab as pl
import scipy.stats
from matplotlib.patches import Ellipse
import pickle
from PIL import Image
from chainconsumer import ChainConsumer

# Tensorflow and Keras
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
# Keras Layers
from keras.layers import InputLayer
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# Import some layers that are useful in the Functional approach
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


# Matplotlib, seaborn and plot pretty
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import seaborn as sns
#%matplotlib inline
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

# Colab in order to download files
#from google.colab import files

# SBI tools
#from sbi import utils, inference
#from sbi import inference
#from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi


# Main simulation class of lenstronomy and deeplenstronomy funcitons
from lenstronomy.Util import util
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.image_util as image_util
from lenstronomy.Data.psf import PSF
import deeplenstronomy.deeplenstronomy as dl
from deeplenstronomy.visualize import view_image


# Scikit-learn for scaling and preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


np.random.seed(42)

file_name_train = "12_model_training_des_500k.pkl"
open_file_train = open(file_name_train, "rb")
dataset_train = pickle.load(open_file_train)
open_file_train.close()

file_name_valid = "12_model_des_valid_100k.pkl"
open_file_valid = open(file_name_valid, "rb")
dataset_valid = pickle.load(open_file_valid)
open_file_valid.close()

file_name_test = "DATA_12_model_des_test.pkl"
open_file_test = open(file_name_test, "rb")
dataset_test = pickle.load(open_file_test)
open_file_test.close()


X_train = dataset_train.CONFIGURATION_1_images
X_val = dataset_valid.CONFIGURATION_1_images
X_test = dataset_test.CONFIGURATION_1_images

X_train = np.moveaxis(X_train,1,-1)
X_val = np.moveaxis(X_val,1,-1)
X_test = np.moveaxis(X_test,1,-1)


metadata_train = dataset_train.CONFIGURATION_1_metadata
metadata_val = dataset_valid.CONFIGURATION_1_metadata
metadata_test = dataset_test.CONFIGURATION_1_metadata

y_train = np.array([metadata_train['PLANE_1-OBJECT_1-MASS_PROFILE_1-theta_E-g'],
                    metadata_train['PLANE_1-OBJECT_1-MASS_PROFILE_1-e1-g'],
                    metadata_train['PLANE_1-OBJECT_1-MASS_PROFILE_1-e2-g'],
                    metadata_train['PLANE_1-OBJECT_1-MASS_PROFILE_1-center_x-g'],
                    metadata_train['PLANE_1-OBJECT_1-MASS_PROFILE_1-center_y-g'],
                    metadata_train['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma1-g'],
                    metadata_train['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma2-g'],
                    metadata_train['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-magnitude-g'],
                    metadata_train['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-R_sersic-g'],
                    metadata_train['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-n_sersic-g'],
                   metadata_train['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e1-g'],
                   metadata_train['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e2-g']])

y_val = np.array([metadata_val['PLANE_1-OBJECT_1-MASS_PROFILE_1-theta_E-g'],
                    metadata_val['PLANE_1-OBJECT_1-MASS_PROFILE_1-e1-g'],
                    metadata_val['PLANE_1-OBJECT_1-MASS_PROFILE_1-e2-g'],
                    metadata_val['PLANE_1-OBJECT_1-MASS_PROFILE_1-center_x-g'],
                    metadata_val['PLANE_1-OBJECT_1-MASS_PROFILE_1-center_y-g'],
                    metadata_val['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma1-g'],
                    metadata_val['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma2-g'],
                    metadata_val['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-magnitude-g'],
                    metadata_val['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-R_sersic-g'],
                    metadata_val['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-n_sersic-g'],
                   metadata_val['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e1-g'],
                   metadata_val['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e2-g']])

y_test = np.array([metadata_test['PLANE_1-OBJECT_1-MASS_PROFILE_1-theta_E-g'],
                    metadata_test['PLANE_1-OBJECT_1-MASS_PROFILE_1-e1-g'],
                    metadata_test['PLANE_1-OBJECT_1-MASS_PROFILE_1-e2-g'],
                    metadata_test['PLANE_1-OBJECT_1-MASS_PROFILE_1-center_x-g'],
                    metadata_test['PLANE_1-OBJECT_1-MASS_PROFILE_1-center_y-g'],
                    metadata_test['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma1-g'],
                    metadata_test['PLANE_1-OBJECT_1-SHEAR_PROFILE_1-gamma2-g'],
                    metadata_test['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-magnitude-g'],
                    metadata_test['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-R_sersic-g'],
                    metadata_test['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-n_sersic-g'],
                   metadata_test['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e1-g'],
                   metadata_test['PLANE_2-OBJECT_1-LIGHT_PROFILE_1-e2-g']])


y_train = np.moveaxis(y_train,0,-1)
y_val = np.moveaxis(y_val,0,-1)
y_test = np.moveaxis(y_test,0,-1)


IMAGE_SHAPE = [32, 32, 1]
NUM_TRAIN_EXAMPLES = 500000
NUM_VAL_EXAMPLES = 50000
NUM_TEST_EXAMPLES = 1000
NUM_CLASSES = 12


tfd = tfp.distributions

# KL divergence weighted by the number of training samples, using
# lambda function to pass as input to the kernel_divergence_fn on
# flipout layers.

kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                           tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))


# Bigger BNN in functional form

model_input = Input(shape=(32,32,1))

# Convolutional part =================
# 1st convolutional chunk
x = tfp.layers.Convolution2DFlipout(
          filters = 16,
          kernel_size=(3,3),
          padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.leaky_relu)(model_input)
x = tfp.layers.Convolution2DFlipout(
          filters = 16,
          kernel_size=(3,3),
          padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.leaky_relu)(model_input)
x = keras.layers.MaxPool2D(pool_size=(2, 2),
                               strides=None,
                               padding='valid')(x)

# 2nd convolutional chunk
x = tfp.layers.Convolution2DFlipout(
          filters = 32,
          kernel_size=(3,3),
          padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.leaky_relu)(x)
x = tfp.layers.Convolution2DFlipout(
          filters = 32,
          kernel_size=(3,3),
          padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.leaky_relu)(x)
x = keras.layers.MaxPool2D(pool_size=(2, 2),
                               strides=None,
                               padding='valid')(x)

# 3rd convolutional chunk
x = tfp.layers.Convolution2DFlipout(
          filters = 48,
          kernel_size=(3,3),
          padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.leaky_relu)(x)
x = tfp.layers.Convolution2DFlipout(
          filters = 48,
          kernel_size=(3,3),
          padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.leaky_relu)(x)
x = keras.layers.MaxPool2D(pool_size=(2, 2),
                               strides=None,
                               padding='valid')(x)

# 4th convolutional chunk
x = tfp.layers.Convolution2DFlipout(
          filters = 64,
          kernel_size=(3,3),
          padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.leaky_relu)(x)
x = tfp.layers.Convolution2DFlipout(
          filters = 64,
          kernel_size=(3,3),
          padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.leaky_relu)(x)
x = tfp.layers.Convolution2DFlipout(
          filters = 64,
          kernel_size=(3,3),
          padding='SAME',
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.leaky_relu)(x)

x = keras.layers.MaxPool2D(pool_size=(2, 2),
                               strides=None,
                               padding='valid')(x)

# ====================================
x = keras.layers.Flatten()(x)
# ====================================

x = tfp.layers.DenseFlipout(
          units = 2048,
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.tanh)(x)
x = tfp.layers.DenseFlipout(
          units = 512,
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.tanh)(x)
x = tfp.layers.DenseFlipout(
          units = 64,
          kernel_divergence_fn=kl_divergence_function,
          activation=tf.nn.tanh)(x)
distribution_params = keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(12))(x)
model_output = tfp.layers.MultivariateNormalTriL(event_size=12)(distribution_params)
model = Model(model_input, model_output)


def negloglik(y_true, y_pred):
    return -tf.reduce_mean(y_pred.log_prob(y_true))

# Define the optimizer
optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.09, rho=0.95)

model.compile(optimizer,
              loss=negloglik,
              metrics=['mae'],experimental_run_tf_function=False)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='12param_500k_50k_500epochs_BNN.h5',
    save_weights_only=True,
    monitor='val_mae',
    mode='min',
    save_best_only=True)

EPOCHS = 500

history = model.fit(x=X_train, y=y_train,
          epochs=EPOCHS, batch_size=256,
          shuffle=True,
          validation_data=(X_val,y_val),
          callbacks=[model_checkpoint_callback])

plt.plot(history.history['loss'][2:])
plt.plot(history.history['val_loss'][2:])
plt.title('model loss')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('12param_model_loss.png')

plt.plot(history.history['mae'][2:])
plt.plot(history.history['val_mae'][2:])
plt.title('model error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('12param_model_error.png')


model.save_weights('12param_500k_50k_500epochs_finalepoch_BNN.h5', overwrite=True)
np.save('history_12param_500k_50k_500epochs_BNN.npy', history.history)
