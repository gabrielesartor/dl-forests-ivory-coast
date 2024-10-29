# import dependencies
import os.path
from os import path
import ee
import geemap
import wxee
import requests
from PIL import Image
import pandas as pd
import numpy as np
import xarray
import rioxarray
import matplotlib.pyplot as plt
from netCDF4 import num2date
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox
import re

from keras.models import *
from keras.metrics import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam_v2 # not used

import pandas as pd

from sklearn.metrics import *
from sklearn.model_selection import train_test_split

print(tf.__version__, flush=True)
print(tf.test.gpu_device_name(), flush=True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')), flush=True)

#################### PATHS DEFINITION ####################

# test_0_xxx : 8.0 - 2.0
# test_1_xxx : 7.0 - 3.0
# test_2_xxx: 6.0 - 4.0
# test_3_xxx: 5.0 - 5.0

dataset_name = "dataset_02_mean"
test_name = "test_1_p1_p99_b32_journal_6bands_2019_2020"
NORMALIZATION = "P1-P99"  # "P1-P99", "Z-SCORE", "MIN-MAX"
class_weight_list = [7.0, 3.0] # weight in the loss for class 0 and 1

at_work_dir = "your_root_dir"
datasets_dir = f"{at_work_dir}/datasets"
results_dir = f"{at_work_dir}/results"

current_dataset_dir = f"{datasets_dir}/{dataset_name}"
current_result_dir = f"{results_dir}/{dataset_name}"
current_test_result_dir = f"{results_dir}/{dataset_name}/{test_name}"
current_model_dir = f"{results_dir}/dataset_02_mean_2019/test_1_p1_p99_b32_journal_6bands"

dataset_sent1_dir = f"{current_dataset_dir}/sentinel1"
download_sent1 = False

dataset_sent2_dir = f"{current_dataset_dir}/sentinel2"
download_sent2 = True

dataset_forest_dir = f"{current_dataset_dir}/forest"
download_forest = True

dataset_sent_cloud_dir = f"{current_dataset_dir}/sentinel2cloud"
download_sent_cloud = False

dataset_noisy_map_dir = f"{current_dataset_dir}/noisy_map"
download_noisy_map = True

# MIN-MAX FACTORS #
min_sent2 = np.load(f"{current_dataset_dir}/norm_factors/min_factors_sent2.npy")
max_sent2 = np.load(f"{current_dataset_dir}/norm_factors/max_factors_sent2.npy")

min_sent1 = np.load(f"{current_dataset_dir}/norm_factors/min_factors_sent1.npy")
max_sent1 = np.load(f"{current_dataset_dir}/norm_factors/max_factors_sent1.npy")

# MEAN-STD FACTORS #
mean_sent2 = np.load(f"{current_dataset_dir}/norm_factors/mean_factors_sent2.npy")
std_sent2 = np.load(f"{current_dataset_dir}/norm_factors/std_factors_sent2.npy")

mean_sent1 = np.load(f"{current_dataset_dir}/norm_factors/mean_factors_sent1.npy")
std_sent1 = np.load(f"{current_dataset_dir}/norm_factors/std_factors_sent1.npy")

# P1-P99 FACTORS #
p1_sent2 = np.load(f"{current_dataset_dir}/norm_factors/p1_factors_sent2.npy")
p99_sent2 = np.load(f"{current_dataset_dir}/norm_factors/p99_factors_sent2.npy")

p1_sent1 = np.load(f"{current_dataset_dir}/norm_factors/p1_factors_sent1.npy")
p99_sent1 = np.load(f"{current_dataset_dir}/norm_factors/p99_factors_sent1.npy")

### 6-BANDS NORM FACTORS ###

# MIN-MAX FACTORS #
min_s2_s1 = np.concatenate([min_sent2,min_sent1])
max_s2_s1 = np.concatenate([max_sent2,max_sent1])

# MEAN-STD FACTORS #
mean_s2_s1 = np.concatenate([mean_sent2,mean_sent1])
std_s2_s1 = np.concatenate([std_sent2,std_sent1])

# P1-P99 FACTORS #
p1_s2_s1 = np.concatenate([p1_sent2,p1_sent1])
p99_s2_s1 =  np.concatenate([p99_sent2,p99_sent1])
#################### PATHS CREATION ####################
if path.exists(datasets_dir) == False: # DATASETS DIR
  os.mkdir(datasets_dir)
  print(f"Created dir: {datasets_dir}")

if path.exists(results_dir) == False: # RESULTS DIR
  os.mkdir(results_dir)
  print(f"Created dir: {results_dir}")

if path.exists(current_dataset_dir) == False: # CURRENT DATASET DIR
  os.mkdir(current_dataset_dir)
  print(f"Created dir: {current_dataset_dir}")

if path.exists(current_result_dir) == False:  # CURRENT RESULTS DIR
  os.mkdir(current_result_dir)
  print(f"Created dir: {current_result_dir}")

if path.exists(current_test_result_dir) == False:  # CURRENT TEST RESULTS DIR
  os.mkdir(current_test_result_dir)
  print(f"Created dir: {current_test_result_dir}")

if path.exists(dataset_sent1_dir) == False:  # SENTINEL1 DIR
  os.mkdir(dataset_sent1_dir)
  print(f"Created dir: {dataset_sent1_dir}")

if path.exists(dataset_sent2_dir) == False:  # SENTINEL2 DIR
  os.mkdir(dataset_sent2_dir)
  print(f"Created dir: {dataset_sent2_dir}")

if path.exists(dataset_forest_dir) == False:  # FNF DIR
  os.mkdir(dataset_forest_dir)
  print(f"Created dir: {dataset_forest_dir}")
########################################

#################### TRAINING PARAMETERS ####################
unet_sent2_model_name = "unet-sent2.hdf5"
unet_sent2_history = "unet-sent2-history.npy"

am_unet_sent2_model_name = "am-unet-sent2.hdf5"
am_unet_sent2_history = "am-unet-sent2-history.npy"

resnet50segnet_sent2_model_name = "resnet50segnet-sent2.hdf5"
resnet50segnet_sent2_history = "resnet50segnet-sent2-history.npy"

fcn32_sent2_model_name = "fcn32-sent2.hdf5"
fcn32_sent2_history = "fcn32-sent2-history.npy"

unet_sent1_model_name = "unet-sent1.hdf5"
unet_sent1_history = "unet-sent1-history.npy"

am_unet_sent1_model_name = "am-unet-sent1.hdf5"
am_unet_sent1_history = "am-unet-sent1-history.npy"

resnet50segnet_sent1_model_name = "resnet50segnet-sent1.hdf5"
resnet50segnet_sent1_history = "resnet50segnet-sent1-history.npy"

fcn32_sent1_model_name = "fcn32-sent1.hdf5"
fcn32_sent1_history = "fcn32-sent1-history.npy"


eps = 30 # epochs
batch_s = 1 # batch size

metrics_list = [TruePositives(), TrueNegatives(), Precision(), Recall(), FalseNegatives(), FalsePositives(), BinaryIoU(), IoU(2, target_class_ids=(0, 1)), BinaryAccuracy(), MeanSquaredError()]

# helper function to perform sort
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

def calculate_sample_weights(label, noisy = None):
  # The weights for each class, with the constraint that:
  #     sum(class_weights) == 1.0
  class_weights = tf.constant(class_weight_list)
  class_weights = class_weights/tf.reduce_sum(class_weights)

  # Create an image of `sample_weights` by using the label at each pixel as an
  # index into the `class weights` .
  sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

  if noisy:
    sample_weights = sample_weights * (1 - noisy) # we want to delete the effect of the noisy
  # print(f"sample_weights: {sample_weights}")

  return sample_weights


#################### DATASET LOADING ####################
NUM_IMGS = 4000

training_images = []
labels_images = []

sent2_images_list = os.listdir(dataset_sent2_dir) # sentinel images names
forest_images_list = os.listdir(dataset_forest_dir) # forest images names

sent2_images_list.sort(key=num_sort)
forest_images_list.sort(key=num_sort)

X_train_list, X_test_val_list, y_train_list, y_test_val_list = train_test_split(sent2_images_list[:NUM_IMGS], forest_images_list[:NUM_IMGS], test_size=0.30, random_state=42)

X_val_list, X_test_list, y_val_list, y_test_list = train_test_split(X_test_val_list, y_test_val_list, test_size=0.5, random_state=42)

# training set creation
X_train = []
y_train = []
w_train = []

invalid_images_list = []

for img_name in X_train_list:
  try:
    # read sentinel-2
    img_xarr_s2 = xarray.open_rasterio(f"{dataset_sent2_dir}/{img_name}")
    img_xarr_s2_t = img_xarr_s2.transpose('y','x','band')

    # read sentinel-1
    img_xarr_s1 = xarray.open_rasterio(f"{dataset_sent1_dir}/{img_name}")
    img_xarr_s1_t = img_xarr_s1.transpose('y','x','band')

    #concatenate
    img_xarr_t = xarray.concat([img_xarr_s2_t, img_xarr_s1_t], 'band')

    # append
    X_train.append(img_xarr_t)
  except Exception as ex:
    print(f"Exception: {ex}... Ignoring img {img_name}")
    invalid_images_list.append(img_name)

for img in invalid_images_list:
  y_train_list.remove(img)
  print(f"Removed {img} from y_train_list")

for img_name in y_train_list:
  img_xarr = xarray.open_rasterio(f"{dataset_forest_dir}/{img_name}")
  img_xarr_t = img_xarr.transpose('y','x','band')
  img_xarr_clf = (img_xarr_t < 3) * 1 # dense and non-dense forest conversion
  y_train.append(img_xarr_clf)

  w = calculate_sample_weights(img_xarr_clf)
  
  w_train.append(w)

# validation set creation
X_val = []
y_val = []
invalid_images_list = []

for img_name in X_val_list:
  try:
    # read sentinel-2
    img_xarr_s2 = xarray.open_rasterio(f"{dataset_sent2_dir}/{img_name}")
    img_xarr_s2_t = img_xarr_s2.transpose('y','x','band')

    # read sentinel-1
    img_xarr_s1 = xarray.open_rasterio(f"{dataset_sent1_dir}/{img_name}")
    img_xarr_s1_t = img_xarr_s1.transpose('y','x','band')

    #concatenate
    img_xarr_t = xarray.concat([img_xarr_s2_t, img_xarr_s1_t], 'band')

    # append
    X_val.append(img_xarr_t)
  except Exception as ex:
    print(f"Exception: {ex}... Ignoring img {img_name}")
    invalid_images_list.append(img_name)

for img in invalid_images_list:
  y_val_list.remove(img)
  print(f"Removed {img} from y_val_list")

for img_name in y_val_list:
  img_xarr = xarray.open_rasterio(f"{dataset_forest_dir}/{img_name}")
  img_xarr_t = img_xarr.transpose('y','x','band')
  img_xarr_clf = (img_xarr_t < 3) * 1 # dense and non-dense forest conversion
  y_val.append(img_xarr_clf)

# test set creation
X_test = []
y_test = []
invalid_images_list = []

for img_name in X_test_list:
  try:
    # read sentinel-2
    img_xarr_s2 = xarray.open_rasterio(f"{dataset_sent2_dir}/{img_name}")
    img_xarr_s2_t = img_xarr_s2.transpose('y','x','band')

    # read sentinel-1
    img_xarr_s1 = xarray.open_rasterio(f"{dataset_sent1_dir}/{img_name}")
    img_xarr_s1_t = img_xarr_s1.transpose('y','x','band')

    #concatenate
    img_xarr_t = xarray.concat([img_xarr_s2_t, img_xarr_s1_t], 'band')

    # append
    X_test.append(img_xarr_t)
  except Exception as ex:
    print(f"Exception: {ex}... Ignoring img {img_name}")
    invalid_images_list.append(img_name)

for img in invalid_images_list:
  y_test_list.remove(img)
  print(f"Removed {img} from y_test_list")

for img_name in y_test_list:
  img_xarr = xarray.open_rasterio(f"{dataset_forest_dir}/{img_name}")
  img_xarr_t = img_xarr.transpose('y','x','band')
  img_xarr_clf = (img_xarr_t < 3) * 1 # dense and non-dense forest conversion
  y_test.append(img_xarr_clf)

print(f"Training set dimensions: X {len(X_train)}, y {len(y_train)}", flush=True)
print(f"Validation set dimensions: X {len(X_val)}, y {len(y_val)}", flush=True)
print(f"Test set dimensions: X {len(X_test)}, y {len(y_test)}", flush=True)

# Normalization (X - min)/(max - min) --> (X - p1) / (p99 - p1)
if NORMALIZATION == "P1-P99":
  print("NORMALIZATION P1-P99", flush=True)
  X_train = (X_train - p1_s2_s1) / (p99_s2_s1 - p1_s2_s1)
  X_val = (X_val - p1_s2_s1) / (p99_s2_s1 - p1_s2_s1)
  X_test = (X_test - p1_s2_s1) / (p99_s2_s1 - p1_s2_s1)

# Normalization zscore = (x -u) / sigma
elif NORMALIZATION == "Z-SCORE":
  print("NORMALIZATION Z-SCORE", flush=True)
  X_train = (X_train - mean_s2_s1) / std_s2_s1
  X_val = (X_val - mean_s2_s1) / std_s2_s1
  X_test = (X_test- mean_s2_s1) / std_s2_s1

# Normalization (x-np.min(x))/(np.max(x)-np.min(x))
else:
  print("NORMALIZATION MIN-MAX", flush=True)
  X_train = (X_train - min_s2_s1) / (max_s2_s1 - min_s2_s1)
  X_val = (X_val - min_s2_s1) / (max_s2_s1 - min_s2_s1)
  X_test = (X_test - min_s2_s1) / (max_s2_s1 - min_s2_s1)

# TensorFlow validation set
X_val_images = np.stack(X_val)
y_val_images = np.stack(y_val)
validation_df = tf.data.Dataset.from_tensor_slices((X_val_images,y_val_images))

#################### MODELS DEFINITION ####################

def trainGenerator(batch_size,
                   image_array,
                   mask_array,
                   weights_array,
                   aug_dict,
                   image_save_prefix  = "image",
                   mask_save_prefix  = "label",
                   weight_save_prefix  = "weight",
                   num_class = 2,
                   save_to_dir = None,
                   target_size = (512,512),
                   seed = 1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    weight_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow(image_array,
                                           batch_size = batch_size,
                                           save_to_dir = save_to_dir,
                                           save_prefix = image_save_prefix,
                                           seed = seed)

    mask_generator = mask_datagen.flow(mask_array,
                                           batch_size = batch_size,
                                           save_to_dir = save_to_dir,
                                           save_prefix = mask_save_prefix,
                                           seed = seed)

    weight_generator = weight_datagen.flow(weights_array,
                                           batch_size = batch_size,
                                           save_to_dir = save_to_dir,
                                           save_prefix = weight_save_prefix,
                                           seed = seed)

    train_generator = zip(image_generator, mask_generator, weight_generator)

    for (img,mask,weight) in train_generator:
        yield (img, mask,weight)

#
# Produce generators for training images
#
X_train_images = np.stack(X_train)
y_train_images = np.stack(y_train)
w_train_images = np.stack(w_train)

# Set parameters for data augmentation
data_gen_args = dict(rotation_range=180,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip = True,
                    fill_mode='reflect',
                    )

'''
  Convolutional block with set parameters and activation layer after
'''

def convBlock(input, filters, kernel, kernel_init='he_normal', act='relu', transpose=False):
  if transpose == False:
    #conv = ZeroPadding2D((1,1))(input)
    conv = Conv2D(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(input)
  else:
    #conv = ZeroPadding2D((1,1))(input)
    conv = Conv2DTranspose(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(input)

  conv = Activation(act)(conv)
  return conv

'''
  U-Net model
'''

def UNet(trained_weights = None, input_size = (512,512,4), drop_rate = 0.25, lr=0.0001):

    ## Can add pretrained weights by specifying 'trained_weights'

    # Input layer
    # inputs = Input(input_size, batch_size=1)
    inputs = Input(input_size)


    ## Contraction phase
    # conv1 = convBlock(conc_0, 64, 3)
    conv1 = convBlock(inputs, 64, 3)
    conv1 = convBlock(conv1, 64, 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = convBlock(pool1, 128, 3)
    conv2 = convBlock(conv2, 128, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #drop2 = Dropout(drop_rate)(pool2)

    conv3 = convBlock(pool2, 256, 3)
    conv3 = convBlock(conv3, 256, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #drop3 = Dropout(drop_rate)(pool3)

    conv4 = convBlock(pool3, 512, 3)
    conv4 = convBlock(conv4, 512, 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #drop4 = Dropout(drop_rate)(pool4)

    conv5 = convBlock(pool4, 1024, 3)
    conv5 = convBlock(conv5, 1024, 3)

    ## Expansion phase
    up6 = (Conv2DTranspose(512, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv5))
    merge6 = concatenate([conv4,up6])
    conv6 = convBlock(merge6, 512, 3)
    conv6 = convBlock(conv6, 512, 3)
    #conv6 = Dropout(drop_rate)(conv6)

    up7 = (Conv2DTranspose(256, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv6))
    merge7 = concatenate([conv3,up7])
    conv7 = convBlock(merge7, 256, 3)
    conv7 = convBlock(conv7, 256, 3)
    #conv7 = Dropout(drop_rate)(conv7)

    up8 = (Conv2DTranspose(128, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv7))
    merge8 = concatenate([conv2,up8])
    conv8 = convBlock(merge8, 128, 3)
    conv8 = convBlock(conv8, 128, 3)
    #conv8 = Dropout(drop_rate)(conv8)

    up9 = (Conv2DTranspose(64, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv8))
    merge9 = concatenate([conv1,up9])
    conv9 = convBlock(merge9, 64, 3)
    conv9 = convBlock(conv9, 64, 3)

    # Output layer
    conv10 = convBlock(conv9, 1, 1, act='sigmoid')
    # Output layer beta

    model = Model(inputs, conv10)

    model.compile(optimizer = adam_v2.Adam(learning_rate = lr), loss = 'binary_crossentropy', metrics = metrics_list)

    if trained_weights != None:
    	model.load_weights(trained_weights)

    return model

# create resnet, unet, etc...

'''
  Convolutional block with two conv layers and two activation layers
'''

def convBlock2(input, filters, kernel, kernel_init='he_normal', act='relu', transpose=False):
  if transpose == False:
    conv = Conv2D(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(input)
    conv = Activation(act)(conv)
    conv = Conv2D(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(conv)
    conv = Activation(act)(conv)
  else:
    conv = Conv2DTranspose(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(input)
    conv = Activation(act)(conv)
    conv = Conv2DTranspose(filters, kernel, padding = 'same', kernel_initializer = kernel_init)(conv)
    conv = Activation(act)(conv)

  return conv

'''
  Attention block/mechanism
'''
def attention_block(x, gating, inter_shape, drop_rate=0.25):

    # Find shape of inputs
    shape_x = K.int_shape(x)
    print(shape_x)
    shape_g = K.int_shape(gating)
    print(shape_g)

    ## Process x vector and gating signal
    # x vector input and processing
    theta_x = Conv2D(inter_shape, kernel_size = 1, strides = 1, padding='same', kernel_initializer='he_normal', activation=None)(x)
    theta_x = MaxPooling2D((2,2))(theta_x)
    shape_theta_x = K.int_shape(theta_x)

    # gating signal ""
    phi_g = Conv2D(inter_shape, kernel_size = 1, strides = 1, padding='same', kernel_initializer='he_normal', activation=None)(gating)
    shape_phi_g = K.int_shape(phi_g)

    # Add components
    concat_xg = add([phi_g, theta_x])
    act_xg = Activation('relu')(concat_xg)

    # Apply convolution
    psi = Conv2D(1, kernel_size = 1, strides = 1, padding='same', kernel_initializer='he_normal', activation=None)(act_xg)

    # Apply sigmoid activation
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)

    # UpSample and resample to correct size
    upsample_psi = UpSampling2D(interpolation='bilinear', size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
    upsample_psi = tf.broadcast_to(upsample_psi, shape=(K.shape(x)[0], shape_x[1], shape_x[2], shape_x[3]) )
    y = multiply([upsample_psi, x])

    return y


'''
  Attention U-Net model
'''

def UNetAM(trained_weights = None, input_size = (512,512,4), drop_rate = 0.25, lr=0.0001, filter_base=16, batch_size = 1):

    ## Can add pretrained weights by specifying 'trained_weights'

    # Input layer
    inputs = Input(input_size)

    ## Contraction phase
    conv = convBlock2(inputs, filter_base, 3)
    #conv0 = Dropout(drop_rate)(conv0)

    conv0 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv0 = convBlock2(conv0, 2 * filter_base, 3)

    pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
    conv1 = convBlock2(pool0, 4 * filter_base, 3)
    #conv1 = Dropout(drop_rate)(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = convBlock2(pool1, 8 * filter_base, 3)
    #conv2 = Dropout(drop_rate)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = convBlock2(pool2, 16 * filter_base, 3)
    #conv3 = Dropout(drop_rate)(conv3)

    ## Expansion phase
    up4 = (Conv2DTranspose(8 * filter_base, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv3))
    # up4 = tf.reshape(up4, (conv3.shape[0], conv3.shape[1]*2, conv3.shape[2]*2, int(conv3.shape[3]/2) ) )
    merge4 = attention_block(conv2, conv3, 8 * filter_base, drop_rate) # Attention gate
    conv4 = concatenate([up4, merge4])
    conv4 = convBlock2(conv4, 8 * filter_base, 3)

    up5 = (Conv2DTranspose(4 * filter_base, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv4))
    # up5 = tf.reshape(up5, (conv4.shape[0], conv4.shape[1]*2, conv4.shape[2]*2, int(conv4.shape[3]/2) ) )
    merge5 = attention_block(conv1, conv4, 4 * filter_base, drop_rate) # Attention gate
    conv5 = concatenate([up5, merge5])
    conv5 = convBlock2(conv5, 4 * filter_base, 3)

    up6 = (Conv2DTranspose(2 * filter_base, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv5))
    # up6 = tf.reshape(up6, (conv5.shape[0], conv5.shape[1]*2, conv5.shape[2]*2, int(conv5.shape[3]/2) ) )
    merge6 = attention_block(conv0, conv5, 2 * filter_base, drop_rate) # Attention gate
    conv6 = concatenate([up6, merge6])
    conv6 = convBlock2(conv6, 2 * filter_base, 3)

    up7 = (Conv2DTranspose(1 * filter_base, kernel_size=2, strides=2, kernel_initializer='he_normal')(conv6))
    # up7 = tf.reshape(up7, (conv6.shape[0], conv6.shape[1]*2, conv6.shape[2]*2, int(conv6.shape[3]/2) ) )
    merge7 = attention_block(conv, conv6, 1 * filter_base, drop_rate) # Attention gate
    conv7 = concatenate([up7, merge7])
    conv7 = concatenate([up7, conv])
    conv7 = convBlock2(conv7, 1 * filter_base, 3)

    ## Output layer
    out = convBlock(conv7, 1, 1, act='sigmoid')
    # out = tf.reshape(out, out.shape)

    model = Model(inputs, out)

    model.compile(optimizer = adam_v2.Adam(learning_rate = lr), loss = binary_crossentropy, metrics = metrics_list)

    if trained_weights != None:
    	model.load_weights(trained_weights)

    return model

# Forked code from: https://github.com/ykamikawa/tf-keras-SegNet

from keras.layers import Layer

'''
  Unpooling using max pooling indices
'''

class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        padding = 'same'
        pool_size = (2,2)
        strides = (2,2)
        if K.backend() == "tensorflow":
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        else:
            errmsg = "{} backend is not supported for layer {}".format(
                K.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]
    
    # added on cineca
    def get_config(self):
      cfg = super().get_config()
      return cfg

class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope(self.name):
            mask = K.cast(mask, "int32")
            input_shape = K.tf.shape(updates, out_type="int32")
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3],
                )
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype="int32")
            batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype="int32"), shape=batch_shape
            )
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype="int32")
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3],
        )

    def get_config(self):
      cfg = super().get_config()
      return cfg

# Custom version of MaxUnpooling2D
# Takes raw layer values and outputs values
# Takes tf.nn.max_pool_with_argmax output as input
def unpool_with_indices(pool, indices, out_size=2):
  print(pool)
  print(indices)
  # Create empty array of appropriate size
  shape = np.array(np.shape(pool))
  shape = np.array((shape[0], out_size * shape[1], out_size * shape[2], shape[3]))
  out = np.zeros(shape)

  # Make upsample
  inds = np.array(indices).flatten()
  outs = np.array(pool).flatten()
  for i in range(len(inds)):
    blk = inds[i] // (shape[2] * shape[3]) # Find which block to place numbers in
    ln  = inds[i] - (blk * shape[3] * shape[2]) # Find which line
    ln2 = ln // (shape[3]) # Find line
    pos = ln % (shape[3]) # Find position
    #print(blk, ln2, pos)
    out[0][blk][ln2][pos] = outs[i]


  #print(out.shape)
  return (out)

# Own custom code
'''
  ResNet Contraction Phase Block
'''

def resnetConvDownBlock(x, filter, kernel, act='relu'):
  # Convolutional Block for encoding phase
  for i in range(3):
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(filters = filter, kernel_size = kernel, kernel_initializer = 'he_normal')(x)
    x = Activation('relu')(x)

  return x

'''
  SegNet Expansion Phase Block
'''
def resnetConvUpBlock(x, skip_connection = None, filter = None, kernel = None, act='relu'):
  # Convolutional block for decoding phase

  out = x

  # Unpooling
  out = UpSampling2D((2,2))(out)

  # Conv Block
  for i in range(3):
    out = ZeroPadding2D((1,1))(out)
    out = Conv2D(filters = filter, kernel_size = kernel, kernel_initializer = 'he_normal')(out)
    out = Activation('relu')(out)

  # Implement skip connection
  if skip_connection != None:
    out = Add()([out, skip_connection])

  return out

def ResNet50SegNet(input_size=(512,512,4), lr = 0.0001, filters = 64, kernel_sz = 3):

  inputs = Input(input_size)

  # Encoder
  # Conv, Conv, Conv, MaxPool #1
  block1 = resnetConvDownBlock(inputs, filter = filters, kernel = kernel_sz)
  pool1, mask1 = MaxPoolingWithArgmax2D((2,2))(block1)
  # Conv, Conv, Conv, MaxPool #2
  block2 = resnetConvDownBlock(pool1, filter = 2 * filters, kernel = kernel_sz)
  pool2, mask2 = MaxPoolingWithArgmax2D((2,2))(block2)
  # Conv, Conv, Conv, MaxPool #3
  block3 = resnetConvDownBlock(pool2, filter = 4 * filters, kernel = kernel_sz)
  pool3, mask3 = MaxPoolingWithArgmax2D((2,2))(block3)
  # Conv, Conv, Conv, MaxPool #4
  block4 = resnetConvDownBlock(pool3, filter = 8 * filters, kernel = kernel_sz)
  pool4, mask4 = MaxPoolingWithArgmax2D((2,2))(block4)
  # Conv, Conv, Conv, MaxPool #5
  block5 = resnetConvDownBlock(pool4, filter = 16 * filters, kernel = kernel_sz)
  pool5, mask5 = MaxPoolingWithArgmax2D((2,2))(block5)

  # Decoder
  # ConvTranspose + Concat, Conv, Conv, Conv #1
  block5_ = resnetConvUpBlock(pool5, filter = 16 * filters, kernel = kernel_sz)
  # ConvTranspose + Concat, Conv, Conv, Conv #2
  block4_ = resnetConvUpBlock(block5_, skip_connection = MaxUnpooling2D((2,2))([pool4, mask4]), filter = 8 * filters, kernel = kernel_sz)
  # ConvTranspose + Concat, Conv, Conv, Conv #3
  block3_ = resnetConvUpBlock(block4_, skip_connection = MaxUnpooling2D((2,2))([pool3, mask3]), filter = 4 * filters, kernel = kernel_sz)
  # ConvTranspose + Concat, Conv, Conv, Conv #4
  block2_ = resnetConvUpBlock(block3_, skip_connection = MaxUnpooling2D((2,2))([pool2, mask2]), filter = 2 * filters, kernel = kernel_sz)
  # ConvTranspose + Concat, Conv, Conv, Conv #5
  block1_ = resnetConvUpBlock(block2_, skip_connection = MaxUnpooling2D((2,2))([pool1, mask1]), filter = filters, kernel = kernel_sz)

  # Output
  outputs = Conv2D(1, kernel_size = 1, strides = 1, kernel_initializer = 'he_normal')(block1_)
  outputs = Activation('sigmoid')(outputs)

  model = Model(inputs, outputs)
  model.compile(optimizer = adam_v2.Adam(learning_rate = lr), loss = binary_crossentropy, metrics = metrics_list)

  return model

# Code forked and modified from: https://github.com/divamgupta/image-segmentation-keras

'''
  FCN32-VGG16 model
'''

def fcn_32(input_size = (256,256,4), lr = 0.0001, drop_rate = 0):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    IMAGE_ORDERING = 'channels_last'
    # Input
    inputs = Input(shape=input_size)

    x = inputs
    levels = []

    ## Encoder
    # Block 1
    x = Conv2D(64, (3, 3), padding='same',
               name='block1_conv1', data_format=IMAGE_ORDERING)(inputs)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same',
               name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                     data_format=IMAGE_ORDERING)(x)
    levels.append(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same',
               name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same',
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                     data_format=IMAGE_ORDERING)(x)
    levels.append(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same',
               name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same',
               name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same',
               name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                     data_format=IMAGE_ORDERING)(x)
    levels.append(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same',
               name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same',
               name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same',
               name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',
                     data_format=IMAGE_ORDERING)(x)
    levels.append(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same',
               name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same',
               name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), padding='same',
               name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool',
                     data_format=IMAGE_ORDERING)(x)

    levels.append(x)

    [f1, f2, f3, f4, f5] = levels

    o = f5

    # Decoder
    o = (Conv2D(4096, (7 , 7 ), padding = 'same', kernel_initializer = 'he_normal', name = "conv6"))(o)
    o = Activation('relu')(o)
    o = Dropout(drop_rate)(o)
    o = (Conv2D(4096, (1 , 1 ), padding = 'same', kernel_initializer = 'he_normal', name = "conv7"))(o)
    o = Activation('relu')(o)
    o = Dropout(drop_rate)(o)

    o = (Conv2D(1, 1, padding='same', kernel_initializer='he_normal', name="scorer1"))(o)
    o = Conv2DTranspose(1, kernel_size=(64,64), padding='same', strides=(32,32), name="Upsample32")(o)
    o = (Conv2D(1, 1, padding='same', kernel_initializer='he_normal', name="output"))(o)

    # Output
    o = Activation('sigmoid')(o)

    model = Model(inputs, o)
    model.compile(optimizer = adam_v2.Adam(learning_rate = lr), loss = binary_crossentropy, metrics = metrics_list)
    model.model_name = "fcn_32"
    return model


########################################

"""## TEST SET ANALYSIS (SENTINEL 2)"""
def compute_accuracy(model, images, labels):
  accuracy_list = []
  for i in range(len(images)):
    y_hat = model.predict(images[i].reshape(1,256,256,6)) > 0.5
    y_hat = np.round(y_hat).flatten()
    acc = accuracy_score(labels[i].flatten(), y_hat)
    accuracy_list.append(acc)
  return np.mean(accuracy_list)

def compute_precision(model, images, labels):
  precision_list = []
  for i in range(len(images)):
    y_hat = model.predict(images[i].reshape(1,256,256,6)) > 0.5
    y_hat = np.round(y_hat).flatten()
    acc = precision_score(labels[i].flatten(), y_hat, average='binary')
    precision_list.append(acc)
  return np.mean(precision_list)

def compute_recall(model, images, labels):
  recall_list = []
  for i in range(len(images)):
    y_hat = model.predict(images[i].reshape(1,256,256,6)) > 0.5
    y_hat = np.round(y_hat).flatten()
    acc = recall_score(labels[i].flatten(), y_hat, average='binary')
    recall_list.append(acc)
  return np.mean(recall_list)

def compute_recall_negative(model, images, labels):
  recall_list = []
  for i in range(len(images)):
    y_hat = model.predict(images[i].reshape(1,256,256,6)) > 0.5
    y_hat = np.round(y_hat).flatten()
    acc = recall_score(labels[i].flatten(), y_hat, pos_label=0, average='binary')
    recall_list.append(acc)
  return np.mean(recall_list)

def compute_recall_micro(model, images, labels):
  recall_list = []
  for i in range(len(images)):
    y_hat = model.predict(images[i].reshape(1,256,256,6)) > 0.5
    y_hat = np.round(y_hat).flatten()
    acc = recall_score(labels[i].flatten(), y_hat, average='micro')
    recall_list.append(acc)
  return np.mean(recall_list)

def compute_recall_macro(model, images, labels):
  recall_list = []
  for i in range(len(images)):
    y_hat = model.predict(images[i].reshape(1,256,256,6)) > 0.5
    y_hat = np.round(y_hat).flatten()
    acc = recall_score(labels[i].flatten(), y_hat, average='macro')
    recall_list.append(acc)
  return np.mean(recall_list)

def compute_f1_score(model, images, labels):
  recall_list = []
  for i in range(len(images)):
    y_hat = model.predict(images[i].reshape(1,256,256,6)) > 0.5
    y_hat = np.round(y_hat).flatten()
    acc = f1_score(labels[i].flatten(), y_hat)
    recall_list.append(acc)
  return np.mean(recall_list)

# new
def compute_auc_score(model, images, labels):
  auc_list = []
  for i in range(len(images)):
    y_hat = model.predict(images[i].reshape(1,256,256,6)) > 0.5
    y_hat = np.round(y_hat).flatten()
    acc = None
    try:
      acc = roc_auc_score(labels[i].flatten(), y_hat)
    except Exception:
      acc = None
    if acc:
      auc_list.append(acc)
  return np.mean(auc_list)

def compute_average_precision_micro(model, images, labels):
  avg_precision_list = []
  for i in range(len(images)):
    y_hat = model.predict(images[i].reshape(1,256,256,6)) > 0.5
    y_hat = np.round(y_hat).flatten()
    acc = average_precision_score(labels[i].flatten(), y_hat, average='micro')
    avg_precision_list.append(acc)
  return np.mean(avg_precision_list)

def compute_average_precision_macro(model, images, labels):
  avg_precision_list = []
  for i in range(len(images)):
    y_hat = model.predict(images[i].reshape(1,256,256,6)) > 0.5
    y_hat = np.round(y_hat).flatten()
    acc = average_precision_score(labels[i].flatten(), y_hat, average='macro')
    avg_precision_list.append(acc)
  return np.mean(avg_precision_list)

# load models
unet = load_model(f'{current_model_dir}/{unet_sent2_model_name}', compile=False)
attention_unet = load_model(f'{current_model_dir}/{am_unet_sent2_model_name}', compile=False)
resnet_segnet = ResNet50SegNet(input_size=(256,256,6))
resnet_segnet.load_weights(f'{current_model_dir}/{resnet50segnet_sent2_model_name}')
fcn32 = load_model(f'{current_model_dir}/{fcn32_sent2_model_name}', compile=False)

# Load dataset again!?
X_test_images = np.stack(X_test)
y_test_images = np.stack(y_test)

# calculate accuracy
unet_accuracy = compute_accuracy(unet, X_test_images, y_test_images)
unet_precision = compute_precision(unet, X_test_images, y_test_images)
unet_recall = compute_recall(unet, X_test_images, y_test_images)
unet_f1_score = compute_f1_score(unet, X_test_images, y_test_images)
unet_recall_negative = compute_recall_negative(unet, X_test_images, y_test_images)
unet_recall_micro = compute_recall_micro(unet, X_test_images, y_test_images)
unet_recall_macro = compute_recall_macro(unet, X_test_images, y_test_images)

unet_auc = compute_auc_score(unet, X_test_images, y_test_images)
unet_precision_micro = compute_average_precision_micro(unet, X_test_images, y_test_images)
unet_precision_macro = compute_average_precision_macro(unet, X_test_images, y_test_images)

print(f"UNet --> accuracy: {unet_accuracy} - precision: {unet_precision} - recall: {unet_recall} - f1 score: {unet_f1_score}")
print(f"UNet --> recall negative: {unet_recall_negative} - recall micro: {unet_recall_micro} - recall macro: {unet_recall_macro}")
print(f"UNet --> auc: {unet_auc} - precision micro: {unet_precision_micro} - precision macro: {unet_precision_macro}")

am_unet_accuracy = compute_accuracy(attention_unet, X_test_images, y_test_images)
am_unet_precision = compute_precision(attention_unet, X_test_images, y_test_images)
am_unet_recall = compute_recall(attention_unet, X_test_images, y_test_images)
am_unet_f1_score = compute_f1_score(attention_unet, X_test_images, y_test_images)
am_unet_recall_negative = compute_recall_negative(attention_unet, X_test_images, y_test_images)
am_unet_recall_micro = compute_recall_micro(attention_unet, X_test_images, y_test_images)
am_unet_recall_macro = compute_recall_macro(attention_unet, X_test_images, y_test_images)

am_unet_auc = compute_auc_score(attention_unet, X_test_images, y_test_images)
am_unet_precision_micro = compute_average_precision_micro(attention_unet, X_test_images, y_test_images)
am_unet_precision_macro = compute_average_precision_macro(attention_unet, X_test_images, y_test_images)

print(f"AM-UNet --> accuracy: {am_unet_accuracy} - precision: {am_unet_precision} - recall: {am_unet_recall} - f1 score: {am_unet_f1_score}")
print(f"AM-UNet --> recall negative: {am_unet_recall_negative} - recall micro: {am_unet_recall_micro} - recall macro: {am_unet_recall_macro}")
print(f"AM-UNet --> auc: {am_unet_auc} - precision micro: {am_unet_precision_micro} - precision macro: {am_unet_precision_macro}")

resnet_segnet_accuracy = compute_accuracy(resnet_segnet, X_test_images, y_test_images)
resnet_segnet_precision = compute_precision(resnet_segnet, X_test_images, y_test_images)
resnet_segnet_recall = compute_recall(resnet_segnet, X_test_images, y_test_images)
resnet_segnet_f1_score = compute_f1_score(resnet_segnet, X_test_images, y_test_images)
resnet_segnet_recall_negative = compute_recall_negative(resnet_segnet, X_test_images, y_test_images)
resnet_segnet_recall_micro = compute_recall_micro(resnet_segnet, X_test_images, y_test_images)
resnet_segnet_recall_macro = compute_recall_macro(resnet_segnet, X_test_images, y_test_images)

resnet_segnet_auc = compute_auc_score(resnet_segnet, X_test_images, y_test_images)
resnet_segnet_precision_micro = compute_average_precision_micro(resnet_segnet, X_test_images, y_test_images)
resnet_segnet_precision_macro = compute_average_precision_macro(resnet_segnet, X_test_images, y_test_images)

print(f"Resnet --> accuracy: {resnet_segnet_accuracy} - precision: {resnet_segnet_precision} - recall: {resnet_segnet_recall} - f1 score: {resnet_segnet_f1_score}")
print(f"Resnet --> recall negative: {resnet_segnet_recall_negative} - recall micro: {resnet_segnet_recall_micro} - recall macro: {resnet_segnet_recall_macro}")
print(f"Resnet --> auc: {resnet_segnet_auc} - precision micro: {resnet_segnet_precision_micro} - precision macro: {resnet_segnet_precision_macro}")

fcn32_accuracy = compute_accuracy(fcn32, X_test_images, y_test_images)
fcn32_precision = compute_precision(fcn32, X_test_images, y_test_images)
fcn32_recall = compute_recall(fcn32, X_test_images, y_test_images)
fcn32_f1_score = compute_f1_score(fcn32, X_test_images, y_test_images)
fcn32_recall_negative = compute_recall_negative(fcn32, X_test_images, y_test_images)
fcn32_recall_micro = compute_recall_micro(fcn32, X_test_images, y_test_images)
fcn32_recall_macro = compute_recall_macro(fcn32, X_test_images, y_test_images)

fcn32_auc = compute_auc_score(fcn32, X_test_images, y_test_images)
fcn32_precision_micro = compute_average_precision_micro(fcn32, X_test_images, y_test_images)
fcn32_precision_macro = compute_average_precision_macro(fcn32, X_test_images, y_test_images)

print(f"FCN32 --> accuracy: {fcn32_accuracy} - precision: {fcn32_precision} - recall: {fcn32_recall} - f1 score: {fcn32_f1_score}")
print(f"FCN32 --> recall negative: {fcn32_recall_negative} - recall micro: {fcn32_recall_micro} - recall macro: {fcn32_recall_macro}")
print(f"FCN32 --> auc: {fcn32_auc} - precision micro: {fcn32_precision_micro} - precision macro: {fcn32_precision_macro}")

# Export results
metrics_sent2 = {'classifier': ['U-Net', 'Attention U-Net', 'ResNet50-SegNet', 'FCN32-VGG16'],
              'accuracy': [unet_accuracy, am_unet_accuracy, resnet_segnet_accuracy, fcn32_accuracy],
              'precision': [unet_precision, am_unet_precision, resnet_segnet_precision, fcn32_precision],
              'recall': [unet_recall, am_unet_recall, resnet_segnet_recall, fcn32_recall],
              'recall_negative': [unet_recall_negative, am_unet_recall_negative, resnet_segnet_recall_negative, fcn32_recall_negative],
              'recall_micro': [unet_recall_micro, am_unet_recall_micro, resnet_segnet_recall_micro, fcn32_recall_micro],
              'recall_macro': [unet_recall_macro, am_unet_recall_macro, resnet_segnet_recall_macro, fcn32_recall_macro],
              'f1_score': [unet_f1_score, am_unet_f1_score, resnet_segnet_f1_score, fcn32_f1_score],
              'auc': [unet_auc, am_unet_auc, resnet_segnet_auc, fcn32_auc],
              'precision_micro': [unet_precision_micro, am_unet_precision_micro, resnet_segnet_precision_micro, fcn32_precision_micro],
              'precision_macro': [unet_precision_macro, am_unet_precision_macro, resnet_segnet_precision_macro, fcn32_precision_macro],
              }
metrics_sent2_pd = pd.DataFrame(metrics_sent2)
metrics_sent2_pd.to_csv(f'{current_test_result_dir}/metrics_s2_s1.csv')
print(metrics_sent2_pd)
