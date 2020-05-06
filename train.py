import warnings
import ast
import os
import multiprocessing
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from model import segmentation_model
from generator import DataGenerator

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


PATH = "./data"
TRAIN_CSV = "train.csv"
TEST_CSV = "valid.csv"

WEIGHT_FILENAME = "hair_segmentation_mobile.h5"
INPUT_SIZE = 192
BATCH_SIZE = 32


def IOU(y_true, y_pred):

    smooth = 1e-6
    y_true, y_pred = K.backend.reshape(
        y_true, [-1]), K.backend.reshape(y_pred, [-1])
    inter = K.backend.sum(y_pred * y_true) + smooth
    union = K.backend.sum(y_pred+y_true) - inter + smooth
    return inter / union


def dice(y_true, y_pred):

    smooth = 1e-6
    y_true, y_pred = K.backend.reshape(
        y_true, [-1]), K.backend.reshape(y_pred, [-1])
    inter = 2.*K.backend.sum(y_pred * y_true) + smooth
    sum_ = K.backend.sum(y_pred+y_true) + smooth
    return inter / sum_


def bce_dice_loss(y_true, y_pred):

    y_true, y_pred = K.backend.reshape(
        y_true, [-1]), K.backend.reshape(y_pred, [-1])
    dice_loss = 1. - dice(y_true, y_pred)
    bce_loss = K.backend.binary_crossentropy(y_true, y_pred, from_logits=False)
    return 0.6*bce_loss + 0.4*dice_loss


def train():

    # Reading train and test csv file
    train_df = pd.read_csv(os.path.join(PATH, TRAIN_CSV))
    test_df = pd.read_csv(os.path.join(PATH, TEST_CSV))

    print(f"train shape : {train_df.shape} and test shape : {test_df.shape}")

    train_generator = DataGenerator(train_df,
                                    BATCH_SIZE,
                                    input_size=INPUT_SIZE,
                                    path='',
                                    is_valid=False)

    valid_generator = DataGenerator(test_df,
                                    BATCH_SIZE*2,
                                    input_size=INPUT_SIZE,
                                    path='',
                                    is_valid=True)

    # Initialize  Model
    print("Loading Model ...")
    model = segmentation_model(input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    # print(model.summary(110))

    learning_rate = 0.001
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss=bce_dice_loss, metrics=[IOU])

    cbks = [ModelCheckpoint(f"./weights/{WEIGHT_FILENAME}", monitor='val_loss', verbose=1,
                            save_best_only=True, mode='min'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1,
                              mode='min', min_delta=0.0001, min_lr=1e-5),
            EarlyStopping(monitor='val_loss', patience=5, verbose=1,
                          restore_best_weights=False)]

    model.fit(train_generator,
              steps_per_epoch=len(train_generator),
              epochs=50,
              verbose=1,
              callbacks=cbks,
              validation_data=valid_generator,
              validation_steps=len(valid_generator),
              shuffle=False,
              workers=multiprocessing.cpu_count())


if __name__ == "__main__":
    train()
