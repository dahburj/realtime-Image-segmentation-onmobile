import cv2
import os
import numpy as np
import pandas as pd
import albumentations as albu
from utils import load_image
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence

seed = 42
np.random.seed(seed)


def augmentation():
    aug = albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20,
                              interpolation=1, border_mode=cv2.BORDER_CONSTANT, p=1),
        albu.RandomBrightnessContrast(p=0.8),
        albu.RandomGamma(p=0.8)])
    return aug


class DataGenerator(Sequence):

    def __init__(self, df, bs, path='./data', input_size=160, is_valid=False):

        self.df = df
        self.bs = bs
        self.path = path
        self.input_size = input_size
        self.is_valid = is_valid
        self.augmentation = augmentation()

    def __len__(self):

        return np.floor(self.df.shape[0] / self.bs).astype(int)

    def on_epoch_end(self):

        if self.is_valid == False:
            self.df = shuffle(self.df, random_state=seed)
            self.df.reset_index(inplace=True, drop=True)

    def __getitem__(self, idx):

        x_batch, y_batch = [], []
        ym_batch = []
        start = idx*self.bs
        end = (idx+1)*self.bs
        image_ids = self.df.image_filename[start:end].values
        mask_ids = self.df.mask_filename[start:end].values
        for i, ids in enumerate(image_ids):

            image = load_image(
                image_ids[i], (self.input_size, self.input_size))
            mask = load_image(
                mask_ids[i], (self.input_size, self.input_size), False)

            # Augmentation
            if not self.is_valid:
                augmented = self.augmentation(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = np.expand_dims(mask, axis=-1)

            x_batch.append(image)
            y_batch.append(mask)

        x_batch = np.array(x_batch, np.float32)
        y_batch = np.array(y_batch, np.float32)

        y_batch = np.concatenate([y_batch, 1-y_batch], axis=-1)

        return x_batch, y_batch
