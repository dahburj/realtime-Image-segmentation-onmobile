import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def load_image(path, size=(None, None), rgb=True):
    """
    Parameters
    ----------
    path : path of the image
    size : size in which image is to be resized
    rgb : flag to convert image  to RGB format
    """
    image = cv2.imread(path)
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w, h = size
    if h and w:
        image = cv2.resize(image, (w, h))

    return np.array(image/255., dtype=np.float32)


def resize_mask(image, mask):
    mask = cv2.resize(mask, image.shape[:2][::-1])
    return np.expand_dims(mask, -1)


def apply_color(image, mask, color_rgb=[255, 0, 0]):
    color = np.zeros(image.shape)
    color[..., 0], color[..., 1], color[..., 2] = color_rgb
    color = color/255.
    if mask.shape[:2] != image.shape[:2]:
        mask = resize_mask(image, mask)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, -1)
    background = image*(1-mask)
    foreground = (mask*image)*color_rgb
    return np.clip(foreground + background, 0, 1)


def display_image(im, figsize=None, ax=None, alpha=None, cmap=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha, cmap=cmap)
    ax.set_axis_off()
    return ax
