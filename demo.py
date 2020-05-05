import cv2
import os
import argparse
import requests
import errno
import numpy as np
import tensorflow.keras as k
from utils import apply_color, resize_mask
from PIL import Image
from io import BytesIO
import urllib

INPUT_SHAPE = (160, 160)


def parse_args():
    # Parse input arguments
    parser = argparse.ArgumentParser(
        description="Convert keras model to tflite")

    parser.add_argument('--model_path',
                        help='name of the h5 file',
                        default="./weights/hair_segmentation_mobile.h5",
                        type=str)

    parser.add_argument("--image_path",
                        help="path of test image",
                        default="",
                        type=str)

    # parser.add_argument("--image_url",
    #                     help="url of test image",
    #                     default="",
    #                     type=str)

    parser.add_argument('--color',
                        help='rgb color value',
                        default=[255, 0, 0],
                        type=int, nargs='+')

    parser.add_argument('--output_path',
                        help='name of the output image file to be saved',
                        default="./output.jpg",
                        type=str)

    return parser.parse_args()


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.array(image/255., dtype=np.float32)

# def load_from_url(url):

#     r = requests.get(url, stream=True)
#     image = Image.open(BytesIO(r.content))
#     image = np.array(image, dtype=np.float32)/255.
#     return image


if __name__ == "__main__":

    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), args.model_path)

    # if args.image_url == '' and args.image_path == '':
    #     raise FileNotFoundError

    model_path = args.model_path
    image_path = args.image_path
    # image_url = args.image_urlapply_color
    color = args.color

    model = k.models.load_model(model_path, compile=False)

    if image_path != '':
        image = load_image(image_path)

    # if image_url != '':
    #     image = load_from_url(image_url)
    #     image = cv2.resize(image, INPUT_SHAPE)

    input_image = cv2.resize(image, INPUT_SHAPE)
    input_image = np.expand_dims(input_image, axis=0)
    prediction = model.predict(input_image)[..., 0:1]

    prediction = apply_color(image, prediction[0], color)

    prediction = (prediction*255).astype(np.uint8)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)

    print(f"Saving output file at {args.output_path}")
    cv2.imwrite(args.output_path, prediction)
