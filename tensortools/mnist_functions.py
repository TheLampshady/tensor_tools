from os import path

import numpy as np
import scipy.misc as misc
from .tensor_functions import build_metadata


def build_mnist_sprite(sprite_images, filename):
    """

    :type sprite_images: list: 2D array of singe channel pixel values
    :type filename: str: Converted to absolute (required by Tensorboard for some versions)
    :return:
    """
    sprite_file = path.abspath(filename)
    x = None
    res = None
    for i in range(32):
        x = None
        for j in range(32):
            img = sprite_images[i * 32 + j, :].reshape((28, 28))
            x = np.concatenate((x, img), axis=1) if x is not None else img
        res = np.concatenate((res, x), axis=0) if res is not None else x

    misc.toimage(256 - res, channel_axis=0).save(sprite_file)

    return sprite_file


def one_hot_to_array(labels):
    return [int(np.where(target == 1)[0]) for target in labels]


def build_mnist_embeddings(data_path, mnist):
    images = mnist.test.images[:1024]
    labels = mnist.test.labels[:1024]
    meta_data = one_hot_to_array(labels)

    sprite_path = build_mnist_sprite(images, path.join(data_path, 'sprite_1024.png'))
    label_path = build_metadata(meta_data, path.join(data_path, 'labels_1024.tsv'))

    return sprite_path, label_path
