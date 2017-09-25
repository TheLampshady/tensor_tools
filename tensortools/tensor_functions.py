import numpy as np
from os import path
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

DEFAULT_STDDEV = 0.1
DEFAULT_BIAS = 0.1

TRUNC_NORM_INIT = tf.truncated_normal
CONST_INIT = tf.constant


def call_initializer(initializer, shape, params):
    """
    Maps parameters to initializer
    :type initializer: tf.Initializer
    :type shape: list
    :type params: dict
    :return:
    """
    if initializer == TRUNC_NORM_INIT:
        params.setdefault("stddev", DEFAULT_STDDEV)
    elif initializer == tf.constant:
        params.setdefault("value", DEFAULT_BIAS)

    # Remove is get_var scope works
    params.setdefault("shape", shape)
    return initializer(**params)


def variable_summaries(var, histogram_name='histogram'):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :type var: tf.Variable
    :type histogram_name: str
    :rtype: tf.Tensor
    """
    mean = tf.reduce_mean(var)
    mean_scalar = tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    stddev_scalar = tf.summary.scalar('stddev', stddev)
    max_scalar = tf.summary.scalar('max', tf.reduce_max(var))
    min_scalar = tf.summary.scalar('min', tf.reduce_min(var))

    histogram = tf.summary.histogram(histogram_name, var)

    return tf.summary.merge([
        mean_scalar,
        stddev_scalar,
        max_scalar,
        min_scalar,
        histogram
    ])


def summary_variable(name, shape, initializer=TRUNC_NORM_INIT, enable_summary=True, **kwargs):
    """
    Create a weight matrix with appropriate initialization.
    :type name: basestring
    :type shape: list <float>
    :type initializer: tf.Initializer
    :type enable_summary: bool
    :rtype: tf.Variable
    """
    # Requires dynamic scope names
    # variable = tf.get_variable(
    #     name,
    #     shape=shape,
    #     initializer=call_initializer(initializer, kwargs)
    # )
    init = call_initializer(initializer, shape, kwargs)
    variable = tf.Variable(init, name=name)

    with tf.name_scope(name):
        if enable_summary:
            variable_summaries(variable, name)

    return variable


def weight_variable(shape, initializer=TRUNC_NORM_INIT, enable_summary=True, **kwargs):
    """
    Create a weight matrix with appropriate initialization.
    :type shape: list <float>
    :type initializer: func
    :type enable_summary: bool
    :rtype: tf.Variable
    """
    return summary_variable(
        "Weights", shape, initializer, enable_summary, **kwargs
    )


def bias_variable(shape, initializer=CONST_INIT, enable_summary=True, **kwargs):
    """
    Create a bias variable with appropriate initialization.
    :type shape: list <float>
    :type initializer: tf.Initializer
    :type enable_summary: bool
    :rtype: tf.Variable
    """
    return summary_variable(
        "Biases", shape, initializer, enable_summary, **kwargs
    )


def build_metadata(data, filename, headers=None):
    """
    Maps word / index to .tsv
    :type data: list
    :type filename: basestring
    :type headers: tuple
    :rtype: basestring
    """
    if headers:
        data_len = len(data[0]) if isinstance(data[0], (list, tuple)) else 1
        if len(headers) != data_len:
            raise TypeError("Header shape must match data")
        data.insert(0, headers)

    meta_file = path.abspath(filename)
    with open(meta_file, "w") as metadata:
        for row in data:
            value = "\t".join([str(x) for x in row]) \
                if isinstance(row, (list, tuple)) \
                else str(row)
            metadata.write("%s\n" % value)

    return meta_file


def embedding_initializer(layer, embedding_batch, writer, image_shape, sprite_path, label_path):
    # Embedding
    nodes = int(layer.shape[-1])
    embedding = tf.Variable(tf.zeros([embedding_batch, nodes]), name="Embedding")
    assignment = embedding.assign(layer)

    config = projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.metadata_path = label_path

    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.image_path = sprite_path
    embedding_config.sprite.single_image_dim.extend(image_shape)
    projector.visualize_embeddings(writer, config)

    return assignment


def embedding_text(embeddings, writer, label_path):
    """
    Sets up embeddings and metadata
    :type embeddings: tf.Variable
    :type writer: tf.summary.FileWriter
    :type label_path: basestring
    """
    config = projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embeddings.name

    # Link embedding to its metadata file
    embedding_config.metadata_path = label_path
    projector.visualize_embeddings(writer, config)


