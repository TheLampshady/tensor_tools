import tensorflow as tf

from .tensor_functions import bias_variable, weight_variable


def get_ann_layer(layer, node_size, keep_prob=None, activate=True, enable_summary=True):
    """
    Artificial Neural Network Layer
    Example: get_artificial_layer(input_x, filters[i] + channels[i:i + 2])
    :param hidden_size: Dimension for layer
    :param activate: For running activation function
    :param enable_summary: For tensorboard

    :type layer: tf.Placeholder
    :type hidden_size: int
    :type keep_prob: tf.Placeholder
    :type activate: bool
    :type enable_summary: bool
    :rtype: tf.Tensor
    """
    with tf.name_scope('ANN_Layer'):
        weights = weight_variable([int(layer.shape[-1]), node_size])
        biases = bias_variable([node_size])

        with tf.name_scope('Wx_plus_b'):
            # Pre-Activation
            preactivate = tf.nn.xw_plus_b(layer, weights, biases)
            if enable_summary:
                tf.summary.histogram('Pre_Activations', preactivate)
            if not activate:
                return preactivate

            # Activation
            activations = tf.nn.relu(preactivate)
            if enable_summary:
                tf.summary.histogram('Activations', activations)
            if not keep_prob:
                return activations

            # Dropout
            return tf.nn.dropout(activations, keep_prob, name="Dropout")


def get_convolution_layer(layer, shape, strides=1, enable_summary=True):
    """
    Example: get_convolution_layer(input_x, filters[i] + channels[i:i + 2])
    :param layer:
    :param shape:
    :param strides:
    :return:
    """
    with tf.name_scope('Layer'):
        weights = weight_variable(shape)
        biases = bias_variable(shape[-1:])

        with tf.name_scope('Wx_plus_b'):
            conv_layer = tf.nn.conv2d(
                layer, weights,
                strides=[1, strides, strides, 1],
                padding='SAME'
            )
            preactivate = conv_layer + biases
            if enable_summary:
                tf.summary.histogram('Pre_Activations', preactivate)
            activations = tf.nn.relu(preactivate)
            if enable_summary:
                tf.summary.histogram('Activations', activations)
            return activations


def get_recurrent_layer_full(layer, shape, batch_size, num_classes, keep_prob=1.0):
    """
    Example: get_recurrent_layer(x, [element_size, hidden_layer_size], 0.9)
    :param layer: input for layer
    :param shape: shape for weight and the bias. Last element will be used to build bias and hidden
    :param batch_size: size of input batch
    :param num_classes: number of possible classifications
    :param keep_prob: percentage to keep to prevent fitting.
    :return:
    """
    with tf.name_scope('Layer'):
        hidden_size = shape[-1]
        weights = weight_variable(shape)
        weights_hidden = weight_variable([hidden_size, hidden_size])
        biases = bias_variable([hidden_size])

        # Processing inputs to work with scan function
        # Current input shape: (batch_size, time_steps, element_size)
        processed_input = tf.transpose(layer, perm=[1, 0, 2])

        # Current input shape now: (time_steps, batch_size, element_size)
        initial_hidden = tf.zeros([batch_size, hidden_size])

        # Getting all state vectors across time
        all_hidden_states = tf.scan(
            lambda prev, x: tf.tanh(
                tf.matmul(prev, weights_hidden) + tf.matmul(x, weights) + biases
            ),
            processed_input,
            initializer=initial_hidden,
            name='states'
        )

        # Weights for output layers
    with tf.name_scope('Layer'):
        weights_out = weight_variable([hidden_size, num_classes])
        biases_out = bias_variable([num_classes])

        with tf.name_scope('linear_layer_weights') as scope:
            # Iterate across time, apply linear layer to all RNN outputs
            all_outputs = tf.map_fn(
                lambda x: tf.matmul(x, weights_out) + biases_out,
                all_hidden_states
            )
            # Get last output    
            logits = all_outputs[-1]
            tf.summary.histogram('outputs', logits)

        return logits


def get_recurrent_layer(layer, hidden_layer_size, keep_prob=1.0):
    # TensorFlow built-in functions
    rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, layer, dtype=tf.float32)
    tf.summary.histogram('RNN_State', state)
    return outputs, state


def get_rnn_multi_layer(layer, hidden_layer_size, _seqlens, vocabulary_size, embedding_dimension, num_layers=2):
    with tf.name_scope("Embeddings"):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dimension], -1.0, 1.0), name='embedding')
        embed = tf.nn.embedding_lookup(embeddings, layer)

    with tf.name_scope('Layer'):
        cells = [
            tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
            for i in range(num_layers)
        ]
        cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)
        outputs, states = tf.nn.dynamic_rnn(cell, embed, sequence_length=_seqlens, dtype=tf.float32)
