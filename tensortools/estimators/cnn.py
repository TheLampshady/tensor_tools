import numpy as np
import tensorflow as tf


class ConvolutionalNeuralNetwork(object):
    def __init__(self, data_shape, hidden_units, classes, model_dir):
        """

        :type data_shape: list
        :type hidden_units: list
        :type classes: int
        :param model_dir: str
        """
        # Specify that all features have real-value data
        feature_columns = [tf.feature_column.numeric_column("x", shape=data_shape)]

        self.classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=hidden_units,
            n_classes=classes,
            model_dir=model_dir)

    def train(self, data, target, steps=100, epochs=None):
        """

        :type data: list
        :type target: list
        :type steps: int
        :type epochs: int or None
        :return:
        """
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(data)},
            y=np.array(target),
            num_epochs=epochs,
            shuffle=True)

        self.classifier.train(input_fn=input_fn, steps=steps)

    def test(self, data, target, epochs=None):
        """

        :type data: list
        :type target: list
        :type epochs: int or None
        :rtype: float or None
        """
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(data)},
            y=np.array(target),
            num_epochs=epochs,
            shuffle=True)

        return self.classifier.evaluate(input_fn=input_fn, steps=len(data)).get("accuracy")

    def predict(self, data):
        """

        :type data: list
        :rtype: list
        """
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(data)},
            num_epochs=1,
            shuffle=False)

        predictions = list(self.classifier.predict(input_fn=input_fn))
        return [p["classes"].astype(np.int16).tolist()[0] for p in predictions]
