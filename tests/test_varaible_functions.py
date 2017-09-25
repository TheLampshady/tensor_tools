import numpy as np
from tests.base import TestBase, tf, INIT_TRUNC_NAME, INIT_CONST_NAME

from tensortools.tensor_functions import bias_variable, weight_variable, \
    DEFAULT_STDDEV


class TestVariables(TestBase):

    def test_default_weights(self):
        weights = weight_variable([3,5])
        with self.test_session():
            weights.initializer.run()
            result = weights.eval()
        self.assertIn(INIT_TRUNC_NAME, weights.initial_value.name)
        self.assertLessEqual(result.max(), DEFAULT_STDDEV * 2)
        self.assertGreaterEqual(result.min(), DEFAULT_STDDEV * -2)

    def test_default_biases(self):
        biases = bias_variable([5])
        with self.test_session():
            biases.initializer.run()
            result = biases.eval()
        self.assertIn(INIT_CONST_NAME, biases.initial_value.name)

        # Numpy does not return accurate floats.
        self.assertNear(result[0].item(), 0.1, err=0.0001)

if __name__ == '__main__':
    tf.test.main()
