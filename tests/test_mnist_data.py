from tests.base import TestBase, tf

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from tensortools.mnist_functions import one_hot_to_array


class TestMNISTFunctions(TestBase):

    def test_one_hot(self):
        expected = 5
        one_hot = np.array(
            [int(i == expected) for i in range(10)]
        ).reshape([1, 10])
        result = one_hot_to_array(one_hot)
        self.assertListEqual(result, [expected])


if __name__ == '__main__':
    tf.test.main()
