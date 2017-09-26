from tests.base import TestBase, tf, TEST_DATA_DIR

from tensortools.estimators import DeepNeuralNetwork

MODEL_DIR = TEST_DATA_DIR + "/model"


class TestNNLayers(TestBase):

    def test_dnn(self):
        train_data = [
            list(range(10)),
            list(range(10, 20)),
            list(range(20, 30)),
        ]
        target = [1, 1, 0]

        test_data = [
            list(range(30, 40)),
        ]
        test_target = [0]

        dnn = DeepNeuralNetwork([10], [5], 2, MODEL_DIR)
        dnn.train(train_data, target)

        accuracy = dnn.test(test_data, test_target)
        self.assertNear(accuracy, 1.0, err=0.00001)

        result = dnn.predict([list(range(10, 20))])
        self.assertEquals(result, [1])

        result = dnn.predict([list(range(20, 30))])
        self.assertEquals(result, [0])


if __name__ == '__main__':
    tf.test.main()
