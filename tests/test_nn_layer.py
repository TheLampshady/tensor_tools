import numpy as np
from tests.base import TestBase, tf, INIT_TRUNC_NAME, INIT_CONST_NAME

from tensortools.nn_layers import get_ann_layer


class TestNNLayers(TestBase):

    def test_ann_layer_runs(self):
        _input = tf.placeholder(tf.float32, [None, 10], name="Input")
        layer = get_ann_layer(_input, node_size=5)
        init = tf.global_variables_initializer()
        merged_summary_op = tf.summary.merge_all()

        self.assertTrue(init)
        self.assertTrue(merged_summary_op is not None)

        with self.test_session():
            init.run()
            feed_dict = {_input: [range(10)]}
            try:
                layer.eval(feed_dict=feed_dict)
            except Exception as e:
                self.fail("Layer did not run. Error: %s" % str(e))

    def test_double_ann_layer_runs(self):
        _input = tf.placeholder(tf.float32, [None, 10], name="Input")
        layer = get_ann_layer(_input, node_size=5)
        layer2 = get_ann_layer(layer, node_size=2, activate=False)
        init = tf.global_variables_initializer()
        merged_summary_op = tf.summary.merge_all()

        self.assertTrue(init)
        self.assertTrue(merged_summary_op is not None)

        with self.test_session():
            init.run()
            feed_dict = {_input: [range(10)]}
            try:
                layer2.eval(feed_dict=feed_dict)
            except Exception as e:
                self.fail("Layer2 did not run. Error: %s" % str(e))


if __name__ == '__main__':
    tf.test.main()
