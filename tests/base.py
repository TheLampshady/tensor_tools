import shutil
import os

import tensorflow as tf

INIT_TRUNC_NAME = "truncated_normal"
INIT_CONST_NAME = "Const"
TEST_DATA_DIR = "data"


class TestBase(tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        if not os.path.exists(TEST_DATA_DIR):
            os.makedirs(TEST_DATA_DIR)
        super(TestBase, self).setUp()

    def tearDown(self):
        shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
        super(TestBase, self).tearDown()
