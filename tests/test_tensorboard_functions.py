
from tests.base import TestBase, tf, TEST_DATA_DIR

from tensortools.tensor_functions import build_metadata


class TestTensorBoard(TestBase):

    def test_meta_list(self):
        data = [2, 6, 9]
        expected = "\n".join([str(x) for x in data]) + "\n"
        filename = build_metadata(data, TEST_DATA_DIR + "/test.tsv")
        with open(filename) as f:
            result = f.read()
        self.assertEqual(result, expected)

    def test_meta_headers(self):
        data = [(1, 1), (2, 2)]
        header = ("Test", "Me")
        expected = "Test\tMe\n1\t1\n2\t2\n"
        filename = build_metadata(data, TEST_DATA_DIR + "/test.tsv", header)
        with open(filename) as f:
            result = f.read()
        self.assertEqual(result, expected)

if __name__ == '__main__':
    tf.test.main()
