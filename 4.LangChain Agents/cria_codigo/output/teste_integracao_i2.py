
import unittest
from my_module import *

class TestMyModule(unittest.TestCase):
    def setUp(self):
        pass

    def test_function1(self):
        self.assertEqual(function1(5), 10)

    def test_function2(self):
        self.assertEqual(function2("hello"), "HELLO")

    def test_function3(self):
        self.assertTrue(function3(True))

    def test_function4(self):
        self.assertEqual(len(function4([1, 2, 3])), 6)

if __name__ == '__main__':
    unittest.main()
