
import unittest
from my_module import *

class TestMyModule(unittest.TestCase):

    def test_function1(self):
        result = function1(5)
        self.assertEqual(result, 10)

    def test_function2(self):
        result = function2("hello")
        self.assertEqual(result, "HELLO")

    def test_function3(self):
        result = function3(True)
        self.assertTrue(result)

    def test_function4(self):
        result = function4([1, 2, 3])
        self.assertEqual(len(result), 6)

if __name__ == '__main__':
    unittest.main()
