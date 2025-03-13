
import unittest
from my_module import *

class TestMyModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_function1(self):
        self.assertEqual(function1(5), 10)
        with self.assertRaises(ValueError):
            function1(-5)

    def test_function2(self):
        self.assertEqual(function2("hello").upper(), "HELLO")
        with self.assertRaises(TypeError):
            function2(123)

    def test_function3(self):
        self.assertTrue(function3(True))
        self.assertFalse(function3(False))

    def test_function4(self):
        result = function4([1, 2, 3])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)
        with self.assertRaises(TypeError):
            function4("abc")

if __name__ == '__main__':
    unittest.main()
