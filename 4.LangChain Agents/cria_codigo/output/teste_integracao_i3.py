
import unittest
from my_module import *

class TestMyModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    def test_function1_positive_input(self):
        self.assertEqual(function1(5), 10)

    def test_function1_negative_input(self):
        with self.assertRaises(ValueError):
            function1(-5)

    def test_function2_string_input(self):
        self.assertEqual(function2("hello"), "HELLO")

    def test_function2_non_string_input(self):
        with self.assertRaises(TypeError):
            function2(123)

    def test_function3_true_input(self):
        self.assertTrue(function3(True))

    def test_function3_false_input(self):
        self.assertFalse(function3(False))

    def test_function4_list_input(self):
        self.assertEqual(len(function4([1, 2, 3])), 6)

    def test_function4_non_list_input(self):
        with self.assertRaises(TypeError):
            function4("abc")

if __name__ == '__main__':
    unittest.main()
