import unittest
from main import sumboth


class test_sumboth(unittest.TestCase):

    def test_regularSum(self):
        sum = sumboth("10.10")
        self.assertEqual(20, sum)

    def test_nonNumeric(self):
        with self.assertRaises(ValueError):
            sumboth("ab.ab")