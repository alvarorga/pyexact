"""Test bitwise functions."""


import sys
import unittest
import numpy as np

sys.path.append('../')
from pyexact.bitwise_funcs import (
    binom, count_bits, get_parity, generate_states
    )

class BinomTestCase(unittest.TestCase):
    """Tests for the binomial coefficient function."""

    def test_several_binoms(self):
        """Test several binom outputs."""
        self.assertEqual(binom(6, 3), 20)
        self.assertEqual(binom(6, 1), 6)
        self.assertEqual(binom(10, 4), 210)
        self.assertEqual(binom(7, 3), 35)

    def test_binoms_with_0(self):
        """Test that if n=0, the binom is 1."""
        self.assertEqual(binom(6, 0), 1)
        self.assertEqual(binom(3, 0), 1)


class CountBitsTestCase(unittest.TestCase):
    """Tests for the bit counting function."""

    def test_count_of_several_numbers(self):
        """Test several binom outputs."""
        self.assertEqual(count_bits(6, 4), 2)
        self.assertEqual(count_bits(6, 7), 2)
        self.assertEqual(count_bits(6, 2), 1)
        self.assertEqual(count_bits(9, 4), 2)
        self.assertEqual(count_bits(15, 5), 4)
        self.assertEqual(count_bits(10, 5), 2)
        self.assertEqual(count_bits(3, 5), 2)
        self.assertEqual(count_bits(16, 5), 1)
        self.assertEqual(count_bits(14, 5), 3)


# class GetParityTestCase(unittest.TestCase):
#     """Tests for the parity function."""
#
#     def test_parity_of_several_states(self):
#         """Test several binom outputs."""
#         self.assertEqual(get_parity(1, 0, 1), 1)
#         self.assertEqual(get_parity(3, 2, 0), -1)
#         self.assertEqual(get_parity(15, 0, 4), -1)
#         self.assertEqual(get_parity(15, 1, 4), 1)
#         self.assertEqual(get_parity(15, 2, 4), -1)
#         self.assertEqual(get_parity(15, 3, 4), 1)


class GenerateStatesTestCase(unittest.TestCase):
    """Tests for the generate states function."""

    def test_some_generated_states(self):
        """Test generated states for some combinations."""
        s = np.array([3, 5, 6, 9, 10, 12])
        self.assertTrue(np.allclose(generate_states(4, 2), s))
        s = np.array([7, 11, 13, 14])
        self.assertTrue(np.allclose(generate_states(4, 3), s))
        s = np.array([7, 11, 13, 14, 19, 21, 22, 25, 26, 28])
        self.assertTrue(np.allclose(generate_states(5, 3), s))
