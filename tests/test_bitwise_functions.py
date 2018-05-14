"""Tests of the bitwise functions."""


import sys
import unittest
import numpy as np

sys.path.append('../')
from pyexact.bitwise_funcs import binom
from pyexact.bitwise_funcs import count_bits
from pyexact.bitwise_funcs import generate_states


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
        pass


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


class GenerateStatesTestCase(unittest.TestCase):
    """Tests for the generate states function."""

    def test_some_generated_states(self):
        """Test generated states for some combinations."""
        s = np.array([3, 5, 6, 9, 10, 12])
        self.assertTrue(np.allclose(generate_states(4, 2), s))
