"""Tests for the diagonalization."""


import sys
import unittest
import numpy as np

sys.path.append('../')
from pyexact.dense_fermionic_operators import (
    fer_de_pc_op, fer_de_sym_pc_op
    )

class ManyBodyFermionicOperatorsTestCase(unittest.TestCase):
    """Test the functions that build many-body fermionic operators."""

    def test_values_in_operator_matrix(self):
        """Test several values of a general operator."""
        J = np.reshape(np.arange(25), (5, 5))
        D = np.reshape(25+np.arange(25), (5, 5))
        H = fer_de_pc_op(5, 3, J, D)
        self.assertAlmostEqual(H[0, 0], 204)
        self.assertAlmostEqual(H[1, 1], 222)
        self.assertAlmostEqual(H[3, 3], 258)
        self.assertAlmostEqual(H[5, 5], 258)
        self.assertAlmostEqual(H[9, 9], 312)
        self.assertAlmostEqual(H[0, 1], 13)
        self.assertAlmostEqual(H[0, 3], 3)
        self.assertAlmostEqual(H[0, 5], -9)
        self.assertAlmostEqual(H[0, 9], 0)
        self.assertAlmostEqual(H[3, 5], 0)
        self.assertAlmostEqual(H[4, 5], 7)
        self.assertAlmostEqual(H[6, 9], -8)
        self.assertAlmostEqual(H[7, 5], 17)
        self.assertAlmostEqual(H[7, 6], 0)
        self.assertAlmostEqual(H[8, 2], 0)
        self.assertAlmostEqual(H[8, 3], -22)
        self.assertAlmostEqual(H[9, 2], 20)

    def test_values_in_symmetric_operator_matrix(self):
        """Test several values of a general symmetric operator."""
        J = np.reshape(np.arange(25), (5, 5))
        D = np.reshape(25+np.arange(25), (5, 5))
        # Make J and D symmetric.
        for j in range(5):
            for i in range(j):
                J[j, i] = J[i, j]
                D[j, i] = D[i, j]
        H = fer_de_sym_pc_op(5, 3, J, D)
        self.assertAlmostEqual(H[0, 0], 188)
        self.assertAlmostEqual(H[1, 1], 198)
        self.assertAlmostEqual(H[3, 3], 242)
        self.assertAlmostEqual(H[5, 5], 226)
        self.assertAlmostEqual(H[9, 9], 296)
        self.assertAlmostEqual(H[0, 1], 13)
        self.assertAlmostEqual(H[0, 3], 3)
        self.assertAlmostEqual(H[0, 5], -9)
        self.assertAlmostEqual(H[0, 9], 0)
        self.assertAlmostEqual(H[3, 5], 0)
        self.assertAlmostEqual(H[4, 5], 7)
        self.assertAlmostEqual(H[6, 9], -8)
        self.assertAlmostEqual(H[7, 5], 13)
        self.assertAlmostEqual(H[7, 6], 0)
        self.assertAlmostEqual(H[8, 2], 0)
        self.assertAlmostEqual(H[8, 3], -14)
        self.assertAlmostEqual(H[9, 2], 4)
