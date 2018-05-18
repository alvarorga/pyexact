"""Tests the computation of expected values."""


import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.append('../')
from pyexact.expected import compute_P
from pyexact.expected import compute_D


class PMatrixTestCase(unittest.TestCase):
    """Test the computation of the P matrix."""

    def setUp(self):
        """Define a reference unnormalized state."""
        self.L = 5
        self.N = 3
        self.state = np.arange(10)

    def test_dense_P_matrix(self):
        """Test several chosen values in a density matrix.

        The matrix is build using the dense representation of the
        b^dagger_i*b_j operator.
        """
        P = compute_P(self.state, self.L, self.N)

        # Compute expected values of number operator.
        self.assertAlmostEqual(P[0, 0], 95)
        self.assertAlmostEqual(P[1, 1], 126)
        self.assertAlmostEqual(P[2, 2], 155)
        self.assertAlmostEqual(P[4, 4], 271)

        # Compute expected values of operator b^dagger_i*b_j.
        self.assertAlmostEqual(P[0, 4], 26)
        self.assertAlmostEqual(P[1, 2], 94)
        self.assertAlmostEqual(P[4, 2], 38)
        self.assertAlmostEqual(P[4, 3], 32)

    def test_sparse_P_matrix(self):
        """Test several chosen values in a density matrix.

        The matrix is build using the sparse representation of the
        b^dagger_i*b_j operator.
        """
        P = compute_P(self.state, self.L, self.N, force_sparse=True)

        # Compute expected values of number operator.
        self.assertAlmostEqual(P[0, 0], 95)
        self.assertAlmostEqual(P[1, 1], 126)
        self.assertAlmostEqual(P[2, 2], 155)
        self.assertAlmostEqual(P[4, 4], 271)

        # Compute expected values of operator b^dagger_i*b_j.
        self.assertAlmostEqual(P[0, 4], 26)
        self.assertAlmostEqual(P[1, 2], 94)
        self.assertAlmostEqual(P[4, 2], 38)
        self.assertAlmostEqual(P[4, 3], 32)

    def test_D_matrix(self):
        """Test several chosen values of the D matrix."""
        D = compute_D(self.state, self.L, self.N)

        self.assertAlmostEqual(D[0, 4], 90)
        self.assertAlmostEqual(D[1, 2], 45)
        self.assertAlmostEqual(D[4, 2], 142)
        self.assertAlmostEqual(D[4, 3], 194)
