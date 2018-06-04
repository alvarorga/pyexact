"""Tests the computation of expected values."""


import sys
import unittest
import numpy as np

sys.path.append('../')
from pyexact.expected import compute_P, compute_D


class NumberConservingPMatrixTestCase(unittest.TestCase):
    """Test the computation of the P matrix.

    All operators conserve particle number.
    """

    def setUp(self):
        """Define a reference unnormalized state."""
        self.L = 5
        self.N = 3
        self.state = np.arange(10)

    def test_number_conserving_dense_hardcore_P_matrix(self):
        """Test several chosen values in a hardcore density matrix.

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

    def test_number_conserving_dense_fermionic_P_matrix(self):
        """Test several chosen values in a fermionic density matrix.

        The matrix is build using the dense representation of the
        c^dagger_i*c_j operator.
        """
        P = compute_P(self.state, self.L, self.N, is_fermionic=True)

        # Compute expected values of number operator.
        self.assertAlmostEqual(P[0, 0], 95)
        self.assertAlmostEqual(P[1, 1], 126)
        self.assertAlmostEqual(P[2, 2], 155)
        self.assertAlmostEqual(P[4, 4], 271)

        # Compute expected values of operator b^dagger_i*b_j.
        self.assertAlmostEqual(P[0, 4], 26)
        self.assertAlmostEqual(P[1, 2], 94)
        self.assertAlmostEqual(P[4, 2], -38)
        self.assertAlmostEqual(P[4, 3], 32)

    def test_number_conserving_sparse_P_matrix(self):
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

class NumberConservingDMatrixTestCase(unittest.TestCase):
    """Test the computation of the D matrix.

    All operators conserve particle number.
    """

    def setUp(self):
        """Define a reference unnormalized state."""
        self.L = 5
        self.N = 3
        self.state = np.arange(10)

    def test_number_conserving_D_matrix(self):
        """Test several chosen values of the D matrix."""
        D = compute_D(self.state, self.L, self.N)

        self.assertAlmostEqual(D[0, 4], 90)
        self.assertAlmostEqual(D[1, 2], 45)
        self.assertAlmostEqual(D[4, 2], 142)
        self.assertAlmostEqual(D[4, 3], 194)

class NumberNonConservingPMatrixTestCase(unittest.TestCase):
    """Test the computation of the P matrix.

    No operator conserves particle number.
    """

    def setUp(self):
        """Define a reference unnormalized state."""
        self.L = 4
        self.state = np.arange(16)

    def test_number_nonconserving_dense_P_matrix(self):
        """Test several chosen values in a density matrix.

        The matrix is build using the dense representation of the
        b^dagger_i*b_j operator.
        """
        P = compute_P(self.state, self.L)

        # Compute expected values of number operator.
        self.assertAlmostEqual(P[0, 0], 680)
        self.assertAlmostEqual(P[1, 1], 740)
        self.assertAlmostEqual(P[2, 2], 860)
        self.assertAlmostEqual(P[3, 3], 1100)

        # Compute expected values of operator b^dagger_i*b_j.
        self.assertAlmostEqual(P[0, 3], 196)
        self.assertAlmostEqual(P[1, 2], 286)
        self.assertAlmostEqual(P[3, 2], 214)
        self.assertAlmostEqual(P[1, 3], 206)

    def test_number_nonconserving_sparse_P_matrix(self):
        """Test several chosen values in a density matrix.

        The matrix is build using the sparse representation of the
        b^dagger_i*b_j operator.
        """
        P = compute_P(self.state, self.L, force_sparse=True)

        # Compute expected values of number operator.
        self.assertAlmostEqual(P[0, 0], 680)
        self.assertAlmostEqual(P[1, 1], 740)
        self.assertAlmostEqual(P[2, 2], 860)
        self.assertAlmostEqual(P[3, 3], 1100)

        # Compute expected values of operator b^dagger_i*b_j.
        self.assertAlmostEqual(P[0, 3], 196)
        self.assertAlmostEqual(P[1, 2], 286)
        self.assertAlmostEqual(P[3, 2], 214)
        self.assertAlmostEqual(P[1, 3], 206)

class NumberNonConservingDMatrixTestCase(unittest.TestCase):
    """Test the computation of the P matrix.

    All operators conserve particle number.
    """

    def setUp(self):
        """Define a reference unnormalized state."""
        self.L = 4
        self.state = np.arange(16)

    def test_number_nonconserving_D_matrix(self):
        """Test several chosen values of the D matrix."""
        D = compute_D(self.state, self.L)

        self.assertAlmostEqual(D[0, 3], 596)
        self.assertAlmostEqual(D[1, 2], 506)
        self.assertAlmostEqual(D[3, 2], 734)
        self.assertAlmostEqual(D[1, 3], 642)
