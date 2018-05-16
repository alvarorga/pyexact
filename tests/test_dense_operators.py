"""Tests for the dense_operators diagonalization."""


import unittest
import numpy as np

from pyexact import dense_operators


class ManyBodyOperatorsTestCase(unittest.TestCase):
    """Test for the functions that build many body operators."""

    def test_values_in_density_matrix(self):
        """Test several chosen values in a density matrix."""
        D = dense_operators.build_mb_interaction(5, 3, 1, 3)
        self.assertAlmostEqual(D[1, 1], 1)
        self.assertAlmostEqual(D[3, 3], 1)
        self.assertAlmostEqual(D[8, 8], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(D, ord='fro'), np.sqrt(3))

    def test_values_in_correlation_matrix(self):
        """Test several chosen values in a correlation matrix."""
        C = dense_operators.build_mb_correlator(5, 3, 0, 2)
        self.assertAlmostEqual(C[1, 3], 1)
        self.assertAlmostEqual(C[4, 6], 1)
        self.assertAlmostEqual(C[7, 9], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(C, ord='fro'), np.sqrt(3))

    def test_values_in_number_matrix(self):
        """Test several chosen values in a number matrix."""
        N_op = dense_operators.build_mb_correlator(5, 3, 4, 4)
        self.assertAlmostEqual(N_op[4, 4], 1)
        self.assertAlmostEqual(N_op[5, 5], 1)
        self.assertAlmostEqual(N_op[6, 6], 1)
        self.assertAlmostEqual(N_op[7, 7], 1)
        self.assertAlmostEqual(N_op[8, 8], 1)
        self.assertAlmostEqual(N_op[9, 9], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(N_op, ord='fro'), np.sqrt(6))

    def test_values_in_operator_matrix(self):
        """Test several values of a general operator."""
        J = np.reshape(np.arange(25), (5, 5))
        D = np.reshape(25+np.arange(25), (5, 5))
        H = dense_operators.de_pc_op(5, 3, J, D)
        self.assertAlmostEqual(H[0, 0], 204)
        self.assertAlmostEqual(H[1, 1], 222)
        self.assertAlmostEqual(H[3, 3], 258)
        self.assertAlmostEqual(H[5, 5], 258)
        self.assertAlmostEqual(H[9, 9], 312)
        self.assertAlmostEqual(H[0, 1], 13)
        self.assertAlmostEqual(H[0, 3], 3)
        self.assertAlmostEqual(H[0, 5], 9)
        self.assertAlmostEqual(H[0, 9], 0)
        self.assertAlmostEqual(H[3, 5], 0)
        self.assertAlmostEqual(H[4, 5], 7)
        self.assertAlmostEqual(H[6, 9], 8)
        self.assertAlmostEqual(H[7, 5], 17)
        self.assertAlmostEqual(H[7, 6], 0)
        self.assertAlmostEqual(H[8, 2], 0)
        self.assertAlmostEqual(H[8, 3], 22)
        self.assertAlmostEqual(H[9, 2], 20)

    def test_values_in_npc_operator(self):
        """Test values of a number nonconserving operator."""
        J = np.reshape(np.arange(16), (4, 4))
        D = np.reshape(25+np.arange(16), (4, 4))
        r = np.arange(1, 5)
        l = np.arange(5, 9)
        print(f'J =\n{J}')
        print(f'D =\n{D}')
        print(f'r =\n{r}')
        print(f'l =\n{l}')
        H = dense_operators.de_npc_op(4, J, D, r, l)
        self.assertAlmostEqual(H[0, 0], 0)
        self.assertAlmostEqual(H[1, 1], 0)
        self.assertAlmostEqual(H[2, 2], 5)
        self.assertAlmostEqual(H[3, 3], 60)
        self.assertAlmostEqual(H[5, 5], 70)
        self.assertAlmostEqual(H[9, 9], 80)
        self.assertAlmostEqual(H[13, 13], 225)
        self.assertAlmostEqual(H[15, 15], 420)
        self.assertAlmostEqual(H[0, 1], 5)
        self.assertAlmostEqual(H[0, 3], 0)
        self.assertAlmostEqual(H[0, 9], 0)
        self.assertAlmostEqual(H[3, 5], 6)
        self.assertAlmostEqual(H[4, 5], 5)
        self.assertAlmostEqual(H[6, 9], 0)
        self.assertAlmostEqual(H[7, 5], 2)
        self.assertAlmostEqual(H[7, 6], 1)
        self.assertAlmostEqual(H[8, 2], 13)
        self.assertAlmostEqual(H[8, 3], 0)
        self.assertAlmostEqual(H[9, 2], 0)
        self.assertAlmostEqual(H[10, 2], 4)
        self.assertAlmostEqual(H[15, 13], 2)

if __name__ == '__main__':
    unittest.main()
