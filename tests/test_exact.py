"""Tests for the exact diagonalization."""


import unittest
import numpy as np

import exact


class ManyBodyOperatorsTestCase(unittest.TestCase):
    """Test for the functions that build many body operators."""

    def setUp(self):
        """Set up TestCase."""
        self.L = 5
        self.N = 3

    def test_values_in_density_matrix(self):
        """Test several chosen values in a density matrix."""
        D = exact.build_mb_density(self.L, self.N, 1, 3)
        self.assertAlmostEqual(D[0, 0], 0)
        self.assertAlmostEqual(D[1, 1], 1)
        self.assertAlmostEqual(D[3, 3], 1)
        self.assertAlmostEqual(D[8, 8], 1)
        self.assertAlmostEqual(np.linalg.norm(D, ord='fro'), np.sqrt(3))

    def test_values_in_correlation_matrix(self):
        """Test several chosen values in a correlation matrix."""
        C = exact.build_mb_correlator(self.L, self.N, 0, 2)
        self.assertAlmostEqual(C[1, 3], 1)
        self.assertAlmostEqual(C[4, 6], 1)
        self.assertAlmostEqual(C[7, 9], 1)
        self.assertAlmostEqual(np.linalg.norm(C, ord='fro'), np.sqrt(3))

    def test_values_in_number_matrix(self):
        """Test several chosen values in a number matrix."""
        N_op = exact.build_mb_correlator(self.L, self.N, 4, 4)
        for i in range(4, 10):
            self.assertAlmostEqual(N_op[i, i], 1)
        self.assertAlmostEqual(np.linalg.norm(N_op, ord='fro'), np.sqrt(6))

        # Repeat test building N_op as a density operator.
        N_op = exact.build_mb_density(self.L, self.N, 3, 3)
        for i in [1, 2, 3, 7, 8, 9]:
            self.assertAlmostEqual(N_op[i, i], 1)
        self.assertAlmostEqual(np.linalg.norm(N_op, ord='fro'), np.sqrt(6))

    def test_values_in_operator_matrix(self):
        """Test several values in a general random operator."""
        J = 1 - 2*np.random.rand(self.L, self.L)
        D = 1 - 2*np.random.rand(self.L, self.L)
        Op = exact.build_mb_operator(self.L, self.N, J, D)
        # Temporary values to compare.
        t_val_0_0 = (J[0, 0]+J[1, 1]+J[2, 2]
                    +D[0, 1]+D[0, 2]+D[1, 2]
                    +D[1, 0]+D[2, 0]+D[2, 1])
        self.assertAlmostEqual(Op[0, 0], t_val_0_0)
        t_val_3_3 = (J[1, 1]+J[2, 2]+J[3, 3]
                     +D[1, 2]+D[1, 3]+D[2, 3]
                     +D[2, 1]+D[3, 1]+D[3, 2])
        self.assertAlmostEqual(Op[3, 3], t_val_3_3)
        t_val_6_6 = (J[1, 1]+J[2, 2]+J[4, 4]
                     +D[1, 2]+D[1, 4]+D[2, 4]
                     +D[2, 1]+D[4, 1]+D[4, 2])
        self.assertAlmostEqual(Op[6, 6], t_val_6_6)
        self.assertAlmostEqual(Op[4, 7], J[1, 3])
        self.assertAlmostEqual(Op[4, 9], 0)
        self.assertAlmostEqual(Op[0, 9], 0)
        self.assertAlmostEqual(Op[5, 4], J[2, 1])
        self.assertAlmostEqual(Op[5, 7], J[2, 3])


if __name__ == '__main__':
    unittest.main()
