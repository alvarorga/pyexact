"""Test exact diagonalization of sparse matrices."""

import unittest
import numpy as np

import exact
import sparse_exact


class SparseManyBodyOperatorsTestCase(unittest.TestCase):
    """Test for the functions that build sparse many body operators."""

    def test_sparse_coincides_with_dense(self):
        """Test that sparse Hamiltonians coincide with dense ones."""
        L = 12
        N = 6
        J = np.random.rand(L, L)
        D = np.random.rand(L, L)
        dense = exact.build_mb_operator(L, N, J, D)
        H = sparse_exact.build_sparse_mb_operator(L, N, J, D)
        H = H.toarray()
        self.assertTrue(np.allclose(dense, H))

    def test_symmetric_sparse_coincides_with_dense(self):
        """Test that sparse Hamiltonians coincide with dense ones."""
        L = 12
        N = 6
        J = np.random.rand(L, L)
        J = (J + J.T)/2
        D = np.random.rand(L, L)
        D = (D + D.T)/2
        dense = exact.build_mb_operator(L, N, J, D)
        H = sparse_exact.build_sparse_mb_operator(L, N, J, D)
        H = H.toarray()
        self.assertTrue(np.allclose(dense, H))

    def test_values_in_operator_matrix(self):
        """Test several values in a general random operator."""
        J = 1 - 2*np.random.rand(5, 5)
        D = 1 - 2*np.random.rand(5, 5)
        Op = exact.build_mb_operator(5, 3, J, D)
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

    def test_values_in_symmetric_operator_matrix(self):
        """Test several values in a general random operator."""
        J = 1 - 2*np.random.rand(5, 5)
        J = (J + J.T)/2
        D = 1 - 2*np.random.rand(5, 5)
        D = (D + D.T)/2
        Op = exact.build_mb_operator(5, 3, J, D)
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
