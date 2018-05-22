"""Tests for the dense_operators diagonalization."""


import sys
import unittest
import numpy as np

sys.path.append('../')
from pyexact import dense_operators


class ManyBodyOperatorsTestCase(unittest.TestCase):
    """Test for the functions that build many body operators."""

    def test_values_of_number_conserving_interaction_matrix(self):
        """Test values of number conserving interactions n_i*n_j."""
        D = dense_operators.de_pc_interaction(5, 3, 0, 4)
        self.assertAlmostEqual(D[4], 1)
        self.assertAlmostEqual(D[5], 1)
        self.assertAlmostEqual(D[7], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(D), np.sqrt(3))

        D = dense_operators.de_pc_interaction(5, 3, 1, 3)
        self.assertAlmostEqual(D[1], 1)
        self.assertAlmostEqual(D[3], 1)
        self.assertAlmostEqual(D[8], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(D), np.sqrt(3))

        D = dense_operators.de_pc_interaction(5, 3, 4, 2)
        self.assertAlmostEqual(D[5], 1)
        self.assertAlmostEqual(D[6], 1)
        self.assertAlmostEqual(D[9], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(D), np.sqrt(3))

    def test_values_in_number_conserving_correlation_matrix(self):
        """Test values of number conserving correlations b^dagger_i*b_j."""
        C = dense_operators.de_pc_correlator(5, 3, 0, 2)
        self.assertAlmostEqual(C[1, 3], 1)
        self.assertAlmostEqual(C[4, 6], 1)
        self.assertAlmostEqual(C[7, 9], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(C), np.sqrt(3))

        C = dense_operators.de_pc_correlator(5, 3, 1, 2)
        self.assertAlmostEqual(C[1, 2], 1)
        self.assertAlmostEqual(C[4, 5], 1)
        self.assertAlmostEqual(C[8, 9], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(C), np.sqrt(3))

        C = dense_operators.de_pc_correlator(5, 3, 4, 3)
        self.assertAlmostEqual(C[4, 1], 1)
        self.assertAlmostEqual(C[5, 2], 1)
        self.assertAlmostEqual(C[6, 3], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(C), np.sqrt(3))

    def test_values_in_number_conserving_number_matrix(self):
        """Test values of number conserving matrix n_i."""
        N_op = dense_operators.de_pc_number_op(5, 3, 1)
        self.assertAlmostEqual(N_op[0], 1)
        self.assertAlmostEqual(N_op[1], 1)
        self.assertAlmostEqual(N_op[3], 1)
        self.assertAlmostEqual(N_op[4], 1)
        self.assertAlmostEqual(N_op[6], 1)
        self.assertAlmostEqual(N_op[8], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(N_op), np.sqrt(6))

        N_op = dense_operators.de_pc_number_op(5, 3, 4)
        self.assertAlmostEqual(N_op[4], 1)
        self.assertAlmostEqual(N_op[5], 1)
        self.assertAlmostEqual(N_op[6], 1)
        self.assertAlmostEqual(N_op[7], 1)
        self.assertAlmostEqual(N_op[8], 1)
        self.assertAlmostEqual(N_op[9], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(N_op), np.sqrt(6))

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

    def test_values_in_symmetric_operator_matrix(self):
        """Test several values of a general symmetric operator."""
        J = np.reshape(np.arange(25), (5, 5))
        D = np.reshape(25+np.arange(25), (5, 5))
        # Make J and D symmetric.
        for j in range(5):
            for i in range(j):
                J[j, i] = J[i, j]
                D[j, i] = D[i, j]
        H = dense_operators.de_sym_pc_op(5, 3, J, D)
        self.assertAlmostEqual(H[0, 0], 188)
        self.assertAlmostEqual(H[1, 1], 198)
        self.assertAlmostEqual(H[3, 3], 242)
        self.assertAlmostEqual(H[5, 5], 226)
        self.assertAlmostEqual(H[9, 9], 296)
        self.assertAlmostEqual(H[0, 1], 13)
        self.assertAlmostEqual(H[0, 3], 3)
        self.assertAlmostEqual(H[0, 5], 9)
        self.assertAlmostEqual(H[0, 9], 0)
        self.assertAlmostEqual(H[3, 5], 0)
        self.assertAlmostEqual(H[4, 5], 7)
        self.assertAlmostEqual(H[6, 9], 8)
        self.assertAlmostEqual(H[7, 5], 13)
        self.assertAlmostEqual(H[7, 6], 0)
        self.assertAlmostEqual(H[8, 2], 0)
        self.assertAlmostEqual(H[8, 3], 14)
        self.assertAlmostEqual(H[9, 2], 4)

    def test_values_in_npc_operator(self):
        """Test values of a number nonconserving operator."""
        J = np.reshape(np.arange(16), (4, 4))
        D = np.reshape(25+np.arange(16), (4, 4))
        r = np.arange(1, 5)
        l = np.arange(5, 9)
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

    def test_values_in_number_nonconserving_number_matrix(self):
        """Test values of number nonconserving matrix n_i."""
        N_op = dense_operators.de_npc_number_op(4, 1)
        self.assertAlmostEqual(N_op[2], 1)
        self.assertAlmostEqual(N_op[3], 1)
        self.assertAlmostEqual(N_op[6], 1)
        self.assertAlmostEqual(N_op[7], 1)
        self.assertAlmostEqual(N_op[10], 1)
        self.assertAlmostEqual(N_op[11], 1)
        self.assertAlmostEqual(N_op[14], 1)
        self.assertAlmostEqual(N_op[15], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(N_op), np.sqrt(8))

        N_op = dense_operators.de_npc_number_op(4, 3)
        self.assertAlmostEqual(N_op[8], 1)
        self.assertAlmostEqual(N_op[9], 1)
        self.assertAlmostEqual(N_op[10], 1)
        self.assertAlmostEqual(N_op[11], 1)
        self.assertAlmostEqual(N_op[12], 1)
        self.assertAlmostEqual(N_op[13], 1)
        self.assertAlmostEqual(N_op[14], 1)
        self.assertAlmostEqual(N_op[15], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(N_op), np.sqrt(8))

    def test_values_in_number_nonconserving_correlation_matrix(self):
        """Test values of number nonconserving correlations."""
        C = dense_operators.de_npc_correlator(4, 0, 2)
        self.assertAlmostEqual(C[1, 4], 1)
        self.assertAlmostEqual(C[3, 6], 1)
        self.assertAlmostEqual(C[9, 12], 1)
        self.assertAlmostEqual(C[11, 14], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(C), 2)

        C = dense_operators.de_npc_correlator(4, 1, 2)
        self.assertAlmostEqual(C[2, 4], 1)
        self.assertAlmostEqual(C[3, 5], 1)
        self.assertAlmostEqual(C[10, 12], 1)
        self.assertAlmostEqual(C[11, 13], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(C), 2)

        C = dense_operators.de_npc_correlator(4, 3, 2)
        self.assertAlmostEqual(C[8, 4], 1)
        self.assertAlmostEqual(C[9, 5], 1)
        self.assertAlmostEqual(C[10, 6], 1)
        self.assertAlmostEqual(C[11, 7], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(C), 2)

    def test_values_of_number_nonconserving_interaction_matrix(self):
        """Test values of number nonconserving interactions n_i*n_j."""
        D = dense_operators.de_npc_interaction(4, 0, 2)
        self.assertAlmostEqual(D[5], 1)
        self.assertAlmostEqual(D[7], 1)
        self.assertAlmostEqual(D[13], 1)
        self.assertAlmostEqual(D[15], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(D), 2)

        D = dense_operators.de_npc_interaction(4, 1, 3)
        self.assertAlmostEqual(D[10], 1)
        self.assertAlmostEqual(D[11], 1)
        self.assertAlmostEqual(D[14], 1)
        self.assertAlmostEqual(D[15], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(D), 2)

        D = dense_operators.de_npc_interaction(4, 2, 1)
        self.assertAlmostEqual(D[6], 1)
        self.assertAlmostEqual(D[7], 1)
        self.assertAlmostEqual(D[14], 1)
        self.assertAlmostEqual(D[15], 1)
        # Make sure that the other elts are 0.
        self.assertAlmostEqual(np.linalg.norm(D), 2)


if __name__ == '__main__':
    unittest.main()
