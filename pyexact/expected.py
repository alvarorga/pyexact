"""Computation of expected values of the 2-reduced density matrices."""

import numpy as np
from scipy.sparse import csr_matrix

from pyexact.dense_operators import (
    de_pc_number_op, de_pc_correlator, de_pc_interaction,
    de_npc_number_op, de_npc_correlator, de_npc_interaction
    )
from pyexact.sparse_operators import sp_pc_correlator, sp_npc_correlator


def compute_P(state, L, N=None, force_sparse=False):
    r"""Compute the DOCI `P` matrix.

    `P` is defined as: :math:`P_{ij} = \langle b^\dagger_i b_j\rangle`.

    Args:
        state (1darray of floats): vector representation of the state.
        L (int): system's length.
        N (int, opt): number of particles in the system. If None, the
            particle number is not conserved.
        force_sparse (bool, opt): for testing purposes. If True, the
            sparse representation of the operator
            :math:`b^\dagger_i b_j` is always used.

    Returns:
        P (2darray of floats): DOCI `P` matrix.

    """
    P = np.zeros((L, L), np.float64)

    is_N_conserved = True if N is not None else False
    is_sparse = True if (L >= 12) or force_sparse else False

    if is_N_conserved:
        # Number operator.
        for i in range(L):
            ni = de_pc_number_op(L, N, i)
            P[i, i] = np.dot(ni, state**2)

        # Correlation terms.
        for i in range(L):
            for j in range(i):  # j < i.
                if is_sparse:
                    v, r, c, ns = sp_pc_correlator(L, N, i, j)
                    bibj = csr_matrix((v, (r, c)), shape=(ns, ns))
                    P[i, j] = np.dot(state, bibj.dot(state))
                else:
                    bibj = de_pc_correlator(L, N, i, j)
                    P[i, j] = np.dot(state, np.dot(bibj, state))
                P[j, i] = P[i, j]
    else:
        # Number operator.
        for i in range(L):
            ni = de_npc_number_op(L, i)
            P[i, i] = np.dot(ni, state**2)

        # Correlation terms.
        for i in range(L):
            for j in range(i):  # j < i.
                if is_sparse:
                    v, r, c, ns = sp_npc_correlator(L, i, j)
                    bibj = csr_matrix((v, (r, c)), shape=(ns, ns))
                    P[i, j] = np.dot(state, bibj.dot(state))
                else:
                    bibj = de_npc_correlator(L, i, j)
                    P[i, j] = np.dot(state, np.dot(bibj, state))
                P[j, i] = P[i, j]

    return P


def compute_D(state, L, N=None):
    r"""Compute the DOCI `D` matrix.

    `D` is defined as: :math:`D_{ij} = \langle n_i n_j\rangle`.

    Args:
        state (1darray of floats): vector representation of the state.
        L (int): system's length.
        N (int, opt): number of particles in the system. If None,
            particle number is not conserved.

    Returns:
        D (2darray of floats): DOCI `D` matrix.

    """
    D = np.zeros((L, L), np.float64)

    is_N_conserved = True if N is not None else False

    if is_N_conserved:
        for i in range(L):
            for j in range(i):  # j < i.
                ninj = de_pc_interaction(L, N, i, j)
                D[i, j] = np.dot(ninj, state**2)
                D[j, i] = D[i, j]
    else:
        for i in range(L):
            for j in range(i):  # j < i.
                ninj = de_npc_interaction(L, i, j)
                D[i, j] = np.dot(ninj, state**2)
                D[j, i] = D[i, j]

    return D
