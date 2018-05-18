"""Script for the automated computation of expected values."""

import numpy as np
from scipy.sparse import csr_matrix

from pyexact.dense_operators import build_mb_number_op
from pyexact.dense_operators import build_mb_correlator
from pyexact.dense_operators import build_mb_interaction
from pyexact.sparse_operators import build_mb_sparse_correlator


def compute_P(state, L, N=None, force_sparse=False):
    """Compute the DOCI P matrix.

    The elements of P are:
        P[i, j] = <b^dagger_i*b_j>.

    Args:
        state (1darray of floats): vector representation of the state.
        L (int): system's length.
        N (int, opt): number of particles in the system. If None, the
            particle number is not conserved.
        force_sparse (bool, opt): for testing purposes. If True, the
            sparse representation of the operator b^dagger_i*b_j is
            always used.

    Returns:
        P (2darray of floats): DOCI P matrix.

    """
    P = np.zeros((L, L), np.float64)

    is_N_conserved = True if N is not None else False
    is_sparse = True if (L >= 12) or force_sparse else False

    if is_N_conserved:
        # Number operator.
        for i in range(L):
            ni = build_mb_number_op(L, N, i)
            P[i, i] = np.dot(ni, state**2)

        # Correlation terms.
        for i in range(L):
            for j in range(i):  # j < i.
                if is_sparse:
                    v, r, c, ns = build_mb_sparse_correlator(L, N, i, j)
                    bibj = csr_matrix((v, (r, c)), shape=(ns, ns))
                    P[i, j] = np.dot(state, bibj.dot(state))
                else:
                    bibj = build_mb_correlator(L, N, i, j)
                    P[i, j] = np.dot(state, np.dot(bibj, state))
                P[j, i] = P[i, j]
    else:
        raise NotImplementedError('Nonconserved N is not implemented yet.')

    return P


def compute_D(state, L, N=None):
    """Compute the DOCI D matrix.

    The elements of D are:
        D[i, j] = <n_i*n_j>.

    Args:
        state (1darray of floats): vector representation of the state.
        L (int): system's length.
        N (int, opt): number of particles in the system. If None, the
            particle number is not conserved.

    Returns:
        D (2darray of floats): DOCI D matrix.

    """
    D = np.zeros((L, L), np.float64)

    is_N_conserved = True if N is not None else False

    if is_N_conserved:
        for i in range(L):
            for j in range(i):  # j < i.
                ninj = build_mb_interaction(L, N, i, j)
                D[i, j] = np.dot(ninj, state**2)
                D[j, i] = D[i, j]
    else:
        raise NotImplementedError('Nonconserved N is not implemented yet.')

    return D