"""Exact diagonalization of sparse Hamiltonians."""

import numpy as np
from numba import njit

from pyexact.bitwise_funcs import binom
from pyexact.bitwise_funcs import generate_states


@njit()
def count_nnz_off_diagonal(A):
    """Count the number of nonzeros outside the diagonal of A."""
    c = 0
    L = A.shape[0]
    for i in range(L):
        for j in range(L):
            if (i != j) and (np.abs(A[i, j] > 1e-7)):
                c += 1
    return c


@njit()
def build_sparse_mb_operator(L, N, J, D):
    """Compute the data/rows/cols of a mb operator in sparse CSR format.

    Note: if jit is not used, the loops must be done with np.arange
        because the 'right shift' function (>>) needs to operate with
        two uint's as types. Example:
        >>> for i in np.arange(L, dtype=np.uint16):

    Args:
        L (int): system's length.
        N (int): particle number.
        J (2darray of floats): hopping matrix.
        D (2darray of floats): interaction matrix.

    Returns:
        vals, rows, cols: arrays to build the sparse Hamiltonian in
            CSR format.

    """
    states = generate_states(L, N)
    num_states = states.size

    vals = np.zeros(num_states*(N*(L-N) + 2), dtype=np.float64)
    rows = np.zeros(num_states*(N*(L-N) + 2), dtype=np.int32)
    cols = np.zeros(num_states*(N*(L-N) + 2), dtype=np.int32)

    # Notation:
    #     s: initial state.
    #     t: final state.
    #     ix_#: index of #.
    c = 0
    for ix_s, s in enumerate(states):
        # On-site terms: n_i.
        for i in range(L):
            if ((s>>i)&np.uint16(1)):
                vals[c] += J[i, i]
        cols[c] = ix_s
        rows[c] = ix_s
        c += 1

        # Hopping terms: b^dagger_i b_j.
        for i in range(L):
            for j in range(L):
                if (np.abs(J[i, j]) > 1e-6) and (j != i):
                    if (not (s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
                        t = s + (1<<i) - (1<<j)
                        ix_t = np.where(states == t)[0][0]
                        vals[c] = J[i, j]
                        rows[c] = ix_t
                        cols[c] = ix_s
                        c += 1

        # Interaction terms: n_i n_j.
        for i in range(L):
            for j in range(L):
                if (np.abs(D[i, j]) > 1e-6) and (i != j):
                    if ((s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
                        vals[c] += D[i, j]
                        rows[c] = ix_s
                        cols[c] = ix_s
        c += 1

    return vals, rows, cols, num_states


@njit()
def build_sparse_symmetric_mb_operator(L, N, J, D):
    """Compute the data/rows/cols of a mb symmetric operator.

    Note: if jit is not used, the loops must be done with np.arange
        because the 'right shift' function (>>) needs to operate with
        two uint's as types. Example:
        >>> for i in np.arange(L, dtype=np.uint16):

    Args:
        L (int): system's length.
        N (int): particle number.
        J (2darray of floats): hopping matrix. Must be symmetric.
        D (2darray of floats): interaction matrix. Must be symmetric.

    Returns:
        vals, rows, cols: arrays to build the sparse Hamiltonian in
            CSR format.

    """
    if (np.sum((J - J.T)**2) > 1e-7) or (np.sum((D - D.T)**2) > 1e-7):
        raise ValueError('J and/or D is not symmetric.')

    states = generate_states(L, N)
    num_states = states.size

    number_nnz_vals = binom(L, N) + count_nnz_off_diagonal(J)*binom(L-2, N-1)
    vals = np.zeros(number_nnz_vals, dtype=np.float64)
    rows = np.zeros(number_nnz_vals, dtype=np.int32)
    cols = np.zeros(number_nnz_vals, dtype=np.int32)

    # Notation:
    #     s: initial state.
    #     t: final state.
    #     ix_#: index of #.
    c = 0
    for ix_s, s in enumerate(states):
        # On-site terms: n_i.
        for i in range(L):
            if ((s>>i)&np.uint16(1)):
                vals[c] += J[i, i]

        # Interaction terms: n_i*n_j.
        for i in range(L):
            for j in range(i):  # j < i.
                if (np.abs(D[i, j]) > 1e-7):
                    if ((s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
                        vals[c] += 2*D[i, j]

        cols[c] = ix_s
        rows[c] = ix_s
        c += 1

        # Hopping terms: b^dagger_i*b_j.
        for i in range(L):
            for j in range(i):
                if (np.abs(J[i, j]) > 1e-7):
                    if (not (s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
                        t = s + (1<<i) - (1<<j)
                        ix_t = np.where(states == t)[0][0]
                        vals[c] = J[i, j]
                        rows[c] = ix_t
                        cols[c] = ix_s
                        c += 1
                        vals[c] = J[i, j]
                        rows[c] = ix_s
                        cols[c] = ix_t
                        c += 1

    return vals, rows, cols, num_states
