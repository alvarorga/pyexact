"""Exact diagonalization of sparse Hamiltonians."""

import time
import numpy as np
from numba import njit, u2, u8
from scipy.sparse import csr_matrix


@njit(u8(u2, u2))
def binom(N, n):
    """Compute the binomial coefficient of (N n).

    Args:
        N (uint16): largest number.
        n (uint16): smallest number.

    Returns:
        bi (uint64): binomial coefficient of (N n).

    """
    n = max(n, N-n)
    bi = 1
    for i in range(n+1, N+1):
        bi = i*bi
    for i in range(1, N-n+1):
        bi = bi//i
    return bi


@njit(u2(u8, u2))
def count_bits(s, size):
    """Count the number of '1' bits in a state s.

    Args:
        s (uint64): state given by an integer number whose binary
            representation equals a state in the Fock space.
        size (uint64): size of the state. Number of positions that certain
            hold the value 1 or 0.

    Returns:
        bits (uint16): number of bits in the state.

    """
    bits = 0
    for i in range(size):
        bits += (s>>i)&1
    return bits


@njit(u8[:](u2, u2))
def generate_states(size, N):
    """Generate all states of a certain size with N particles.

    Args:
        size (uint16): size of the generated states.
        N (uint16): number of particles in the states.

    Returns.
        states (uint64[:]): all states of size size and N particles.

    """
    num_states = binom(size, N)
    states = np.zeros(num_states, dtype=np.uint64)

    pos = 0
    # Minimum state with size = size+1.
    max_state = 1<<size
    for s in range(max_state):
        if count_bits(s, size) == N:
            states[pos] = s
            pos += 1

    return states


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
def _build_sp_mb_operator_rows_cols(L, N, J, D):
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
def _build_sp_symmetric_mb_operator_rows_cols(L, N, J, D):
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


def build_sparse_mb_operator(L, N, J, D):
    """Compute the sparse matrix of a many body operator.

    The sparse format for the output is CSR.

    Args:
        L (int): system's length.
        N (int): particle number.
        J (2darray of floats): hopping matrix.
        D (2darray of floats): interaction matrix.

    Returns:
        H (CSR sparse array of floats): sparse Hamiltonian matrix.

    """
    # Check if J and D are symmetric.
    t0 = time.time()
    if np.allclose(J, J.T) and np.allclose(D, D.T):
        vals, rows, cols, ns = _build_sp_symmetric_mb_operator_rows_cols(L, N,
                                                                         J, D)
    else:
        vals, rows, cols, ns = _build_sp_mb_operator_rows_cols(L, N, J, D)
    nnz = np.count_nonzero(vals)
    t1 = time.time()
    H = csr_matrix((vals, (rows, cols)), shape=(ns, ns))
    t2 = time.time()

    print_time_and_sparsity_info = True
    if print_time_and_sparsity_info:
        print('Number of states: {}'.format(ns))
        print('Sparsity = {:4.3f}%'.format(100*nnz/ns**2))
        print('Building the vals, rows, cols: {:4.3f} s'.format(t1-t0))
        print('Building the CSR matrix: {:4.3f} s'.format(t2-t1))
    return H
