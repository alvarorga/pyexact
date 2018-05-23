"""Exact diagonalization of sparse Hamiltonians."""

import numpy as np
from numba import njit

from pyexact.bitwise_funcs import binom, generate_states


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
def sp_pc_op(L, N, J, D):
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
            for j in range(L):
                if (np.abs(D[i, j]) > 1e-6) and (i != j):
                    if ((s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
                        vals[c] += D[i, j]
        cols[c] += ix_s
        rows[c] += ix_s
        c += 1

        # Hopping terms: b^dagger_i*b_j.
        for i in range(L):
            for j in range(L):
                if (np.abs(J[i, j]) > 1e-6) and (j != i):
                    if (not (s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
                        t = s + (1<<i) - (1<<j)
                        ix_t = np.where(states == t)[0][0]
                        vals[c] += J[i, j]
                        rows[c] += ix_t
                        cols[c] += ix_s
                        c += 1

    return vals, rows, cols, num_states


@njit()
def sp_sym_pc_op(L, N, J, D):
    """Compute sparse matrix of a symmetric operator.

    Note: if jit is not used, the loops must be done with np.arange
        because the 'right shift' function (>>) needs to operate with
        two uint's as types. Example:
        >>> for i in np.arange(L, dtype=np.uint16):

    Args:
        L (int): system's length.
        N (int): particle number.
        J (2darray of floats): hopping matrix. Must be symmetric.
        D (2darray of floats): interaction matrix.

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

        cols[c] += ix_s
        rows[c] += ix_s
        c += 1

        # Hopping terms: b^dagger_i*b_j.
        for i in range(L):
            for j in range(i):
                if (np.abs(J[i, j]) > 1e-7):
                    if (not (s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
                        t = s + (1<<i) - (1<<j)
                        ix_t = np.where(states == t)[0][0]
                        vals[c] += J[i, j]
                        rows[c] += ix_t
                        cols[c] += ix_s
                        c += 1
                        vals[c] += J[i, j]
                        rows[c] += ix_s
                        cols[c] += ix_t
                        c += 1

    return vals, rows, cols, num_states


@njit()
def sp_npc_op(L, J, D, r, l):
    """Compute sparse matrix of number nonconserving operator.

    Args:
        L (int): system's length.
        J (2darray of floats): hopping matrix. Must be symmetric.
        D (2darray of floats): interaction matrix. Must be symmetric.
        r (1darray of floats): raise matrix. Operator: b^dagger_i.
        l (1darray of floats): lower matrix. Operator: b_i.

    Returns:
        vals, rows, cols: arrays to build the sparse Hamiltonian in
            CSR format.

    """
    num_states = 1<<L
    number_nnz_vals = (num_states
                       + count_nnz_off_diagonal(J)*(1<<(L-2))
                       + r.size*(1<<(L-1))
                       + l.size*(1<<(L-1)))
    vals = np.zeros(number_nnz_vals, dtype=np.float64)
    rows = np.zeros(number_nnz_vals, dtype=np.int32)
    cols = np.zeros(number_nnz_vals, dtype=np.int32)

    # Notation:
    #     s: initial state.
    #     t: final state.
    #     ix_#: index of #.
    c = 0
    for s in range(num_states):
        # On-site terms: n_i.
        for i in range(L):
            if ((s>>i)&np.uint16(1)):
                vals[c] += J[i, i]

        # Interaction terms: n_i*n_j.
        for i in range(L):
            for j in range(i):  # j < i.
                if (np.abs(D[i, j]) > 1e-7):
                    if ((s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
                        vals[c] += D[i, j]

                if (np.abs(D[j, i]) > 1e-7):
                    if ((s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
                        vals[c] += D[j, i]

        cols[c] = s
        rows[c] = s
        c += 1

        # Hopping terms: b^dagger_i*b_j.
        for i in range(L):
            for j in range(i):  # j < i.
                if (np.abs(J[i, j]) > 1e-7):
                    if (not (s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
                        t = s + (1<<i) - (1<<j)
                        vals[c] += J[i, j]
                        rows[c] = t
                        cols[c] = s
                        c += 1

                if (np.abs(J[j, i]) > 1e-7):
                    if (not (s>>j)&np.uint16(1)) and ((s>>i)&np.uint16(1)):
                        t = s + (1<<j) - (1<<i)
                        vals[c] += J[j, i]
                        rows[c] = t
                        cols[c] = s
                        c += 1

        # Raising terms: b^dagger_i.
        for i in range(L):
            if (np.abs(r[i]) > 1e-7):
                if not (s>>i)&np.uint16(1):
                    t = s + (1<<i)
                    vals[c] += r[i]
                    rows[c] = t
                    cols[c] = s
                    c += 1

        # Lowering terms: b_i.
        for i in range(L):
            if (np.abs(l[i]) > 1e-7):
                if (s>>i)&np.uint16(1):
                    t = s - (1<<i)
                    vals[c] += l[i]
                    rows[c] = t
                    cols[c] = s
                    c += 1

    return vals, rows, cols, num_states


def sp_pc_correlator(L, N, i, j):
    """Build a many-body correlation operator b^dagger_i*b_j, i != j.

    Args:
        L (int): system's length.
        N (int): particle number.
        i (int): position of the creation operator.
        j (int): position of the annihilation operator.

    Returns:
        vals, rows, cols (1darray of floats): sparse
            representation of the mb corelation operator.
        num_states (int): number of states.

    """
    if i == j:
        raise ValueError('i and j must be different.')

    states = generate_states(L, N)
    num_states = states.size

    number_nnz_vals = binom(L-2, N-1)
    vals = np.zeros(number_nnz_vals, dtype=np.float64)
    rows = np.zeros(number_nnz_vals, dtype=np.int32)
    cols = np.zeros(number_nnz_vals, dtype=np.int32)

    c = 0
    for ix_s, s in enumerate(states):
        if (not (s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
            t = s + (1<<i) - (1<<j)
            ix_t = np.where(states == t)[0][0]
            vals[c] += 1
            rows[c] = ix_t
            cols[c] = ix_s
            c += 1

    return vals, rows, cols, num_states


@njit()
def sp_npc_correlator(L, i, j):
    """Compute sparse matrix of number nonconserving correlator.

    Args:
        L (int): system's length.
        J (2darray of floats): hopping matrix. Must be symmetric.
        D (2darray of floats): interaction matrix. Must be symmetric.
        r (1darray of floats): raise matrix. Operator: b^dagger_i.
        l (1darray of floats): lower matrix. Operator: b_i.

    Returns:
        vals, rows, cols: arrays to build the sparse Hamiltonian in
            CSR format.

    """
    if i == j:
        raise ValueError('i and j must be different.')

    num_states = 1<<L
    number_nnz_vals = 1<<(L-2)
    vals = np.zeros(number_nnz_vals, dtype=np.float64)
    rows = np.zeros(number_nnz_vals, dtype=np.int32)
    cols = np.zeros(number_nnz_vals, dtype=np.int32)

    c = 0
    for s in range(num_states):
        if (not (s>>i)&np.uint16(1)) and ((s>>j)&np.uint16(1)):
            t = s + (1<<i) - (1<<j)
            vals[c] += 1
            rows[c] = t
            cols[c] = s
            c += 1

    return vals, rows, cols, num_states
