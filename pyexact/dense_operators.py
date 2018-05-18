"""Scripts for computing exact diagonalizations."""

import numpy as np
from numba import njit

from pyexact.bitwise_funcs import generate_states


@njit()
def de_pc_op(L, N, J, D):
    """Build a many-body operator with 2-particle terms.

    Args:
        L (int): system's length.
        N (int): particle number.
        J (2darray of floats): hopping matrix.
        D (2darray of floats): interaction matrix.

    Returns:
        H (2darray of floats): many-body operator.

    """
    states = generate_states(L, N)
    num_states = states.size

    H = np.zeros((num_states, num_states), np.float64)

    # Notation:
    #     s: initial state.
    #     t: final state.
    #     ix_#: index of #.
    for (ix_s, s) in enumerate(states):
        for i in range(L):
            # On-site terms: n_i.
            if (s>>i)&1:
                H[ix_s, ix_s] += J[i, i]

            for j in range(L):
                if i != j:
                    # Hopping terms: b^dagger_i b_j.
                    if not (s>>i)&1 and (s>>j)&1:
                        t = s + (1<<i) - (1<<j)
                        ix_t = np.where(states == t)[0][0]
                        H[ix_t, ix_s] += J[i, j]

                    # Interaction terms: n_i n_j.
                    if (s>>i)&1 and (s>>j)&1:
                        H[ix_s, ix_s] += D[i, j]

    return H


@njit()
def de_npc_op(L, J, D, r, l):
    """Build a dense number nonconserving operator.

    Args:
        L (int): system's length.
        J (2darray of floats): hopping matrix.
        D (2darray of floats): interaction matrix.
        r (1darray of floats): raising operator: b^dagger_i.
        l (1darray of floats): lowering operator: b_i.

    Returns:
        H (2darray of floats): many-body operator.

    """
    num_states = 1<<L

    H = np.zeros((num_states, num_states), np.float64)

    # Notation:
    #     s: initial state.
    #     t: final state.
    for s in range(1<<L):
        for i in range(L):
            # On-site terms: n_i.
            if (s>>i)&1:
                H[s, s] += J[i, i]

            for j in range(L):
                if i != j:
                    # Hopping terms: b^dagger_i*b_j.
                    if not (s>>i)&1 and (s>>j)&1:
                        t = s + (1<<i) - (1<<j)
                        H[t, s] += J[i, j]

                    # Interaction terms: n_i*n_j.
                    if (s>>i)&1 and (s>>j)&1:
                        H[s, s] += D[i, j]

            # Raising operator: b^dagger_i.
            if not (s>>i)&1:
                t = s + (1<<i)
                H[t, s] += r[i]

            # Lowering operator: b_i.
            if (s>>i)&1:
                t = s - (1<<i)
                H[t, s] += l[i]

    return H


@njit()
def build_mb_number_op(L, N, i):
    """Build a many-body number operator n_i.

    Args:
        L (int): system's length.
        N (int): particle number.
        i (int): position of the number operator.

    Returns:
        C (1darray of floats): many-body corelation operator.

    """
    states = generate_states(L, N)
    num_states = states.size

    C = np.zeros(num_states, np.float64)

    for ix_s, s in enumerate(states):
        if (s>>i)&1:
            C[ix_s] += 1

    return C


@njit()
def build_mb_correlator(L, N, i, j):
    """Build a many-body correlation operator b^dagger_i b_j, i != j.

    Args:
        L (int): system's length.
        N (int): particle number.
        i (int): position of the creation operator.
        j (int): position of the annihilation operator.

    Returns:
        C (2darray of floats): many-body corelation operator.

    """
    if i == j:
        raise ValueError('i and j must be different.')

    states = generate_states(L, N)
    num_states = states.size

    C = np.zeros((num_states, num_states), np.float64)

    for ix_s, s in enumerate(states):
        if (not (s>>i)&1) and ((s>>j)&1):
            t = s + (1<<i) - (1<<j)
            ix_t = np.where(states==t)[0][0]
            C[ix_t, ix_s] += 1

    return C


@njit()
def build_mb_interaction(L, N, i, j):
    """Build the many-body operator n_i n_j.

    Args:
        L (int): system's length.
        N (int): particle number.
        i (int): position of the first number operator.
        j (int): position of the second number operator.

    Returns:
        D (1darray of floats): many-body density operator.

    """
    if i == j:
        raise ValueError('i and j must be different.')

    states = generate_states(L, N)
    num_states = states.size

    D = np.zeros(num_states, np.float64)

    for ix_s, s in enumerate(states):
        if (s>>i)&1 and (s>>j)&1:
            D[ix_s] += 1

    return D
