"""Scripts for computing exact diagonalizations."""

import numpy as np
from numba import njit, u2, f8

from bitwise_funcs import generate_states


@njit(f8[:, :](u2, u2, f8[:, :], f8[:, :]))
def build_mb_operator(L, N, J, D):
    """Build a many-body operator with 2-particle terms.

    Args:
        L (uint16): system's length.
        N (uint16): particle number.
        J (float64[:, :]): hopping matrix.
        D (float64[:, :]): interaction matrix.

    Returns:
        H (float64[:, :]]): many-body operator.

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


@njit(f8[:, :](u2, u2, u2, u2))
def build_mb_correlator(L, N, i, j):
    r"""Build a many-body correlation operator a^\dagger_i a_j.

    Args:
        L (uint16): system's length.
        N (uint16): particle number.
        i (uint16): position of the creation operator.
        j (uint16): position of the annihilation operator.

    Returns:
        C (float64[:, :]]): many-body corelation operator.

    """
    states = generate_states(L, N)
    num_states = states.size

    C = np.zeros((num_states, num_states))

    # Notation:
    #     s: initial state.
    #     t: final state.
    #     ix_#: index of #.
    if i == j:  # Number operator.
        for (ix_s, s) in enumerate(states):
            if (s>>i)&1:
                C[ix_s, ix_s] += 1

    else:  # not i == j.
        for (ix_s, s) in enumerate(states):
            if (not (s>>i)&1) and ((s>>j)&1):
                t = s + (1<<i) - (1<<j)
                ix_t = np.where(states==t)[0][0]
                C[ix_t, ix_s] += 1

    return C


@njit(f8[:, :](u2, u2, u2, u2))
def build_mb_density(L, N, i, j):
    """Build the many-body operator n_i n_j.

    Args:
        L (uint16): system's length.
        N (uint16): particle number.
        i (uint16): position of the first number operator.
        j (uint16): position of the second number operator.

    Returns:
        D (float64[:, :]]): many-body density operator.

    """
    if not i == j:
        states = generate_states(L, N)
        num_states = states.size

        D = np.zeros((num_states, num_states))

        # Notation:
        #     s: initial state.
        #     ix_s: index of s.
        for (ix_s, s) in enumerate(states):
            # Interaction terms: n_i n_j.
            if (s>>i)&1 and (s>>j)&1:
                D[ix_s, ix_s] += 1
    else:  # i == j.
        D = build_mb_correlator(L, N, i, i)

    return D
