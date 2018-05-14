"""Scripts for computing exact diagonalizations."""

import numpy as np
from numba import njit, u2, u4, f8


@njit(u4(u2, u2))
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


@njit(u2(u4, u2))
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


@njit(u4[:](u2, u2))
def generate_states(size, N):
    """Generate all states of a certain size with N particles.

    Args:
        size (uint16): size of the generated states.
        N (uint16): number of particles in the states.

    Returns:
        states (uint32[:]): all states of size size and N particles.

    """
    num_states = binom(size, N)
    states = np.zeros(num_states, dtype=np.uint32)

    pos = 0
    # Minimum state with size = size+1.
    max_state = 1<<size
    for s in range(max_state):
        if count_bits(s, size) == N:
            states[pos] = s
            pos += 1

    return states


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
