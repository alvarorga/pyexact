"""Functions for working with bits."""

import numpy as np
from numba import njit


@njit()
def binom(N, n):
    """Compute the binomial coefficient of (N n).

    Args:
        N (int): largest number.
        n (int): smallest number.

    Returns:
        bi (int): binomial coefficient of (N n).

    """
    n = max(n, N-n)
    bi = 1
    for i in range(n+1, N+1):
        bi = i*bi
    for i in range(1, N-n+1):
        bi = bi//i
    return bi


@njit()
def count_bits(s, size):
    """Count the number of '1' bits in a state s.

    Args:
        s (int): state given by an integer number whose binary
            representation equals a state in the Fock space.
        size (int): size of the state. Number of positions that certain
            hold the value 1 or 0.

    Returns:
        bits (int): number of bits in the state.

    """
    bits = 0
    for i in range(size):
        bits += (s>>i)&1
    return bits


@njit()
def generate_states(size, N):
    """Generate all states of a certain size with N particles.

    Args:
        size (int): size of the generated states.
        N (int): number of particles in the states.

    Returns:
        states (1darray of uints): all states of size size and N
            particles.

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
