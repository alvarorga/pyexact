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

    Examples:
    >>> binom(6, 3)
    20
    >>> binom(5, 1)
    5
    >>> binom(7, 3)
    35

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

    Examples:
    >>> count_bits(4, 5)
    1
    >>> count_bits(5, 5)
    2
    >>> count_bits(7, 5)
    3

    """
    bits = 0
    for i in range(size):
        bits += (s>>i)&1
    return bits


@njit()
def get_parity(s, i, j):
    """Count the '1' bits in a state s between to sites i and j.

    Args:
        s (int): state given by an integer number whose binary
            representation equals a state in the Fock space.
        i, j (int): sites between which we count parity.

    Returns:
        p (int): +1 if the number of '1' bits between i and j is even
            and -1 if it is odd.

    Examples:
    >>> get_parity(2, 0, 4)
    -1
    >>> get_parity(15, 1, 4)
    1
    >>> get_parity(15, 1, 3)
    -1

    """
    # Put i, j in order.
    i, j = sorted((i, j))

    bits = 0
    for k in range(i+1, j):
        bits += (s>>k)&1
    return 1-2*(bits%2)


@njit()
def get_parity_at_i(s, i):
    """Count the '1' bits in a state s from 0 (inclusive) to i.

    Args:
        s (int): state given by an integer number whose binary
            representation equals a state in the Fock space.
        i (int): site until which parity is counted.

    Returns:
        p (int): +1 if the number of '1' bits from 0 to i is even and
            -1 if it is odd.

    Examples:
    >>> get_parity_at_i(2, 3)
    -1
    >>> get_parity_at_i(15, 2)
    1
    >>> get_parity_at_i(7, 4)
    -1

    """
    bits = 0
    for k in range(i):
        bits += (s>>k)&1
    return 1-2*(bits%2)


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
