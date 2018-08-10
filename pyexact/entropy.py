"""Compute entanglement entropy."""

import numpy as np
from scipy.linalg import svdvals
# from numba import njit

from pyexact.bitwise_funcs import generate_states


def compute_entropy_singular_vals(psi, L, N, i):
    """Compute the singular values of a state decomposition.

    We divide the state in two parts given by a position and then
    do return its singular values. The space partition is done between
    i-1 and i.

    Args:
        psi (1darray of floats): state vector.
        L (int): number of lattice sites.
        N (int): number of particles.
        i (int): position where the state is partitioned.

    Returns:
        svals (1darray of floats): singular values.

    """
    svals = None

    # States in the whole lattice with N particles.
    states = generate_states(L, N)

    # Get the maximum and minimum number of particles that fit in the
    # subspace 0 to i-1 (both inclusive).
    num_min = N - min(L-i, N)
    num_max = min(i, N)

    for n in range(num_min, num_max+1):
        # Generate all states in the interval (0, i-1) with n
        # particles.
        a_states = generate_states(i, n)
        num_a_states = a_states.size
        # Generate all states in the interval (i, L-1) with N-n
        # particles.
        b_states = generate_states(L-i, N-n)
        num_b_states = b_states.size
        A = np.zeros((num_a_states, num_b_states), dtype=np.float64)

        for ia, a in enumerate(a_states):
            for ib, b in enumerate(b_states):
                # Tensor multiply a and b to produce a state in (0, L).
                ab = np.left_shift(a, L-i) + b
                A[ia, ib] = psi[np.nonzero(states == ab)]

        if n == num_min:
            svals = svdvals(A)
        else:
            svals = np.concatenate((svals, svdvals(A)))

    return svals


def compute_entanglement_entropy(psi, L, N, i):
    """Compute the entanglement entropy of a state.

    We divide the state in two parts between i-1 and i.

    Args:
        psi (1darray of floats): state vector.
        L (int): number of lattice sites.
        N (int): number of particles.
        i (int): position where the state is partitioned.

    Returns:
        S (float): entanglement entropy.

    """
    svals = compute_entropy_singular_vals(psi, L, N, i)
    # Remove 0 singular values.
    svals = svals[np.nonzero(svals)]
    S = -np.dot(svals, np.log2(svals))
    return S


def compute_entanglement_spectrum(psi, L, N, i):
    """Compute the entanglement spectrum of a state.

    We divide the state in two parts between i-1 and i.

    Args:
        psi (1darray of floats): state vector.
        L (int): number of lattice sites.
        N (int): number of particles.
        i (int): position where the state is partitioned.

    Returns:
        es (1darray of floats): entanglement spectrum without infinite
            values.

    """
    svals = compute_entropy_singular_vals(psi, L, N, i)
    # Remove 0 singular values.
    svals = svals[np.nonzero(svals)]
    es = -np.log(svals)
    return es
