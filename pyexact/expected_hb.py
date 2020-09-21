"""Computation of expected values of hard-core boson states.

The computations are done in a faster way if they are basis independent."""

import numpy as np
from numba import njit
from pyexact.bitwise_funcs import binsearch

@njit()
def fcompute_1Di(basis, state, i):
    """Compute D_i."""
    out = 0.
    for ix_s, s in enumerate(basis):
        if (s>>i)&1:
            out += np.abs(state[s])**2
    return out

@njit()
def fcompute_2Dij(basis, state, i, j):
    """Compute D_ij."""
    if i == j:
        return fcompute_1Di(basis, state, i)

    out = 0.
    for ix_s, s in enumerate(basis):
        if ((s>>i)&1) and ((s>>j)&1):
            out += np.abs(state[s])**2
    return out

@njit()
def fcompute_3Dijk(basis, state, i, j, k):
    """Compute D_ijk."""
    if i == j:
        return fcompute_2Dij(basis, state, i, k)
    elif i == k or j == k:
        return fcompute_2Dij(basis, state, i, j)

    out = 0.
    for ix_s, s in enumerate(basis):
        if ((s>>i)&1) and ((s>>j)&1) and ((s>>k)&1):
            out += np.abs(state[s])**2
    return out

@njit()
def fcompute_2Pij(basis, state, i, j):
    """Compute P_ij."""
    if i == j:
        return fcompute_1Di(basis, state, i)

    out = 0.
    for ix_s, s in enumerate(basis):
        if not (s>>i)&1 and (s>>j)&1:
            t = s + (1<<i) - (1<<j)
            ix_t = binsearch(states, t)
            out += state[ix_s]*np.conj(state[ix_t])
    return out

@njit()
def fcompute_3Pijk(basis, state, i, j, k):
    """Compute P^i_jk."""
    if i == j or j == k:
        return 0.
    elif j == k:
        return fcompute_2Dij(basis, state, i, j)

    out = 0.
    for ix_s, s in enumerate(basis):
        if (s>>i)&1 and not (s>>j)&1 and (s>>k)&1:
            t = s + (1<<j) - (1<<k)
            ix_t = binsearch(states, t)
            out += state[ix_s]*np.conj(state[ix_t])
    return out

@njit()
def fcompute_4Rijkl(basis, state, i, j, k, l):
    """Compute P^ij_kl."""
    if i == j:
        return fcompute_3Pijk(basis, state, i, k, l)
    elif i == k or i == l or j == k or j == l:
        return 0.
    elif k == l:
        return fcompute_2Dijk(basis, state, i, j, k)

    out = 0.
    for ix_s, s in enumerate(basis):
        if (s>>i)&1 and (s>>j)&1 and not (s>>k)&1 and (s>>l)&1:
            t = s + (1<<k) - (1<<l)
            ix_t = binsearch(states, t)
            out += state[ix_s]*np.conj(state[ix_t])
    return out

@njit()
def fcompute_4Pijkl(basis, state, i, j, k, l):
    """Compute P_ijkl."""
    if i == j or k == l:
        return 0.
    elif i == k:
        return fcompute_3Pijk(basis, state, i, j, l)
    elif i == l:
        return fcompute_3Pijk(basis, state, i, j, k)
    elif j == k:
        return fcompute_3Pijk(basis, state, j, i, l)
    elif j == l:
        return fcompute_3Pijk(basis, state, j, i, k)

    out = 0.
    for ix_s, s in enumerate(basis):
        if not (s>>i)&1 and not (s>>j)&1 and (s>>k)&1 and (s>>l)&1:
            t = s + (1<<i) + (1<<j) - (1<<k) - (1<<l)
            ix_t = binsearch(states, t)
            out += state[ix_s]*np.conj(state[ix_t])
    return out
