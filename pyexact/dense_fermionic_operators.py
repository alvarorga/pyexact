"""Scripts for computing exact diagonalizations."""

import numpy as np
from numba import njit, prange

from pyexact.bitwise_funcs import generate_states, get_parity


# Hard-core boson operators.


@njit(parallel=True)
def de_pc_op(L, N, J, D):
    """Build a many-body operator with 2-particle terms.

    Note: numba parallel does not support the enumerate function, which
    would make the code more readable. Instead we have to make the
    outermost loops with range().

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
    for ix_s in prange(num_states):
        s = states[ix_s]
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


@njit(parallel=True)
def de_sym_pc_op(L, N, J, D):
    """Build a many-body symmetric operator with 2-particle terms.

    Note: numba parallel does not support the enumerate function, which
    would make the code more readable. Instead we have to make the
    outermost loops with range().

    Args:
        L (int): system's length.
        N (int): particle number.
        J (2darray of floats): symmetric hopping matrix.
        D (2darray of floats): interaction matrix.

    Returns:
        H (2darray of floats): many-body operator.

    """
    if np.sum((J - J.T)**2) > 1e-7:
        raise ValueError('J is not symmetric.')

    # Put all elts of D in the lower triangle. Making a copy of D
    # instead of working with views prevents the function to make
    # changes in D outside of it.
    D = np.copy(D)
    for i in range(L):
        for j in range(i):  # j < i.
            D[i, j] += D[j, i]
            D[j, i] = 0

    states = generate_states(L, N)
    num_states = states.size

    H = np.zeros((num_states, num_states), np.float64)

    # Notation:
    #     s: initial state.
    #     t: final state.
    #     ix_#: index of #.
    for ix_s in prange(num_states):
        s = states[ix_s]
        for i in range(L):
            # On-site terms: n_i.
            if np.abs(J[i, i]) > 1e-7:
                if (s>>i)&1:
                    H[ix_s, ix_s] += J[i, i]

            for j in range(i):
                # Hopping terms: b^dagger_i b_j.
                if np.abs(J[i, j]) > 1e-7:
                    if not (s>>i)&1 and (s>>j)&1:
                        t = s + (1<<i) - (1<<j)
                        ix_t = np.where(states == t)[0][0]
                        H[ix_t, ix_s] += J[i, j]
                        H[ix_s, ix_t] += J[i, j]

                # Interaction terms: n_i n_j.
                if np.abs(D[i, j]) > 1e-7:
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
def de_pc_number_op(L, N, i):
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
def de_pc_correlator(L, N, i, j):
    """Build a many-body correlation operator b^dagger_i*b_j, i != j.

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
def de_pc_interaction(L, N, i, j):
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


@njit()
def de_npc_number_op(L, i):
    """Build a dense number nonconserving number operator.

    Args:
        L (int): system's length.
        i (int): site of the number operator.

    Returns:
        C (1darray of floats): many-body number operator.

    """
    num_states = 1<<L

    C = np.zeros(num_states, np.float64)

    for s in range(1<<L):
        if (s>>i)&1:
            C[s] += 1

    return C


@njit()
def de_npc_correlator(L, i, j):
    """Build a dense number nonconserving correlation b^dagger_i*b_j.

    Args:
        L (int): system's length.
        i (int): site of the creation operator.
        j (int): site of the annihilation operator.

    Returns:
        C (2darray of floats): many-body correlation operator.

    """
    if i == j:
        raise ValueError('i and j must be different.')

    num_states = 1<<L

    C = np.zeros((num_states, num_states), np.float64)

    for s in range(1<<L):
        if not (s>>i)&1 and (s>>j)&1:
            t = s + (1<<i) - (1<<j)
            C[t, s] += 1

    return C


@njit()
def de_npc_interaction(L, i, j):
    """Build a dense number nonconserving interaction n_i*n_j.

    Args:
        L (int): system's length.
        i (int): site of one number operator.
        j (int): site of the other number operator.

    Returns:
        C (1darray of floats): many-body interaction operator.

    """
    if i == j:
        raise ValueError('i and j must be different.')

    num_states = 1<<L

    C = np.zeros(num_states, np.float64)

    for s in range(1<<L):
        if (s>>i)&1 and (s>>j)&1:
            C[s] += 1

    return C


# Fermionic operators.


@njit(parallel=True)
def fer_de_pc_op(L, N, J, D):
    """Build a many-body fermionic operator with 2-particle terms.

    Note: numba parallel does not support the enumerate function, which
    would make the code more readable. Instead we have to make the
    outermost loops with range().

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
    for ix_s in prange(num_states):
        s = states[ix_s]
        for i in range(L):
            # On-site terms: n_i.
            if (s>>i)&1:
                H[ix_s, ix_s] += J[i, i]

            for j in range(L):
                if i != j:
                    # Hopping terms: b^dagger_i b_j.
                    if not (s>>i)&1 and (s>>j)&1:
                        t = s + (1<<i) - (1<<j)
                        par = get_parity(t, i, j)
                        ix_t = np.where(states == t)[0][0]
                        H[ix_t, ix_s] += par*J[i, j]

                    # Interaction terms: n_i n_j.
                    if (s>>i)&1 and (s>>j)&1:
                        H[ix_s, ix_s] += D[i, j]

    return H


@njit(parallel=True)
def fer_de_sym_pc_op(L, N, J, D):
    """Build a many-body symmetric fermionic operator.

    Note: numba parallel does not support the enumerate function, which
    would make the code more readable. Instead we have to make the
    outermost loops with range().

    Args:
        L (int): system's length.
        N (int): particle number.
        J (2darray of floats): symmetric hopping matrix.
        D (2darray of floats): interaction matrix.

    Returns:
        H (2darray of floats): many-body operator.

    """
    if np.sum((J - J.T)**2) > 1e-7:
        raise ValueError('J is not symmetric.')

    # Put all elts of D in the lower triangle. Making a copy of D
    # instead of working with views prevents the function to make
    # changes in D outside of it.
    D = np.copy(D)
    for i in range(L):
        for j in range(i):  # j < i.
            D[i, j] += D[j, i]
            D[j, i] = 0

    states = generate_states(L, N)
    num_states = states.size

    H = np.zeros((num_states, num_states), np.float64)

    # Notation:
    #     s: initial state.
    #     t: final state.
    #     ix_#: index of #.
    for ix_s in prange(num_states):
        s = states[ix_s]
        for i in range(L):
            # On-site terms: n_i.
            if np.abs(J[i, i]) > 1e-7:
                if (s>>i)&1:
                    H[ix_s, ix_s] += J[i, i]

            for j in range(i):
                # Hopping terms: b^dagger_i b_j.
                if np.abs(J[i, j]) > 1e-7:
                    if not (s>>i)&1 and (s>>j)&1:
                        t = s + (1<<i) - (1<<j)
                        par = get_parity(t, i, j)
                        ix_t = np.where(states == t)[0][0]
                        H[ix_t, ix_s] += par*J[i, j]
                        H[ix_s, ix_t] += par*J[i, j]

                # Interaction terms: n_i n_j.
                if np.abs(D[i, j]) > 1e-7:
                    if (s>>i)&1 and (s>>j)&1:
                        H[ix_s, ix_s] += D[i, j]

    return H
