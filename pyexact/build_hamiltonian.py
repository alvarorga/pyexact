"""High-level function to build specific Hamiltonians."""

import numpy as np
from scipy.sparse import csr_matrix

from pyexact.dense_operators import de_pc_op, de_sym_pc_op, de_npc_op
from pyexact.sparse_operators import sp_pc_op, sp_sym_pc_op, sp_npc_op

def build_mb_hamiltonian(J, D, L, N=None, r=None, l=None):
    r"""Build a full many body Hamiltonian.

    The supported Hamiltonians are of the form

    .. math::

        H = \sum_{ij} J_{ij}\ b^\dagger_i b_j + D_{ij}\ n_i n_j.

    Args:
        J (2darray of floats): hopping matrix :math:`b^\dagger_i b_j`.
        D (2darray of floats): interaction matrix :math:`n_i n_j`.
        L (int): system's length.
        N (int, opt): number of particles. If None, particle number is
            not conserved.
        r (2darray of floats, opt): raising operator :math:`b^\dagger_i`.
        l (2darray of floats, opt): lowering operator :math:`b_i`.

    Returns:
        H (CSR 2darray of floats or 2darray of floats): Hamiltonian
            matrix.

    """
    if ((np.shape(J) != (L, L)) or (np.shape(D) != (L, L))
            or (np.size(r) != L) or (np.shape(l) != L)):
        raise ValueError(f'The size L = {L} mismatches the dimension of the'
                         ' J, D, r, and/or r matrices.')

    # Too many words but more explicit relations.
    is_N_conserved = True if N is not None else False
    is_H_sparse = True if L > 14 else False
    is_J_symmetric = True if np.allclose(J, J.T) else False

    if is_N_conserved:
        if is_H_sparse:
            if is_J_symmetric:
                v, r, c, ns = sp_sym_pc_op(L, N, J, D)
            else:
                v, r, c, ns = sp_pc_op(L, N, J, D)
            H = csr_matrix((v, (r, c)), shape=(ns, ns))
        else:
            if is_J_symmetric:
                H = de_sym_pc_op(L, N, J, D)
            else:
                H = de_pc_op(L, N, J, D)
    else:
        if is_H_sparse:
            v, r, c, ns = sp_npc_op(L, J, D, r, l)
            H = csr_matrix((v, (r, c)), shape=(ns, ns))
        else:
            H = de_npc_op(L, J, D, r, l)

    return H
