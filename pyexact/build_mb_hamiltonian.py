"""High-level function to build specific Hamiltonians."""

import numpy as np
from scipy.sparse import csr_matrix

from pyexact.dense_hardcore_operators import de_pc_op, de_sym_pc_op, de_npc_op
from pyexact.sparse_operators import sp_pc_op, sp_sym_pc_op, sp_npc_op
from pyexact.dense_fermionic_operators import fer_de_pc_op


def build_mb_hamiltonian(J, D, L, N=None, r=None, l=None, is_fermionic=False):
    r"""Build a full many body Hamiltonian.

    The supported Hamiltonians are of the form

    .. math::

        H = \sum_{ij} J_{ij}\ d^\dagger_i d_j + D_{ij}\ n_i n_j,

    where :math:`d^\dagger_i` can be either a hard-core boson creation
    operator :math:`b^\dagger_i` or a fermionic creation operator
    :math:`c^\dagger_i`, and :math:`n_i` is the usual number operator
    :math:`b^\dagger_i b_i` and :math:`c^\dagger_i c_i`.

    Args:
        J (2darray of floats): hopping matrix :math:`b^\dagger_i b_j`.
        D (2darray of floats): interaction matrix :math:`n_i n_j`.
        L (int): system's length.
        N (int, opt): number of particles. If None, particle number is
            not conserved.
        r (2darray of floats, opt): raising operator :math:`b^\dagger_i`.
        l (2darray of floats, opt): lowering operator :math:`b_i`.
        is_fermionic (bool, opt): True if Hamiltonian has fermionic
            statistics.

    Returns:
        H (CSR of floats or 2darray of floats): Hamiltonian matrix.

    """
    if ((np.shape(J) != (L, L)) or (np.shape(D) != (L, L))
            or (r is not None and np.size(r) != L)
            or (l is not None and np.size(l) != L)):
        raise ValueError(f'The size L = {L} mismatches the dimension of the'
                         ' J, D, r, and/or r matrices.')

    # Too many words but more explicit relations.
    is_N_conserved = True if N is not None else False
    is_H_sparse = True if L > 14 else False
    is_J_symmetric = True if np.allclose(J, J.T) else False

    # Initialize r and l to 0 if N is not conserved and these matrices
    # are not given.
    if not is_N_conserved:
        if r is None:
            r = np.zeros(L, np.float64)
        if l is None:
            l = np.zeros(L, np.float64)

    # Select the properties of the Hamiltonian and compute it.
    if not is_fermionic:
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
    else:
        H = fer_de_pc_op(L, N, J, D)

    return H
