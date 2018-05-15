"""High-level function to build specific Hamiltonians."""

from scipy.sparse import csr_matrix

from pyexact.dense_operators import build_mb_operator
from pyexact.sparse_operators import build_sparse_mb_operator
from pyexact.sparse_operators import build_sparse_symmetric_mb_operator

def build_mb_hamiltonian(J, D, L, N=None, R=R, L=L):
    """Build a full many body Hamiltonian.

    Args:
        J (2darray of floats): hopping matrix: terms b^dagger_i*b_j.
        D (2darray of floats): interaction matrix: terms n_i*n_j.
        L (int): system's length.
        N (int, opt): number of particles. If None, particle number is
            not conserved.
        R (2darray of floats): raising operator: terms b^dagger_i.
        L (2darray of floats): lowering operator: terms b_i.

    Returns:
        H (CSR 2darray of floats or 2darray of floats): Hamiltonian
            matrix.

    """
    is_N_conserved = True if N is not None else False
    is_H_sparse = True if L > 14 else False
    is_J_symmetric = True if np.allclose(J, J.T) else False

    if is_N_conserved:
        if is_H_sparse:
            # v, r, c, ns stand for values, rows, cols, and number of
            # many body states.
            if is_J_symmetric:
                # Make D also symmetric.
                D = (D + D.T)/2
                v, r, c, ns = build_sparse_symmetric_mb_operator(L, N, J, D)
            else:
                v, r, c, ns = build_sparse_mb_operator(L, N, J, D)
        H = csr_matrix((v, (r, c)), shape=(ns, ns))
        else:
            H = build_mb_operator(L, N, J, D)
