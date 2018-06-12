"""Benchmark the construction of sparse full Hamiltonians."""

import sys
import time
import numpy as np

sys.path.append('../')
from pyexact.sparse_operators import sp_pc_op, sp_sym_pc_op


def benchmark_sparse_symmetric_number_conserving_op(L, N):
    """Benchmark construction of sym number conserving operator."""
    J = np.random.rand(L, L)
    J = (J + J.T)/2
    D = np.random.rand(L, L)
    D = (D + D.T)/2
    # Run once to compile with numba.
    _, _, _, _ = sp_sym_pc_op(L, N, J, D)
    t0 = time.time()
    for _ in range(3):
        _, _, _, _ = sp_sym_pc_op(L, N, J, D)
    t1 = time.time()
    print('\nBenchmark: sparse symmetric number conserving Hamiltonian.')
    print(f'Time: {(t1-t0)/3:4.2f} s')

    return


def benchmark_sparse_number_conserving_op(L, N):
    """Benchmark construction of number conserving operator."""
    J = np.random.rand(L, L)
    D = np.random.rand(L, L)
    _, _, _, _ = sp_pc_op(L, N, J, D)
    t0 = time.time()
    # Run once to compile with numba.
    for _ in range(3):
        _, _, _, _ = sp_pc_op(L, N, J, D)
    t1 = time.time()
    print('\nBenchmark: sparse number conserving Hamiltonian.')
    print(f'Time: {(t1-t0)/3:4.2f} s')

    return


if __name__ == '__main__':

    # System's length and particle number.
    L = 16
    N = L//2

    print(f'Running benchmarks with {L} sites and {N} particles.')

    benchmark_sparse_symmetric_number_conserving_op(L, N)
    benchmark_sparse_number_conserving_op(L, N)
