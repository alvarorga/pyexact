"""Benchmark the construction of dense full Hamiltonians."""

import sys
import time
import numpy as np

sys.path.append('../')
from pyexact import dense_operators


def benchmark_dense_symmetric_number_conserving_op(L, N):
    """Benchmark construction of sym number conserving operator."""
    J = np.random.rand(L, L)
    J = (J + J.T)/2
    D = np.random.rand(L, L)
    D = (D + D.T)/2
    # Run once to compile with numba.
    _ = dense_operators.de_sym_pc_op(L, N, J, D)
    t0 = time.time()
    for _ in range(3):
        _ = dense_operators.de_sym_pc_op(L, N, J, D)
    t1 = time.time()
    print('\nBenchmark: dense symmetric number conserving Hamiltonian.')
    print(f'Time: {((t1-t0)/3):4.2f} s')

    return


def benchmark_dense_number_conserving_op(L, N):
    """Benchmark construction of number conserving operator."""
    J = np.random.rand(L, L)
    D = np.random.rand(L, L)
    _ = dense_operators.de_pc_op(L, N, J, D)
    t0 = time.time()
    # Run once to compile with numba.
    for _ in range(3):
        _ = dense_operators.de_pc_op(L, N, J, D)
    t1 = time.time()
    print('\nBenchmark: dense number conserving Hamiltonian.')
    print(f'Time: {((t1-t0)/3):4.2f} s')

    return


if __name__ == '__main__':

    # System's length and particle number.
    L = 14
    N = L//2

    print(f'Running benchmarks with {L} sites and {N} particles.')

    benchmark_dense_symmetric_number_conserving_op(L, N)
    benchmark_dense_number_conserving_op(L, N)
