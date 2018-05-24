"""Benchmark the construction of dense full Hamiltonians."""

import sys
import time
import numpy as np

sys.path.append('../')
from pyexact.dense_operators import de_pc_op, de_sym_pc_op, de_npc_op


def benchmark_dense_symmetric_number_conserving_op(L, N):
    """Benchmark construction of sym number conserving operator."""
    J = np.random.rand(L, L)
    J = (J + J.T)/2
    D = np.random.rand(L, L)
    D = (D + D.T)/2
    # Run once to compile with numba.
    _ = de_sym_pc_op(L, N, J, D)
    t0 = time.time()
    for _ in range(3):
        de_sym_pc_op(L, N, J, D)
    t1 = time.time()
    print('\nBenchmark: dense symmetric number conserving Hamiltonian.')
    print(f'Time: {(t1-t0)/3:4.2f} s')

    return


def benchmark_dense_number_conserving_op(L, N):
    """Benchmark construction of number conserving operator."""
    J = np.random.rand(L, L)
    D = np.random.rand(L, L)
    _ = de_pc_op(L, N, J, D)
    t0 = time.time()
    # Run once to compile with numba.
    for _ in range(3):
        de_pc_op(L, N, J, D)
    t1 = time.time()
    print('\nBenchmark: dense number conserving Hamiltonian.')
    print(f'Time: {(t1-t0)/3:4.2f} s')

    return


def benchmark_dense_number_nonconserving_op(L):
    """Benchmark construction of number nonconserving operator."""
    J = np.random.rand(L, L)
    D = np.random.rand(L, L)
    r = np.random.rand(L)
    l = np.random.rand(L)
    _ = de_npc_op(L, J, D, r, l)
    t0 = time.time()
    # Run once to compile with numba.
    for _ in range(3):
        de_pc_op(L, N, J, D)
    t1 = time.time()
    print('\nBenchmark: dense number nonconserving Hamiltonian.')
    print(f'Time: {(t1-t0)/3:4.2f} s')

    return


if __name__ == '__main__':

    # Number conserving.
    # System's length and particle number.
    L = 14
    N = L//2

    print(f'Running number conserving benchmarks with {L} sites'
          f' and {N} particles.')

    benchmark_dense_symmetric_number_conserving_op(L, N)
    benchmark_dense_number_conserving_op(L, N)

    # Number nonconserving.
    # System's length.
    L = 14

    print(f'Running number nonconserving benchmarks with {L} sites.')

    benchmark_dense_number_nonconserving_op(L)
