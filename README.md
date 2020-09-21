# pyExact
Python library for computing exact diagonalizations of hard-core bosons
and fermionic systems.

## Dependencies

To use the library you need the following set of packages:
* Numpy
* Scipy
* Numba (if you cannot install Numba then you could remove all
    `@njit()` decorators above the functions, but the scripts will run
     much slower)

### Usage

* To build a full Hamiltonian use the function `build_mb_hamiltonian` in
module `build_mb_hamiltonian.py`. The function automatically takes care of
writing the Hamiltonian in a dense or sparse format in the most
efficient way.

* To compute the expected values of the matrices *P* and *D*, you have
to go to functions `compute_P` and `compute_D` in module
`compute_expected.py`.
