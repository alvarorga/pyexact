# pyexact
Python library for computing exact diagonalizations of hard-core bosons
or spin systems. A patch for fermionic systems will be done when the
full library is completed.

## Usage

* To build a full Hamiltonian use the function `build_mb_hamiltonian` in
module `build_hamiltonian.py`. The function automatically takes care of
writing the Hamiltonian in a dense or sparse format, so you do not have
to specify.

* To compute the expected values of the matrices *P* and *D*, you have
to go to functions `compute_P` and `compute_D` in module
`compute_expected.py`. 

## Notation
The names of the functions that assemble the full Hamiltonians have
been shortened due to their long nature. The names are subject to the
following notation:

* `de_` and `sp_`: dense and sparse operators.
* `sym_`: the hopping matrix `J` must be symmetric.
* `npc_` and `pc_`: does not conserve particle number and conserves it.
* `_op`: common ending notation for all functions.
