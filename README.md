# pyexact
Python library for computing exact diagonalizations of hard-core bosons
or spin systems. A patch for fermionic systems will be done when the
full library is completed.

## Notation
The names of the functions that assemble the full Hamiltonians have
been shortened due to their long nature. The names are subject to the
following notation:

* `de_` and `sp_`: dense and sparse operators.
* `sym_`: the hopping matrix `J` must be symmetric.
* `npc_` and `pc_`: does not conserve particle number and conserves it.
* `_op`: common ending notation for all functions.
