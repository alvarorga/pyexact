# pyExact
Python library for computing exact diagonalizations of hard-core bosons
or spin systems. A patch for fermionic systems will be done when the
full library is completed.

## Dependencies

To use the library you need the following set of packages
* Numpy
* Scipy
* Numba (you could remove all `@njit()` decorators above the functions, but the scripts will run much slower)
* Sphinx (only for building the documentation)

## Documentation
I have not yet uploaded the docs to the Internet, but in the meantime
you can go to the *docs* directory and put in the console
```shell
$ make html
```
This will create a set of *.html* pages in the directory *docs/_build*
with the documentation, which explains the usage of the package.

## Usage

* To build a full Hamiltonian use the function `build_mb_hamiltonian` in
module `build_hamiltonian.py`. The function automatically takes care of
writing the Hamiltonian in a dense or sparse format, so you do not have
to specify.

* To compute the expected values of the matrices *P* and *D*, you have
to go to functions `compute_P` and `compute_D` in module
`compute_expected.py`.
