# pyExact
Python library for computing exact diagonalizations of hard-core bosons
and fermionic systems.

## Dependencies

To use the library you need the following set of packages (in
parentheses the library versions that I used the last time when
testing, may not work with older ones):
* Numpy (1.14.3)
* Scipy (1.1.0)
* Numba (0.38.0)(if you cannot install Numba then you could remove all
    `@njit()` decorators above the functions, but the scripts will run
     much slower)
* Sphinx (1.7.4)(only for building the documentation)

## Documentation
I have not yet uploaded the docs to the Internet, but in the meantime
you can go to the *docs* directory and put in the console
```shell
$ make html
```
This will create a set of *.html* pages in the directory *docs/_build*
with the documentation, which explains the usage of the package. In
case you cannot see the docs, a brief summary of the usage is appended
here.

### Usage

* To build a full Hamiltonian use the function `build_mb_hamiltonian` in
module `build_mb_hamiltonian.py`. The function automatically takes care of
writing the Hamiltonian in a dense or sparse format in the most
efficient way.

* To compute the expected values of the matrices *P* and *D*, you have
to go to functions `compute_P` and `compute_D` in module
`compute_expected.py`.
