"""Test entropy functions."""


import sys
import unittest
import numpy as np

sys.path.append('../')
from pyexact.entropy import (
    compute_entropy_singular_vals, compute_entanglement_entropy,
    compute_entanglement_spectrum
    )


class SVDValsTestCase(unittest.TestCase):
    """Test the computation of the singular values."""

    def test_svals_one(self):
        """Test singular values."""
        L = 4
        N = 2
        psi = np.zeros(6)
        psi[0] = 1/np.sqrt(2)
        psi[-1] = 1/np.sqrt(2)
        i = 2

        svals = compute_entropy_singular_vals(psi, L, N, i)
        self.assertTrue(np.allclose(svals, [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]))

    def test_svals_two(self):
        """Test singular values."""
        L = 4
        N = 2
        psi = np.zeros(6)
        psi[0] = 1/np.sqrt(2)
        psi[1] = 1/2
        psi[4] = 1/2
        i = 2

        svals = compute_entropy_singular_vals(psi, L, N, i)
        self.assertTrue(np.allclose(svals, [1/np.sqrt(2), 1/2, 1/2, 0]))

    def test_svals_three(self):
        """Test singular values."""
        L = 4
        N = 2
        psi = np.zeros(6)
        psi[0] = 1/np.sqrt(2)
        psi[2] = 1/2
        psi[3] = 1/2
        i = 2

        svals = compute_entropy_singular_vals(psi, L, N, i)
        self.assertTrue(np.allclose(svals, [1/np.sqrt(2), 1/2, 1/2, 0]))

    def test_svals_four(self):
        """Test singular values."""
        L = 4
        N = 2
        psi = np.zeros(6)
        psi[1] = 1/2
        psi[3] = -np.sqrt(3)/2
        i = 1

        svals = compute_entropy_singular_vals(psi, L, N, i)
        self.assertTrue(np.allclose(svals, [1/2, np.sqrt(3)/2]))


class EntanglementEntropyTestCase(unittest.TestCase):
    """Test the computation of the entanglement entropy."""

    def test_ent_entropy_one(self):
        """Test entanglement entropy."""
        L = 4
        N = 2
        psi = np.zeros(6)
        psi[0] = 1/np.sqrt(2)
        psi[-1] = 1/np.sqrt(2)
        i = 2

        # nonzero svals = 1/np.sqrt(2), 1/np.sqrt(2).
        S = compute_entanglement_entropy(psi, L, N, i)
        self.assertAlmostEqual(S, 1/np.sqrt(2))

    def test_ent_entropy_two(self):
        """Test entanglement entropy."""
        L = 4
        N = 2
        psi = np.zeros(6)
        psi[0] = 1/np.sqrt(2)
        psi[1] = 1/2
        psi[4] = 1/2
        i = 2

        # nonzero svals = 1/np.sqrt(2), 1/2, 1/2.
        S = compute_entanglement_entropy(psi, L, N, i)
        self.assertAlmostEqual(S, 1 + 1/(2*np.sqrt(2)))


class EntanglementSpectrumTestCase(unittest.TestCase):
    """Test the computation of the entanglement spectrum."""

    def test_ent_spectrum_one(self):
        """Test entanglement spectrum."""
        L = 4
        N = 2
        psi = np.zeros(6)
        psi[0] = 1/np.sqrt(2)
        psi[-1] = 1/np.sqrt(2)
        i = 2

        # nonzero svals = 1/np.sqrt(2), 1/np.sqrt(2).
        es = compute_entanglement_spectrum(psi, L, N, i)
        self.assertTrue(np.allclose(es, [np.log(2)/2, np.log(2)/2]))

    def test_ent_spectrum_two(self):
        """Test entanglement spectrum."""
        L = 4
        N = 2
        psi = np.zeros(6)
        psi[0] = 1/np.sqrt(2)
        psi[1] = 1/2
        psi[4] = 1/2
        i = 2

        # nonzero svals = 1/np.sqrt(2), 1/2, 1/2.
        es = compute_entanglement_spectrum(psi, L, N, i)
        self.assertTrue(np.allclose(es, [np.log(2)/2, np.log(2), np.log(2)]))
