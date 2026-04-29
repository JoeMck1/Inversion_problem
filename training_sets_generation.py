import random
from typing import Callable

import numpy as np


class TrainingSets:
    def __init__(
        self, energies: list, n_b: int, n_nodes: int, sigma: float
    ) -> None:
        self.energies = energies
        self.e_max = energies[-1]

        self.n_b = n_b
        self.n_nodes = n_nodes

        # sigma is a 'resolution parameter'
        self.sigma = sigma

        random.seed(n_b)  # MIGHT NEED CHANGING
        self.e_0 = 0.8  # random.uniform(0.2, 1.3)  # GeV
        pass

    @staticmethod
    def exp_map(e):
        return 1 - 2 * np.exp(-e)

    def exp_map_max(self):
        x_max = self.exp_map(self.e_max)
        return x_max

    def exp_map_min(self):
        x_min = self.exp_map(self.e_0)
        return x_min

    @staticmethod
    def t_n(n: int) -> Callable:
        """Return the Chebyshev polynomial of the first kind T_n(x).

        Parameters
        ----------
        n : int
            The degree of the returned polynomial.

        Returns
        -------
        cheb_poly : callable
            A callable Chebyshev polynomial. Calling t_n(x) will yield
            the value of t_n at x.
        """

        cheb_poly = np.polynomial.chebyshev.Chebyshev.basis(n)
        return cheb_poly

    def t_n_min(self, n):
        cheb_min = self.t_n(n)(self.exp_map_min())
        return cheb_min

    def chebyshev_nodes(self):

        # use range(1, self.n_nodes + 1) and not range(self.n_nodes)
        # because the Chebyshev-Gauss quadrature is defined for i = 1
        # to n.
        nodes = [
            np.cos((np.pi * ((2 * i) - 1)) / (2 * self.n_nodes))
            for i in range(1, self.n_nodes + 1)
        ]
        return nodes

    def gui(self, node):
        x = (
            (node + 1) * (self.exp_map_max() - self.exp_map_min()) / 2
        ) + self.exp_map_min()
        return x

    def k_sigmas(self, node):
        """Gaussian smearing kernel"""
        omega = np.log((1 - self.gui(node)) / 2)

        kernel = np.exp(
            -((self.energies + omega) ** 2) / (2 * self.sigma**2)
        ) / (self.sigma * (2 * np.pi) ** (1 / 2))
        return kernel

    def i_n_Ej(self, n):

        i_n_terms = []
        for node in self.chebyshev_nodes():
            energy_terms = []
            for k_sigma in self.k_sigmas(node):
                prefactor = (
                    (self.exp_map_max() - self.exp_map_min())
                    * (1 - node**2) ** (1 / 2)
                ) / (2 * (1 - self.gui(node)))

                term = (
                    prefactor
                    * k_sigma
                    * (self.t_n(n)(self.gui(node)) - self.t_n_min(n))
                )
                energy_terms.append(term)
            i_n_terms.append(energy_terms)

        i_n = (np.pi / self.n_nodes) * np.sum(i_n_terms, axis=0)
        return i_n

    def random_coefficients(self):
        epsilon = 10e-7

        rns = [random.uniform(-1, 1) for _ in range(self.n_b)]
        coeffs = [rns[0]] + [
            rn / (idx + 1) ** (1 + epsilon) for idx, rn in enumerate(rns[1:])
        ]
        return coeffs

    def smeared_spectral_densities(self):

        i_n_matrix = np.array([self.i_n_Ej(n) for n in range(self.n_b)])

        smeared_densities = self.random_coefficients() @ i_n_matrix

        return smeared_densities

    def spectral_densities(self):
        mapped_Es = self.exp_map(self.energies)
        random_coeffs = self.random_coefficients()

        total_rhos = []
        for e in mapped_Es:
            if self.e_0 >= e:
                total_rhos.append(0)
                continue

            total_rho = sum(
                random_coeffs[n - 1] * (self.t_n(n)(e) - self.t_n_min(n))
                for n in range(1, self.n_b + 1)
            )
            total_rhos.append(total_rho)

        return total_rhos


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sigma = 0.5 / (2 * (np.log(4)))  # GeV
    muon_mass = 0.10566  # GeV
    e_max, e_min = 24 * muon_mass, muon_mass
    e_range = np.linspace(e_min, e_max, 47)

    n_b_values = [16, 32, 128, 512]

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Latin Modern Math"],
        }
    )
    fig, ax = plt.subplots()

    for n_b in n_b_values:
        training_sets = TrainingSets(
            energies=e_range, n_b=n_b, n_rho=10, n_nodes=50, sigma=sigma
        )

        smeared = training_sets.smeared_spectral_densities()

        ax.plot(e_range, smeared, label=rf"$N_b$ = {n_b}")

    ax.set_xlabel(r"$E$ [GeV]")
    ax.set_ylabel(r"$\hat{\rho}_\sigma(E)$")
    ax.set_title(r"Smeared Spectral Densities for different $N_b$")
    ax.legend()

    plt.show()
