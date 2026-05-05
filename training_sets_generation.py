from typing import Callable, Literal

import numpy as np


class TrainingSets:
    def __init__(
        self, energies: list, n_b: int, n_ch_nodes: int, sigma: float
    ) -> None:
        self.energies = energies
        self.e_max = energies[-1]

        self.n_b = n_b  # number of basis functions.
        self.n_ch_nodes = n_ch_nodes  # number of Chebyshev nodes.
        self.sigma = sigma  # sigma is a 'resolution parameter'.

        # self.rho = n_rho # number of random spectral densities.

        self.rng = np.random.default_rng(seed=n_b)  # MIGHT NEED CHANGING
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

    def gui(self, node):
        x = (
            (node + 1) * (self.exp_map_max() - self.exp_map_min()) / 2
        ) + self.exp_map_min()
        return x

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
            np.cos((np.pi * ((2 * i) - 1)) / (2 * self.n_ch_nodes))
            for i in range(1, self.n_ch_nodes + 1)
        ]
        return nodes

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

        i_n = (np.pi / self.n_ch_nodes) * np.sum(i_n_terms, axis=0)
        return i_n

    def random_coefficients(self):
        epsilon = 10e-7

        rns = self.rng.uniform(-1, 1, size=self.n_b)  # use seeded rng
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

            total_rho = np.heaviside((e - self.exp_map_min()), 0) * sum(
                random_coeffs[n - 1] * (self.t_n(n)(e) - self.t_n_min(n))
                for n in range(1, self.n_b + 1)
            )
            total_rhos.append(total_rho)

        return total_rhos


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    muon_mass = 0.10566  # GeV
    e_max, e_min = 24 * muon_mass, muon_mass
    e_range = np.linspace(e_min, e_max, 47)

    sigma = [0.44, 0.63]  # GeV

    # The n_rho and n_b values used to define the training sets in the
    # paper are given below.
    # At fixed n_b each training set includes the smaller ones i.e.
    # T_sig(N_b, 100e3) includes T_sig(N_b, 50e3).
    n_rhos = [50e3, 100e3, 200e3, 400e3, 800e3]
    n_bs = [16, 32, 128, 512]

    # training_sets = TrainingSets(
    #    energies=e_range, n_b=32, n_ch_nodes=10, sigma=sigma[0]
    # )
    # training_sets.random_coefficients()
    # print(training_sets.e_0)

    def smeared_spectral_densities_plot(
        n_b_values, e_range, sigma, smearing: Literal["Smeared", "Unsmeared"]
    ):
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
                energies=e_range, n_b=n_b, n_ch_nodes=50, sigma=sigma
            )

            if smearing == "Smeared":
                densities = training_sets.smeared_spectral_densities()
                ax.set_ylabel(r"$\hat{\rho}_\sigma(E)$")

            elif smearing == "Unsmeared":
                densities = training_sets.spectral_densities()
                ax.set_ylabel(r"$\rho (E)$")

            ax.plot(e_range, densities, label=rf"$N_b$ = {n_b}")

        ax.set_xlabel(r"$E$ [GeV]")
        ax.set_title(rf"{smearing} Spectral Densities for different $N_b$")
        ax.legend()

        plt.show()

    smeared_spectral_densities_plot(
        n_b_values=n_bs,
        e_range=e_range,
        sigma=sigma[1],
        smearing="Unsmeared",
    )
