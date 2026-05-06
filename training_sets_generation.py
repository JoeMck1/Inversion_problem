from typing import Callable, Literal
from numpy.typing import ArrayLike
import numpy as np


class TrainingSet:
    def __init__(
        self,
        energies: list,
        n_b: int,
        n_ch_nodes: int,
        sigma: float,
        lattice_data: np.ndarray,
    ) -> None:
        self.energies = energies
        self.e_max = energies[-1]

        self.n_b = n_b  # number of basis functions.
        self.n_ch_nodes = n_ch_nodes  # number of Chebyshev nodes.
        self.sigma = sigma  # sigma is a 'resolution parameter'.

        self.lattice_data = lattice_data  # shape (bootstraps, time-slices)
        self.cov_mat = np.cov(self.lattice_data, rowvar=False, bias=False)
        self.n_t = len(self.lattice_data[0])  # number of time-slices

        self.time_slices = np.arange(1, self.n_t + 1)

        # self.rho = n_rho # number of random spectral densities.

        # deterministic RNG per instance
        self.rng = np.random.default_rng(seed=n_b)  # MIGHT NEED CHANGING
        self.e_0 = 0.8  # random.uniform(0.2, 1.3)  # GeV

        # precompute once
        self.coeffs = self._random_coefficients()
        self.nodes = self._chebyshev_nodes()

        # precompute mapped Chebyshev nodes once
        self.x_nodes = self._mapped_chebyshev_nodes(self.nodes)
        pass

    @staticmethod
    def exp_map(e: ArrayLike) -> np.ndarray:
        """Return an exponential mapping of the input e.

        This is a function used to map energies to the [-1, 1] domain.

        Parameters
        ----------
        e: ArrayLike
            The input energy or energies.

        Returns
        -------
        x : ndarray
            Array of the mapped energy value(s).
        """

        x = 1 - 2 * np.exp(-e)
        return x

    def exp_map_max(self) -> np.floating:
        """Return the exponential mapping of the maximum input energy.

        Returns
        -------
        x_max : floating
            The np.floating of the mapped E_max value.
        """
        x_max = self.exp_map(self.e_max)
        return x_max

    def exp_map_min(self) -> np.floating:
        """Return the exponential mapping of the minimum input energy.

        NOTE: in the literature E_min is denoted E_0.

        Returns
        -------
        x_min : floating
            The np.floating of the mapped E_min value.
        """
        x_min = self.exp_map(self.e_0)
        return x_min

    def _mapped_chebyshev_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Return the mapped Chebyshev nodes.

        This mapping arises from the map needed for mapping the
        [x_0, x_max] domain to [-1, 1].

        Parameters
        ----------
        nodes : ndarray
            Array of Chebyshev nodes to be mapped.

        Returns
        -------
        x : ndarray
            Array of mapped Chebyshev nodes.
        """
        x = (
            (nodes + 1) * (self.exp_map_max() - self.exp_map_min()) / 2
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

    def _chebyshev_nodes(self) -> np.ndarray:
        """Generate an array of Chebyshev nodes.

        Returns
        -------
        nodes: ndarray
            An array of Chebyshev nodes, given by:
            x_i = cos(pi(2i -1)/2N), where N is the number of
            Chebyshev nodes. We are free to choose this number.
        """

        # use range(1, self.n_nodes + 1) and not range(self.n_nodes)
        # because the Chebyshev-Gauss quadrature is defined for i = 1
        # to n.

        i = np.arange(1, self.n_ch_nodes + 1)
        nodes = np.cos(((np.pi * ((2 * i) - 1)) / (2 * self.n_ch_nodes)))

        return nodes

    def _random_coefficients(self):
        epsilon = 10e-7
        rns = self.rng.uniform(-1, 1, size=self.n_b)  # use seeded rng

        coeffs = np.empty(self.n_b)
        coeffs[0] = rns[0]
        coeffs[1:] = rns[1:] / (np.arange(1, self.n_b) ** (1 + epsilon))
        return coeffs

    def _k_sigma_matrix(self) -> np.ndarray:
        """Generate Gaussian smearing kernel.

        Returns a matrix of values for the Gaussian smearing kernel.
        This is matrix since the kernel depend on energy and the
        integration variable omega, which gets mapped to x and then to
        g(u).

        Returns
        -------
        kernel : ndarray
            A 2-dimensional array of shape
            (number of Chebyshev nodes, number of energies).
        """
        omega = -np.log((1 - self.x_nodes) / 2)[:, None]  # shape (nodes, 1)
        energies = self.energies[None, :]  # shape (1, energies)

        kernel = np.exp(-((energies - omega) ** 2) / (2 * self.sigma**2)) / (
            self.sigma * (2 * np.pi) ** (1 / 2)
        )  # shape (nodes, energies)
        return kernel

    def _b_T_matrix(self):
        omega = -np.log((1 - self.x_nodes) / 2)[:, None]  # shape (nodes, 1)
        times = self.time_slices[None, :]  # shape (1, times)

        basis = ((omega**2) / (12 * np.pi**2)) * (
            np.exp(-omega * times) + np.exp(-omega * (self.n_t - times))
        )
        return basis  # shape (nodes, times)

    def _integration(self, function):

        prefactor = (
            (self.exp_map_max() - self.exp_map_min())
            * np.sqrt(1 - self.nodes**2)
            / (2 * (1 - self.x_nodes))
        )[
            :, None
        ]  # shape (nodes, 1)

        weighted_kernel = prefactor * function  # shape (nodes, energies)

        result = []

        for n in range(self.n_b):
            Tn_nodes = self.t_n(n)(self.x_nodes)[:, None]  # (nodes, 1)
            Tn_min = self.t_n(n)(self.exp_map_min())

            term = weighted_kernel * (Tn_nodes - Tn_min)
            i_n = (np.pi / self.n_ch_nodes) * term.sum(axis=0)

            result.append(i_n)

        return np.array(result)  # shape (n_b, n_E)

    def smeared_spectral_density(self):
        i_n_matrix = self._integration(self._k_sigma_matrix())
        smeared_density = self.coeffs @ i_n_matrix  # shape (n_E,)

        return smeared_density

    def noisy_correlator(self):
        j_n_matrix = self._integration(self._b_T_matrix())
        correlator = self.coeffs @ j_n_matrix  # shape (n_t,)

        # ---- Add in noise ----
        normalisation = (
            correlator[0] / np.mean(self.lattice_data, axis=0)[0]
        ) ** 2
        covariance = normalisation * self.cov_mat
        noisy_corr = self.rng.multivariate_normal(correlator, covariance)

        return noisy_corr

    def spectral_densities(self):
        mapped_Es = self.exp_map(self.energies)

        densities = []
        for e in mapped_Es:
            if e <= self.exp_map_min():
                densities.append(0)
                continue

            val = sum(
                self.coeffs[n]
                * (self.t_n(n + 1)(e) - self.t_n(n + 1)(self.exp_map_min()))
                for n in range(self.n_b)
            )

            densities.append(val)

        return densities


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    muon_mass = 0.10566  # GeV
    e_max, e_min = 24 * muon_mass, muon_mass
    e_range = np.linspace(e_min, e_max, 47)

    # lattice data from the 64 ensemble for the 2pi.g5.001 operator
    two_pt_data = np.load("2pt_data_64.npy")

    sigma = [0.44, 0.63]  # GeV

    # The n_rho and n_b values used to define the training sets in the
    # paper are given below.
    # At fixed n_b each training set includes the smaller ones i.e.
    # T_sig(N_b, 100e3) includes T_sig(N_b, 50e3).
    n_rhos = [50e3, 100e3, 200e3, 400e3, 800e3]
    n_bs = [16, 32, 128, 512]

    training_sets = TrainingSet(
        energies=e_range,
        n_b=32,
        n_ch_nodes=5000,
        sigma=sigma[0],
        lattice_data=two_pt_data,
    )

    output = training_sets.noisy_correlator()

    print(output.shape)
    print((output))

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
            training_sets = TrainingSet(
                energies=e_range,
                n_b=n_b,
                n_ch_nodes=50,
                sigma=sigma,
                lattice_data=two_pt_data,
            )

            if smearing == "Smeared":
                densities = training_sets.input_output_pair()[1]
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
