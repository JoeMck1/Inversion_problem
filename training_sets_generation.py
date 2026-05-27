import logging
import os
import time
from typing import Callable, Literal

import h5py
import numpy as np
from numpy.typing import ArrayLike

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class TrainingSet:
    def __init__(
        self,
        energies: list,
        n_b: int,
        n_ch_nodes: int,
        sigma: float | tuple,
        lattice_data: np.ndarray,
        seed: int,
    ) -> None:
        self.energies = energies
        self.e_max = energies[-1]

        self.n_b = n_b  # number of basis functions.
        self.n_ch_nodes = n_ch_nodes  # number of Chebyshev nodes.
        self.sigma = sigma  # smearing parameter.

        self.lattice_data = lattice_data  # shape (bootstraps, time-slices)
        self.lattice_mean = np.mean(self.lattice_data, axis=0)
        self.cov_mat = np.cov(self.lattice_data, rowvar=False, bias=False)
        self.n_t = self.lattice_data.shape[1]
        self.time_slices = np.arange(1, self.n_t + 1)

        # deterministic RNG per instance
        self.rng = np.random.default_rng(seed=seed)
        self.e_0 = self.rng.uniform(0.2, 1.3)  # GeV

        # precompute once
        self.coeffs = self._random_coefficients()
        self.nodes = self._chebyshev_nodes()
        self.x_nodes = self._mapped_chebyshev_nodes(self.nodes)

        self._T_matrix = np.array(
            [self.t_n(n)(self.x_nodes) for n in range(self.n_b)]
        )  # shape (n_b, n_ch_nodes)
        self._T_min = np.array(
            [self.t_n(n)(self.exp_map_min()) for n in range(self.n_b)]
        )  # shape (n_b,)
        pass

    # ── static helpers ──────────────────────────────────────────────────────

    @staticmethod
    def exp_map(e: ArrayLike) -> np.ndarray:
        return 1 - 2 * np.exp(-np.asarray(e))

    def exp_map_max(self) -> np.floating:
        return self.exp_map(self.e_max)

    def exp_map_min(self) -> np.floating:
        return self.exp_map(self.e_0)

    @staticmethod
    def t_n(n: int) -> Callable:
        return np.polynomial.chebyshev.Chebyshev.basis(n)

    # ── private helpers ────────────────────────────────────────────────────

    def _mapped_chebyshev_nodes(self, nodes: np.ndarray) -> np.ndarray:
        return (
            (nodes + 1) * (self.exp_map_max() - self.exp_map_min()) / 2
        ) + self.exp_map_min()

    def _chebyshev_nodes(self) -> np.ndarray:

        # use range(1, self.n_nodes + 1) and not range(self.n_nodes)
        # because the Chebyshev-Gauss quadrature is defined for i = 1
        # to n.

        i = np.arange(1, self.n_ch_nodes + 1)
        return np.cos(((np.pi * ((2 * i) - 1)) / (2 * self.n_ch_nodes)))

    def _random_coefficients(self):
        epsilon = 1e-7
        rns = self.rng.uniform(-1, 1, size=self.n_b)  # use seeded rng
        coeffs = np.empty(self.n_b)
        coeffs[0] = rns[0]
        coeffs[1:] = rns[1:] / (np.arange(1, self.n_b) ** (1 + epsilon))
        return coeffs

    def _k_sigma_matrix(self, sig) -> np.ndarray:
        omega = -np.log((1 - self.x_nodes) / 2)[:, None]
        # omega has shape (n_ch_nodes, 1)
        energies = self.energies[None, :]  # shape (1, n_E)
        return np.exp(-((energies - omega) ** 2) / (2 * sig**2)) / (
            sig * np.sqrt(2 * np.pi)
        )  # has shape (n_ch_nodes, n_E)

    def _b_T_matrix(self) -> np.ndarray:
        omega = -np.log((1 - self.x_nodes) / 2)[:, None]
        # omega has shape (n_ch_nodes, 1)
        times = self.time_slices[None, :]  # shape (1, n_T)

        return ((omega**2) / (12 * np.pi**2)) * (
            np.exp(-omega * times) + np.exp(-omega * (self.n_t - times))
        )  # shape (n_ch_nodes, n_T)

    def _integration(self, function) -> np.ndarray:
        # function is smearing kernel or basis function b_T

        prefactor = (
            (self.exp_map_max() - self.exp_map_min())
            * np.sqrt(1 - self.nodes**2)
            / (2 * (1 - self.x_nodes))
        )  # prefactor shape (n_ch_nodes, 1)

        weighted = prefactor[:, None] * function  # (n_ch_nodes, n_E or n_T)

        T_shifted = self._T_matrix - self._T_min[:, None]
        result = (np.pi / self.n_ch_nodes) * T_shifted @ weighted

        return result  # shape (n_b, n_E or n_T)

    # ── public methods ─────────────────────────────────────────────────────

    def smeared_spectral_densities(self) -> list[np.ndarray]:
        return [
            self.coeffs @ self._integration(self._k_sigma_matrix(sig))
            for sig in self.sigma
        ]

    def noisy_correlator(self) -> np.ndarray:
        j_n_matrix = self._integration(self._b_T_matrix())
        correlator = self.coeffs @ j_n_matrix  # shape (n_t,)

        # Add in noise
        normalisation = (correlator[0] / self.lattice_mean[0]) ** 2

        covariance = normalisation * self.cov_mat
        noisy_corr = self.rng.multivariate_normal(correlator, covariance)

        return noisy_corr  # shape (n_t,)

    def spectral_densities(self) -> list[float]:
        mapped_Es = self.exp_map(self.energies)
        x_min = self.exp_map_min()
        densities = []
        for e in mapped_Es:
            if e <= x_min:
                densities.append(0.0)
                continue
            val = sum(
                self.coeffs[n] * (self.t_n(n + 1)(e) - self.t_n(n + 1)(x_min))
                for n in range(self.n_b)
            )

            densities.append(float(val))

        return densities


# ── HDF5 generation ───────────────────────────────────────────────────────────


def generate_all_training_sets(
    lattice_data: np.ndarray,
    energies: np.ndarray,
    sigmas: list[float],
    n_b_values: list[int],
    n_rho_values: list[int],
    n_ch_nodes: int,
    output_dir: str = "training_sets",
    overwrite: bool = False,
) -> None:
    """Generate all training sets from the paper and save to HDF5.

    For each N_b, one HDF5 file is written containing ALL N_rho levels.
    Smaller N_rho sets are stored as contiguous slices of the largest one,
    so the full 800k array is generated once and sliced — no data is
    duplicated on disk.

    Parameters
    ----------
    lattice_data : ndarray, shape (n_configs, n_T)
        Raw lattice correlator configurations used for noise injection
        and covariance estimation.
    energies : ndarray, shape (n_E,)
        Output energy grid in GeV.
    sigmas : list[float]
        Smearing widths in GeV, e.g. [0.44, 0.63].
    n_b_values : list[int]
        Numbers of basis functions, e.g. [16, 32, 128, 512].
    n_rho_values : list[int]
        Training set sizes in ascending order,
        e.g. [50_000, 100_000, 200_000, 400_000, 800_000].
    n_ch_nodes : int
        Number of Gauss-Chebyshev quadrature nodes.
    output_dir : str
        Directory in which HDF5 files are written.
    overwrite : bool
        If False (default), skip files that already exist on disk.

    File layout
    -----------
    <output_dir>/training_set_nb{n_b}_nrho{n_rho_max}.h5

        /metadata/
            sigmas          float64 (n_sigma,)
            n_b             int64   scalar
            n_rho_max       int64   scalar      largest set stored
            n_rho_levels    int64   (n_levels,) all N_rho levels present
            n_ch_nodes      int64   scalar
            energies        float64 (n_E,)
            time_slices     float64 (n_T,)
            e0_values       float64 (n_rho_max,) per-sample E0 thresholds

        /sigma_{sigma_str}/
            correlators     float64 (n_rho_max, n_T)
            smeared_rho     float64 (n_rho_max, n_E)

    To load the subset for N_rho = k, simply slice [:k] from each dataset.
    """

    os.makedirs(output_dir, exist_ok=True)

    n_rho_values = sorted(n_rho_values)  # ensure ascending
    n_rho_max = n_rho_values[-1]
    n_E = len(energies)
    n_T = lattice_data.shape[1]

    log.info(
        "Generating training sets: N_b=%s, N_rho levels=%s, sigmas=%s",
        n_b_values,
        n_rho_values,
        sigmas,
    )
    log.info("Output directory: %s", os.path.abspath(output_dir))

    total_files = len(n_b_values)
    for file_idx, n_b in enumerate(n_b_values, 1):

        fname = os.path.join(
            output_dir, f"training_set_nb{n_b}_nrho{n_rho_max}.h5"
        )

        if os.path.exists(fname) and not overwrite:
            log.info(
                "[%d/%d] Skipping %s (already exists)",
                file_idx,
                total_files,
                fname,
            )
            continue

        log.info(
            "[%d/%d] Generating N_b=%d, N_rho_max=%d ...",
            file_idx,
            total_files,
            n_b,
            n_rho_max,
        )
        t_start = time.perf_counter()

        # Pre-allocate arrays for all samples across all sigmas
        all_correlators = np.empty((n_rho_max, n_T), dtype=np.float64)
        all_smeared = {
            sig: np.empty((n_rho_max, n_E), dtype=np.float64) for sig in sigmas
        }
        all_e0 = np.empty(n_rho_max, dtype=np.float64)

        for rho_idx in range(n_rho_max):
            # Deterministic, well-separated seed per (n_b, rho_idx)
            seed = int(n_b * 1_000_000 + rho_idx)

            ts = TrainingSet(
                energies=energies,
                n_b=n_b,
                n_ch_nodes=n_ch_nodes,
                sigma=sigmas,
                lattice_data=lattice_data,
                seed=seed,
            )

            all_correlators[rho_idx] = ts.noisy_correlator()
            sds = ts.smeared_spectral_densities()  # list of (n_E,)
            for sig, sd in zip(sigmas, sds):
                all_smeared[sig][rho_idx] = sd
            all_e0[rho_idx] = ts.e_0

            if (rho_idx + 1) % 10_000 == 0:
                elapsed = time.perf_counter() - t_start
                rate = (rho_idx + 1) / elapsed
                remaining = (n_rho_max - rho_idx - 1) / rate
                log.info(
                    "  N_b=%d  %d/%d  (%.0f pairs/s, ~%.0fs remaining)",
                    n_b,
                    rho_idx + 1,
                    n_rho_max,
                    rate,
                    remaining,
                )

        # ── Write HDF5 ────────────────────────────────────────────────────────
        with h5py.File(fname, "w") as f:

            # ── metadata ─────────────────────────────────────────────────────
            meta = f.create_group("metadata")
            meta.create_dataset("sigmas", data=np.array(sigmas))
            meta.create_dataset("n_b", data=n_b)
            meta.create_dataset("n_rho_max", data=n_rho_max)
            meta.create_dataset(
                "n_rho_levels", data=np.array(n_rho_values, dtype=np.int64)
            )
            meta.create_dataset("n_ch_nodes", data=n_ch_nodes)
            meta.create_dataset("energies", data=energies)
            meta.create_dataset(
                "time_slices", data=np.arange(1, n_T + 1, dtype=np.float64)
            )
            meta.create_dataset("e0_values", data=all_e0)

            # Human-readable attributes on the root group
            f.attrs["description"] = (
                "Training set for Buzzicotti et al. (2024) spectral density extraction. "
                "Slice datasets to [:n_rho] to obtain the subset for a given N_rho level."
            )
            f.attrs["n_b"] = n_b
            f.attrs["n_rho_max"] = n_rho_max
            f.attrs["sigmas"] = str(sigmas)

            # ── data, one group per sigma ─────────────────────────────────────
            for sig in sigmas:
                sig_str = str(sig).replace(".", "p")  # e.g. "0.44" -> "0p44"
                grp = f.create_group(f"sigma_{sig_str}")
                grp.attrs["sigma"] = sig

                grp.create_dataset(
                    "correlators",
                    data=all_correlators,
                    compression="gzip",
                    compression_opts=4,
                    chunks=(min(1000, n_rho_max), n_T),
                )
                grp.create_dataset(
                    "smeared_rho",
                    data=all_smeared[sig],
                    compression="gzip",
                    compression_opts=4,
                    chunks=(min(1000, n_rho_max), n_E),
                )

        elapsed = time.perf_counter() - t_start
        size_mb = os.path.getsize(fname) / 1024**2
        log.info(
            "  Saved %s  (%.1f MB, %.1fs)",
            fname,
            size_mb,
            elapsed,
        )

    log.info("All training sets complete.")


# ── Convenience loader ──────────────────────────────────────────────────────


def load_training_set(
    filepath: str,
    sigma: float,
    n_rho: int | None = None,
) -> dict:
    """Load a training set from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    sigma : float
        Smearing width to load, e.g. 0.44.
    n_rho : int or None
        Number of pairs to load. If None, loads all pairs in the file.

    Returns
    -------
    dict with keys:
        correlators   : ndarray (n_rho, n_T)
        smeared_rho   : ndarray (n_rho, n_E)
        energies      : ndarray (n_E,)
        e0_values     : ndarray (n_rho,)
        n_b           : int
        sigma         : float
        n_rho_levels  : ndarray  all N_rho levels stored in the file
    """
    sig_str = str(sigma).replace(".", "p")

    with h5py.File(filepath, "r") as f:
        n_rho_max = int(f["metadata"]["n_rho_max"][()])
        if n_rho is None:
            n_rho = n_rho_max
        if n_rho > n_rho_max:
            raise ValueError(
                f"Requested n_rho={n_rho} exceeds n_rho_max={n_rho_max} in {filepath}"
            )

        grp = f[f"sigma_{sig_str}"]
        return {
            "correlators": grp["correlators"][:n_rho],
            "smeared_rho": grp["smeared_rho"][:n_rho],
            "energies": f["metadata"]["energies"][()],
            "e0_values": f["metadata"]["e0_values"][:n_rho],
            "n_b": int(f["metadata"]["n_b"][()]),
            "sigma": sigma,
            "n_rho_levels": f["metadata"]["n_rho_levels"][()],
        }


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    muon_mass = 0.10566  # GeV
    e_max, e_min = 24 * muon_mass, muon_mass
    e_range = np.linspace(e_min, e_max, 47)

    sigma = [0.44, 0.63]  # GeV

    n_rhos = [50_000, 100_000, 200_000, 400_000, 800_000]
    n_bs = [16, 32, 128, 512]

    # lattice data from the 64 ensemble for the 2pi.g5.001 operator
    two_pt_data = np.load("2pt_data_64.npy")

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
                seed=n_b,
            )

            if smearing == "Smeared":
                densities = training_sets.smeared_spectral_densities()
                ax.set_ylabel(r"$\hat{\rho}_\sigma(E)$")

            elif smearing == "Unsmeared":
                densities = training_sets.spectral_densities()
                ax.set_ylabel(r"$\rho (E)$")

            ax.plot(e_range, densities, label=rf"$N_b$ = {n_b}")

        ax.set_xlabel(r"$E$ [GeV]")
        ax.set_title(
            rf"{smearing} Spectral Densities for different $N_b$ ($\sigma = 0.63$GeV, $E_0=0.8$, Seed = $32$)"
        )
        ax.legend()

        plt.show()

    generate_all_training_sets(
        lattice_data=two_pt_data,
        energies=e_range,
        sigmas=sigma,
        n_b_values=n_bs,
        n_rho_values=n_rhos,
        n_ch_nodes=1024,  # adjust based on your convergence test
        output_dir="training_sets",
        overwrite=False,
    )
