import logging
import os
import time

import h5py
import numpy as np
from scipy.fft import dct

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class TrainingSetContext:
    """Computed once. Everything here is independent of n_b and rho_idx."""

    def __init__(self, energies, sigmas, n_ch_nodes, n_T):
        self.n_ch_nodes = n_ch_nodes
        self.energies = energies
        self.n_T = n_T
        self.time_slices = np.arange(1, n_T + 1)
        self.unfolded_n_T = 2 * n_T

        k = np.arange(n_ch_nodes)
        self.nodes = np.cos(np.pi * (k + 0.5) / n_ch_nodes)
        self.omega = -np.log((1 - self.nodes) / 2)  # E_of_x(nodes)
        self.weight = np.sqrt(1 - self.nodes**2) / (1 - self.nodes)

        t = self.time_slices[None, :]
        w = self.omega[:, None]
        self.b_T_matrix = (w**2 / (12 * np.pi**2)) * (
            np.exp(-w * t) + np.exp(-w * (self.unfolded_n_T - t))
        )

        E = energies[None, :]
        self.k_sigma_matrices = {
            sig: np.exp(-((E - w) ** 2) / (2 * sig**2))
            / (sig * np.sqrt(2 * np.pi))
            for sig in sigmas
        }


class TrainingSet:
    def __init__(
        self,
        ctx: TrainingSetContext,
        n_b: int,
        seed: int,
    ) -> None:
        self.ctx = ctx
        self.n_b = n_b  # number of basis functions.

        # deterministic RNG per instance
        self.rng = np.random.default_rng(seed=seed)
        self.e_0 = self.rng.uniform(0.2, 1.3)  # GeV
        self.coeffs = self._random_coefficients()

        step = np.heaviside(self.ctx.omega - self.e_0, 0)
        self._step_weight = (step * self.ctx.weight)[:, None]

        n_vals = np.arange(self.n_b + 1)
        self._T_xmin = np.cos(n_vals * np.arccos(self.x_min()))

    # ── static helpers ──────────────────────────────────────────────────────

    @staticmethod
    def x_of_E(E: np.ndarray) -> np.ndarray:
        return 1 - 2 * np.exp(-np.asarray(E))

    def x_min(self) -> np.floating:
        return self.x_of_E(self.e_0)

    # ── private helpers ────────────────────────────────────────────────────

    def _random_coefficients(self):
        epsilon = 1e-7
        rns = self.rng.uniform(-1, 1, size=self.n_b + 1)  # use seeded rng
        coeffs = np.empty(self.n_b + 1)
        coeffs[0] = rns[0]
        coeffs[1:] = rns[1:] / (np.arange(1, self.n_b + 1) ** (1 + epsilon))
        return coeffs

    def _integration(self, function):
        """
        result = pi/N sum^{n_b}_{n=1} c_n * (F_n - F_0 * T_n(x_min))

        F_n = sum^{N-1}_{k=0} T_n(x_k) * g(x_k)

        g(x) = Theta(-ln((1-x)/2) - E_0) * sqrt(1-x^2)/(1-x) * function

        function = b_T(t, omega) or K_sigma(E, omega)
        """
        # function is either K_sigma, or b_T

        g_xk = self._step_weight * function
        # g_xk shape (n_ch_modes, n_E or n_T)

        # Generate n_b + 1 as \rho is defined for n = 0 to n = n_b,
        # i.e. a total of n_b + 1 terms.
        # F_n shape (n-b + 1, n_E or n_T)
        F_n = dct(g_xk, type=2, axis=0)[: self.n_b + 1, :] / 2
        F_0 = F_n[0, :]  # shape (n_E or n_T,)
        prefactor = np.pi / self.ctx.n_ch_nodes

        result = prefactor * (
            self.coeffs @ (F_n - (self._T_xmin[:, None] * F_0[None, :]))
        )
        return result  # shape (n_E, or n_T,)

    # ── public methods ─────────────────────────────────────────────────────

    def smeared_spectral_densities(self) -> list[np.ndarray]:
        return [
            self._integration(matrix)
            for matrix in self.ctx.k_sigma_matrices.values()
        ]

    def correlator(self) -> np.ndarray:
        return self._integration(self.ctx.b_T_matrix)

    def relative_noise(self, lattice_data) -> np.ndarray:
        correlator = self.correlator()

        lattice_mean = np.mean(lattice_data, axis=0)
        cov_mat = np.cov(lattice_data, rowvar=False, bias=False)

        # Add in noise
        normalisation = (correlator[0] / lattice_mean[0]) ** 2
        covariance = normalisation * cov_mat

        rel_noise_std = np.sqrt(np.diag(covariance)) / np.abs(correlator)

        return rel_noise_std

    def spectral_densities(self) -> np.ndarray:
        mapped_Es = self.x_of_E(self.ctx.energies)  # (n_E,)
        x_min = self.x_min()

        # T_E (n_b+1, n_E)
        # T_xmin (n_b + 1,)

        T_E = np.polynomial.chebyshev.chebvander(mapped_Es, self.n_b).T
        T_xmin = np.polynomial.chebyshev.chebvander(
            np.array([x_min]), self.n_b
        )[0, :]

        diff = T_E - T_xmin[:, None]  # (n_b + 1, n_E)
        densities = self.coeffs @ diff  # (n_E,)

        # Zero out energies below the threshold
        densities[mapped_Es <= x_min] = 0.0

        return densities


# ── HDF5 generation ───────────────────────────────────────────────────────────


def generate_all_training_sets(
    energies: np.ndarray,
    sigmas: list[float],
    n_T: int,
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

    log.info(
        "Generating training sets: N_b=%s, N_rho levels=%s, sigmas=%s",
        n_b_values,
        n_rho_values,
        sigmas,
    )
    log.info("Output directory: %s", os.path.abspath(output_dir))

    total_files = len(n_b_values)

    ctx = TrainingSetContext(energies, sigmas, n_ch_nodes, n_T)

    for file_idx, n_b in enumerate(n_b_values, 1):

        fname = os.path.join(output_dir, f"ts_nb{n_b}.h5")

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

            ts = TrainingSet(ctx, n_b, seed)

            all_correlators[rho_idx] = ts.correlator()
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
    muon_mass = 0.10566  # GeV
    e_min, e_max = muon_mass, 24 * muon_mass
    e_range = np.linspace(e_min, e_max, 47)

    sigma = [0.44, 0.63]  # GeV

    n_rhos = [50_000, 100_000, 200_000, 400_000, 800_000]
    n_bs = [16, 32, 128, 512]

    # lattice data from the 48 ensemble for the 2pi.g5.001 operator
    # we remove the irrelevant t=0 time-slice
    two_pt_data = np.load("noise_injection_data/2pt_data_64_folded.npy")[:, 1:]
    n_T = len(two_pt_data[0])

    ch_nodes = 10_000

    output_dir = "/home/joe/work/Inversion_problem/Training_sets/64_data"

    generate_all_training_sets(
        energies=e_range,
        sigmas=sigma,
        n_T=n_T,
        n_b_values=n_bs,
        n_rho_values=n_rhos,
        n_ch_nodes=ch_nodes,
        output_dir=output_dir,
    )
