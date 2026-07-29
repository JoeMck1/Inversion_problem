"""
error_analysis.py

Implementation of the statistical + systematic error-estimation
procedure for ML-reconstructed smeared spectral densities, following
Section V.A ("Procedure to estimate the total error").

Pipeline
--------
1.  For an ensemble of Nr replica machines trained at a fixed
    architecture/training-set vector N = (Nn, Nb, Nrho), feed every
    bootstrap sample of the lattice correlator, c = 1...Nc, through
    every replica r = 1..Nr, to obtain

        rho_pred_sigma(E, N, c, r).

2.  At fixed N and r, take the mean and error over c

        rho_pred_sigma(E, N, r),   Delta^latt_sigma(E, N, r)

    and then average both over the Nr replicas.

3.  The std of rho_pred_sigma(E, N, r) over r gives the network error
    Delta^net_sigma(E, N).

4.  Delta^stat_sigma(E, N) = sqrt(Delta^latt^2 + Delta^net^2)

5.  The quoted central value / statistical error of the final result
    are simply the N = N_max entries.

6.  Systematic errors are obtained by comparing the N_max result
    against three fixed reference vectors N_ref_rho, N_ref_b, N_ref_n.
    Each reference vector is chosen to maximise the respective
    systematic error. For each X in {rho, b, n} a "pull" is formed
    and used to weight the raw difference by the Gaussian
    probability that it is not a statistical fluctuation.

7.  All four contributions (stat, and the three systematics) are added in
    quadrature to give the total error.

Everything below is written to broadcast over arbitrary leading batch axes
(e.g. sigma, E, ...), so you can feed in whole arrays at once rather than
looping in Python.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erf

N_VECTORS = {
    "Nmax": {"Nn": "arcL", "Nb": 512, "Nrho": 800_000},
    "Nref_rho": {"Nn": "arcL", "Nb": 512, "Nrho": 50_000},
    "Nref_b": {"Nn": "arcL", "Nb": 16, "Nrho": 800_000},
    "Nref_n": {"Nn": "arcS", "Nb": 512, "Nrho": 800_000},
}

SYSTEMATIC_REF_LABEL = {"rho": "Nref_rho", "b": "Nref_b", "n": "Nref_n"}


@dataclass
class StatResult:
    """Result for rho_pred_sigma(E, N), delta_sigma(E, N)s."""

    central: np.ndarray  # rho_pred_sigma(E, N)
    delta_latt: np.ndarray  # Delta^latt_sigma(E, N)
    delta_net: np.ndarray  # Delta^net_sigma(E, N)
    delta_stat: np.ndarray  # Delta^stat_sigma(E, N)


def compute_stat_result(rho_pred: np.ndarray) -> StatResult:
    """
    Computes the StatResult instance attributes for a single, fixed N.

    Parameters
    ----------
    rho_pred : ndarray, shape (..., Nc, Nr)
        rho_pred_sigma(E, N, c, r) for a fixed N. Leading axes (...)
        can carry sigma and/or E (or anything else you want broadcast
        over); the last two axes must be (bootstrap sample c, replica
        machine r), in that order.

    Returns
    -------
    StatResult, each field of shape (...,) -- the trailing (Nc, Nr)
    axes are reduced away.
    """
    rho_pred = np.asarray(rho_pred)
    if rho_pred.ndim < 2:
        raise ValueError("rho_pred needs at least a trailing (Nc, Nr) shape")

    # Bootstrap/jackknife mean & error at fixed r (reduce over c, axis -2).
    central_r = rho_pred.mean(axis=-2)  # (..., Nr)
    delta_latt_r = rho_pred.std(axis=-2, ddof=1)  # (..., Nr)

    # Average both over the Nr replica machines -> Eq. (51).
    central = central_r.mean(axis=-1)
    delta_latt = delta_latt_r.mean(axis=-1)

    # Spread of the per-replica central values over r -> network error.
    delta_net = central_r.std(axis=-1, ddof=1)

    # Sum in quadrature -> Eq. (52).
    delta_stat = np.sqrt(delta_latt**2 + delta_net**2)

    return StatResult(
        central=central,
        delta_latt=delta_latt,
        delta_net=delta_net,
        delta_stat=delta_stat,
    )


def pull_variable(
    nmax_result: StatResult, ref_result: StatResult
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the pull variable.

    Returns
    -------
    (abs_diff, pull) : both ndarray, same shape as the StatResult fields.
    """
    abs_diff = np.abs(nmax_result.central - ref_result.central)
    denom = np.sqrt(nmax_result.delta_stat**2 + ref_result.delta_stat**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = np.where(denom > 0, abs_diff / denom, np.inf)
    return abs_diff, pull


def systematic_error(
    nmax_result: StatResult, ref_result: StatResult
) -> tuple[np.ndarray, np.ndarray]:
    """
    Delta^X_sigma(E), the Gaussian-weighted difference between the
    N_max result and the result at a fixed reference vector N_ref_X.

    Returns
    -------
    (delta_X, pull)
    """
    abs_diff, pull = pull_variable(nmax_result, ref_result)
    weight = erf(np.abs(pull) / np.sqrt(2))
    return abs_diff * weight, pull


def total_error(
    delta_stat: np.ndarray,
    delta_rho: np.ndarray,
    delta_b: np.ndarray,
    delta_n: np.ndarray,
) -> np.ndarray:
    return np.sqrt(delta_stat**2 + delta_rho**2 + delta_b**2 + delta_n**2)


def run_error_analysis(raw_predictions: dict[str, np.ndarray]) -> dict:
    """
    Parameters
    ----------
    raw_predictions : dict
        Must contain the four keys "Nmax", "Nref_rho", "Nref_b",
        "Nref_n" (see N_VECTORS above), each mapping to an ndarray of
        shape (..., Nc, Nr): every prediction obtained by feeding all
        Nc bootstrap samples through all Nr replica machines of the
        ensemble trained at that particular N. Leading axes (...) are
        arbitrary batch axes (sigma, E, ...) and must match across all
        four arrays.

    Returns
    -------
    dict with:
        'stat_results' : {label: StatResult} for all four N vectors
        'pulls'        : {'rho'/'b'/'n': pull array}
        'delta_sys'    : {'rho'/'b'/'n': Delta^X array}
        'central'      : rho_pred_sigma(E)     (= Nmax central value)
        'delta_stat'   : Delta^stat_sigma(E)   (= Nmax delta_stat)
        'delta_tot'    : Delta^tot_sigma(E)
    """
    required = set(N_VECTORS)
    missing = required - raw_predictions.keys()
    if missing:
        raise ValueError(
            f"raw_predictions is missing required key(s): {missing}"
        )

    stat_results = {
        label: compute_stat_result(arr)
        for label, arr in raw_predictions.items()
    }

    nmax_result = stat_results["Nmax"]

    pulls: dict[str, np.ndarray] = {}
    delta_sys: dict[str, np.ndarray] = {}
    for X, ref_label in SYSTEMATIC_REF_LABEL.items():
        d_sys, pull = systematic_error(nmax_result, stat_results[ref_label])
        pulls[X] = pull
        delta_sys[X] = d_sys

    delta_tot = total_error(
        nmax_result.delta_stat,
        delta_sys["rho"],
        delta_sys["b"],
        delta_sys["n"],
    )

    return {
        "stat_results": stat_results,
        "pulls": pulls,
        "delta_sys": delta_sys,
        "central": nmax_result.central,
        "delta_stat": nmax_result.delta_stat,
        "delta_tot": delta_tot,
    }


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n_E = 6
    Nc = 800  # bootstrap samples
    Nr = 20  # replica machines

    # "True" spectral density values at Nmax (arbitrary synthetic numbers).
    true_rho = np.linspace(0.2, 1.3, n_E)

    def make_predictions(true_vals, latt_noise, net_noise, bias=0.0):
        """
        Build a synthetic rho_pred(E, c, r) array:
          - per-replica offsets drawn with std = net_noise (network error)
          - per-(c,r) bootstrap noise with std = latt_noise (lattice error)
          - a constant bias, to simulate imperfect convergence at a
            reference N vector.
        """
        replica_offsets = rng.normal(0.0, net_noise, size=(Nr,))
        base = true_vals[:, None, None] + bias  # (n_E, 1, 1)
        base = base + replica_offsets[None, None, :]  # (n_E, 1, Nr)
        noise = rng.normal(0.0, latt_noise, size=(n_E, Nc, Nr))
        return base + noise

    raw_predictions = {
        # Nmax: best converged, smallest biases/noise.
        "Nmax": make_predictions(
            true_rho, latt_noise=0.01, net_noise=0.003, bias=0.0
        ),
        # Reference vectors: deliberately less converged (larger bias)
        # to mimic real Nb/Nn/Nrho truncation effects.
        "Nref_rho": make_predictions(
            true_rho, latt_noise=0.01, net_noise=0.004, bias=0.02
        ),
        "Nref_b": make_predictions(
            true_rho, latt_noise=0.01, net_noise=0.004, bias=-0.015
        ),
        "Nref_n": make_predictions(
            true_rho, latt_noise=0.01, net_noise=0.006, bias=0.03
        ),
    }

    results = run_error_analysis(raw_predictions)

    header = f"{'E_idx':>5} {'central':>10} {'d_stat':>10} {'d_rho':>10} {'d_b':>10} {'d_n':>10} {'d_tot':>10}"
    print(header)
    print("-" * len(header))
    for i in range(n_E):
        print(
            f"{i:>5} "
            f"{results['central'][i]:>10.5f} "
            f"{results['delta_stat'][i]:>10.5f} "
            f"{results['delta_sys']['rho'][i]:>10.5f} "
            f"{results['delta_sys']['b'][i]:>10.5f} "
            f"{results['delta_sys']['n'][i]:>10.5f} "
            f"{results['delta_tot'][i]:>10.5f}"
        )

    print("\nPull variables (P^X):")
    for X in ("rho", "b", "n"):
        print(f"  {X}: {np.round(results['pulls'][X], 3)}")
