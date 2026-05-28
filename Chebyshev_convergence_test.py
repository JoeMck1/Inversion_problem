"""
Convergence test for Gauss-Chebyshev quadrature applied to the
overlap integrals I_n(E_j) and J_n(t_tau) from Buzzicotti et al. (2024).

Tests convergence of:
  I_n(E_j) = integral_{E0}^{Emax} K_sigma(E_j, omega) B_n(omega) domega
  J_n(t_tau) = integral_{E0}^{Emax} b_T(t_tau, omega) B_n(omega) domega

as a function of the number of Gauss-Chebyshev quadrature nodes M.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_chebyt

# ── Physical constants ────────────────────────────────────────────────────────
M_MU = 0.10566  # muon mass in GeV
E_MIN = M_MU  # output grid minimum
E_MAX = 24 * M_MU  # output grid maximum (~2.536 GeV)
E0 = 0.5  # representative threshold in GeV
T_LAT = 64  # temporal extent in lattice units
SIGMAS = [0.44, 0.63]  # smearing widths in GeV


# ── Exponential map and its inverse ──────────────────────────────────────────
def x_of_E(E):
    return 1.0 - 2.0 * np.exp(-E)


def E_of_x(x):
    return -np.log((1.0 - x) / 2.0)


# ── Chebyshev basis functions B_n(E) ─────────────────────────────────────────
x_min = x_of_E(E0)
x_max = x_of_E(E_MAX)


def B_n(n, E):
    x = x_of_E(E)
    return eval_chebyt(n, x) - eval_chebyt(n, x_min)


# ── Smearing kernel K_sigma(E, omega) ────────────────────────────────────────
def K_sigma(E, omega, sigma):
    return np.exp(-0.5 * ((E - omega) / sigma) ** 2) / (
        sigma * np.sqrt(2 * np.pi)
    )


# ── Lattice correlator kernel b_T(t, omega) ──────────────────────────────────
def b_T(t, omega, T=T_LAT):
    return (omega**2 / (12 * np.pi**2)) * (
        np.exp(-t * omega) + np.exp(-(T - t) * omega)
    )


# ── Gauss-Chebyshev quadrature on [E0, E_MAX] ────────────────────────────────
def gauss_chebyshev_nodes_weights(M):
    """
    Returns M quadrature nodes in E-space and the associated weights,
    mapping [-1,1] -> [x_min, x_max] -> E.

    Gauss-Chebyshev: integral_{-1}^{1} f(u)/sqrt(1-u^2) du ~ (pi/M) sum_i f(u_i)
    We absorb sqrt(1-u_i^2) and the Jacobian dE/du into the integrand.
    """
    i = np.arange(1, M + 1)
    u_nodes = np.cos(np.pi * (2 * i - 1) / (2 * M))  # nodes in [-1,1]
    x_nodes = x_min + (u_nodes + 1) / 2 * (x_max - x_min)
    # nodes in [x_min, x_max]
    E_nodes = E_of_x(x_nodes)  # nodes in [E0, E_MAX]
    jacobian = (x_max - x_min) / (2 * (1 - x_nodes))  # dx/du * dE/dx = dE/du
    sqrt_factor = np.sqrt(1 - u_nodes**2)  # absorb Chebyshev weight
    weights = (np.pi / M) * jacobian * sqrt_factor  # full quadrature weights
    return E_nodes, weights


# ── Compute overlap integrals ─────────────────────────────────────────────────
def compute_I_n(n, E_j, sigma, M):
    """I_n(E_j) for smearing kernel."""
    E_nodes, weights = gauss_chebyshev_nodes_weights(M)
    integrand = K_sigma(E_j, E_nodes, sigma) * B_n(n, E_nodes)
    return np.sum(weights * integrand)


def compute_J_n(n, t_tau, M):
    """J_n(t_tau) for correlator kernel."""
    E_nodes, weights = gauss_chebyshev_nodes_weights(M)
    integrand = b_T(t_tau, E_nodes) * B_n(n, E_nodes)
    return np.sum(weights * integrand)


# ── Convergence test ──────────────────────────────────────────────────────────
M_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
M_ref = 100000  # reference value treated as "exact"

# Representative test cases
test_ns = [1, 32, 128, 512]  # basis function indices
test_E_js = [0.11, 1.25, 2.53]  # output energies in GeV
test_t_taus = [1, 16, 32, 64]  # time slices (small, mid, large t)

# ── Run convergence for I_n ───────────────────────────────────────────────────
results_I = {}  # key: (sigma, n, E_j)
for sigma in SIGMAS:
    for n in test_ns:
        for E_j in test_E_js:
            ref = compute_I_n(n, E_j, sigma, M_ref)
            vals = [compute_I_n(n, E_j, sigma, M) for M in M_values]
            rel_err = [abs(v - ref) / (abs(ref) + 1e-300) for v in vals]
            results_I[(sigma, n, E_j)] = rel_err

# ── Run convergence for J_n ───────────────────────────────────────────────────
results_J = {}  # key: (n, t_tau)
for n in test_ns:
    for t_tau in test_t_taus:
        ref = compute_J_n(n, t_tau, M_ref)
        vals = [compute_J_n(n, t_tau, M) for M in M_values]
        rel_err = [abs(v - ref) / (abs(ref) + 1e-300) for v in vals]
        results_J[(n, t_tau)] = rel_err

# ── Plotting ──────────────────────────────────────────────────────────────────
COLORS = plt.cm.viridis(np.linspace(0.15, 0.85, len(test_ns)))
LINE_STYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "x"]
THRESHOLD = 1e-6

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(
    2,
    2,
    figure=fig,
    hspace=0.38,
    wspace=0.32,
    left=0.07,
    right=0.97,
    top=0.91,
    bottom=0.08,
)

ax_titles = [
    rf"$I_n(E_j)$: $\sigma = {SIGMAS[0]}$ GeV",
    rf"$I_n(E_j)$: $\sigma = {SIGMAS[1]}$ GeV",
    rf"$J_n(t_\tau)$: varying $t_\tau$, $n={test_ns[0]}$",
    r"$J_n(t_\tau)$: varying $n$, $t_\tau=32$",
]
axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("white")
    ax.set_title(title, color="black", fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color="#444444", fontsize=9)
    ax.set_ylabel(ylabel, color="#444444", fontsize=9)
    ax.tick_params(colors="#444444", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#bbbbbb")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(M_values)
    ax.set_xticklabels([str(m) for m in M_values], fontsize=7)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.5)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axhline(
        THRESHOLD,
        color="#cc3333",
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
        label=r"$10^{-6}$ threshold",
    )


# Panel 0 & 1: I_n convergence for each sigma
for panel_idx, sigma in enumerate(SIGMAS):
    ax = axes[panel_idx]
    style_ax(
        ax,
        ax_titles[panel_idx],
        "M (quadrature nodes)",
        f"Relative error vs M={M_ref}",
    )
    for ni, n in enumerate(test_ns):
        for ei, E_j in enumerate(test_E_js):
            key = (sigma, n, E_j)
            ax.plot(
                M_values,
                results_I[key],
                color=COLORS[ni],
                linestyle=LINE_STYLES[ei],
                marker=MARKERS[ei],
                markersize=4,
                linewidth=1.2,
                label=rf"$n={n}$, $E_j={E_j}$ GeV",
            )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        fontsize=6.5,
        facecolor="white",
        edgecolor="#bbbbbb",
        labelcolor="black",
        ncol=2,
        loc="lower left",
    )

# Panel 2: J_n convergence varying t_tau at fixed n=0
ax = axes[2]
style_ax(
    ax, ax_titles[2], "M (quadrature nodes)", f"Relative error vs M={M_ref}"
)
t_colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(test_t_taus)))
for ti, t_tau in enumerate(test_t_taus):
    key = (test_ns[0], t_tau)
    ax.plot(
        M_values,
        results_J[key],
        color=t_colors[ti],
        linestyle=LINE_STYLES[ti],
        marker=MARKERS[ti],
        markersize=4,
        linewidth=1.2,
        label=rf"$t_\tau={t_tau}$",
    )
ax.legend(
    fontsize=7.5,
    facecolor="white",
    edgecolor="#bbbbbb",
    labelcolor="black",
    loc="lower left",
)

# Panel 3: J_n convergence varying n at fixed t_tau=32
ax = axes[3]
style_ax(
    ax, ax_titles[3], "M (quadrature nodes)", f"Relative error vs M={M_ref}"
)
for ni, n in enumerate(test_ns):
    key = (n, 32)
    ax.plot(
        M_values,
        results_J[key],
        color=COLORS[ni],
        linestyle=LINE_STYLES[ni],
        marker=MARKERS[ni],
        markersize=4,
        linewidth=1.2,
        label=rf"$n={n}$",
    )
ax.legend(
    fontsize=7.5,
    facecolor="white",
    edgecolor="#bbbbbb",
    labelcolor="black",
    loc="lower left",
)

fig.suptitle(
    "Gauss–Chebyshev Quadrature Convergence\n"
    r"$I_n(E_j) = \int K_\sigma B_n\,d\omega$ and $J_n(t_\tau) = \int b_T B_n\,d\omega$",
    color="black",
    fontsize=13,
    y=0.97,
)

plt.savefig(
    "/home/joe/work/Inversion_problem/Plots/convergence_test_n1.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved convergence_test.png")

# ── Print summary table ───────────────────────────────────────────────────────
threshold = 1e-6
print(f"\nSmallest M achieving relative error < {threshold:.0e}:\n")
print(f"{'Integral':<35} {'Min M':>8}")
print("-" * 45)
for sigma in SIGMAS:
    for n in test_ns:
        for E_j in test_E_js:
            errs = results_I[(sigma, n, E_j)]
            converged = [
                M_values[i] for i, e in enumerate(errs) if e < threshold
            ]
            M_conv = converged[0] if converged else ">2048"
            print(
                f"I_n: sigma={sigma}, n={n:>2}, E_j={E_j:.1f} GeV   {str(M_conv):>8}"
            )
print()
for n in test_ns:
    for t_tau in test_t_taus:
        errs = results_J[(n, t_tau)]
        converged = [M_values[i] for i, e in enumerate(errs) if e < threshold]
        M_conv = converged[0] if converged else ">2048"
        print(
            f"J_n: n={n:>2}, t_tau={t_tau:>2}                    {str(M_conv):>8}"
        )
