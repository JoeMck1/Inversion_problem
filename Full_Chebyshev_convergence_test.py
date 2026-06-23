import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct

# ── Physical constants ────────────────────────────────────────────────────────
M_MU = 0.10566
E_MAX = 24 * M_MU  # ~2.536 GeV
E0 = 0.5  # representative threshold in GeV
T_LAT = 96
SIGMAS = [0.44, 0.63]


# ── Maps ──────────────────────────────────────────────────────────────────────
def x_of_E(E):
    return 1.0 - 2.0 * np.exp(-np.asarray(E))


def E_of_x(x):
    return -np.log((1.0 - x) / 2.0)


x_min = x_of_E(E0)


# ── Kernels ───────────────────────────────────────────────────────────────────
def K_sigma(E_j, omega, sigma):
    return np.exp(-0.5 * ((E_j - omega) / sigma) ** 2) / (
        sigma * np.sqrt(2 * np.pi)
    )


def b_T(t, omega, T=T_LAT):
    return (omega**2 / (12 * np.pi**2)) * (
        np.exp(-t * omega) + np.exp(-(T - t) * omega)
    )


# ── Nodes and weight ──────────────────────────────────────────────────────────
def chebyshev_nodes(N):
    """x_k = cos(pi*(k + 1/2)/N),  k = 0, ..., N-1"""
    k = np.arange(N)
    return np.cos(np.pi * (k + 0.5) / N)


def g_weights(x_nodes):
    """Theta(E(x_k) - E0) * sqrt(1 - x_k^2) / (1 - x_k)"""
    active = E_of_x(x_nodes) > E0
    w = np.sqrt(1.0 - x_nodes**2) / (1.0 - x_nodes)
    return np.where(active, w, 0.0)


# ── Core: compute all A_n via DCT-II ─────────────────────────────────────────
def compute_A_n(n_b, kernel_vals, N):
    """
    Returns A_n for n = 0, ..., n_b, shape (n_b+1,).

    Uses the identity T_n(x_k) = cos(pi*n*(k+1/2)/N) so that
        F_n = sum_k T_n(x_k) * g(x_k) = dct(g, type=2)[n] / 2

    kernel_vals : (N,) array evaluated at E(x_k)
    """
    x_nodes = chebyshev_nodes(N)
    gk = g_weights(x_nodes) * kernel_vals  # (N,)

    # DCT-II: scipy returns 2*F_n for n=0,...,N-1, so divide by 2.
    # Zero-pad to (n_b+1,) in case N < n_b+1.
    raw = dct(gk, type=2, norm=None) / 2  # (N,)
    F = np.zeros(n_b + 1)
    F[: min(N, n_b + 1)] = raw[: min(N, n_b + 1)]  # (n_b+1,)
    F0 = F[0]

    # T_n(x_min) via cosine identity on the scalar x_min:
    # T_n(x_min) = cos(n * arccos(x_min))
    n_vals = np.arange(n_b + 1)
    T_xmin = np.cos(n_vals * np.arccos(x_min))  # (n_b+1,)

    return (np.pi / N) * (F - T_xmin * F0)  # (n_b+1,)


# ── Full result: sum_n c_n * A_n ─────────────────────────────────────────────
def compute_full(coeffs, kernel_vals, N):
    A = compute_A_n(len(coeffs) - 1, kernel_vals, N)
    return coeffs @ A


# ── Kernel evaluation at nodes ────────────────────────────────────────────────
def smearing_kernel_at_nodes(E_j, sigma, N):
    return K_sigma(E_j, E_of_x(chebyshev_nodes(N)), sigma)


def correlator_kernel_at_nodes(t, N):
    return b_T(t, E_of_x(chebyshev_nodes(N)))


# ── Test parameters ───────────────────────────────────────────────────────────
# Extend N_values into the large-N regime where C(t) converges
N_values = [2**n for n in np.arange(6, 21)]
N_ref = 2**23
THRESHOLD = 1e-6

test_ns = [1, 32, 128, 512]
test_E_js = [0.11, 1.25, 2.53]
test_t_taus = [1, 24, 48, 96]

# Fixed n_b and random coefficients for full-result tests
N_B = 512
rng = np.random.default_rng(42)
eps = 1e-7
coeffs = np.empty(N_B + 1)
rns = rng.uniform(-1, 1, N_B + 1)
coeffs[0] = rns[0]
coeffs[1:] = rns[1:] / np.arange(1, N_B + 1) ** (1 + eps)

print("Computing reference values ...", end="", flush=True)

# ── 1. A_n^K ─────────────────────────────────────────────────────────────────
results_AK = {}
n_b_max = max(test_ns)
for sigma in SIGMAS:
    for n in test_ns:
        for E_j in test_E_js:
            ref = compute_A_n(
                n_b_max, smearing_kernel_at_nodes(E_j, sigma, N_ref), N_ref
            )[n]
            errs = [
                abs(
                    compute_A_n(
                        n_b_max, smearing_kernel_at_nodes(E_j, sigma, N), N
                    )[n]
                    - ref
                )
                / (abs(ref) + 1e-300)
                for N in N_values
            ]
            results_AK[(sigma, n, E_j)] = errs

# ── 2. A_n^b ─────────────────────────────────────────────────────────────────
results_Ab = {}
for n in test_ns:
    for t in test_t_taus:
        ref = compute_A_n(
            n_b_max, correlator_kernel_at_nodes(t, N_ref), N_ref
        )[n]
        errs = [
            abs(
                compute_A_n(n_b_max, correlator_kernel_at_nodes(t, N), N)[n]
                - ref
            )
            / (abs(ref) + 1e-300)
            for N in N_values
        ]
        results_Ab[(n, t)] = errs

# ── 3. Full rho_hat ───────────────────────────────────────────────────────────
results_rho = {}
for sigma in SIGMAS:
    for E_j in test_E_js:
        ref = compute_full(
            coeffs, smearing_kernel_at_nodes(E_j, sigma, N_ref), N_ref
        )
        errs = [
            abs(
                compute_full(
                    coeffs, smearing_kernel_at_nodes(E_j, sigma, N), N
                )
                - ref
            )
            / (abs(ref) + 1e-300)
            for N in N_values
        ]
        results_rho[(sigma, E_j)] = errs

# ── 4. Full C(t) ──────────────────────────────────────────────────────────────
results_Ct = {}
for t in test_t_taus:
    ref = compute_full(coeffs, correlator_kernel_at_nodes(t, N_ref), N_ref)
    errs = [
        abs(compute_full(coeffs, correlator_kernel_at_nodes(t, N), N) - ref)
        / (abs(ref) + 1e-300)
        for N in N_values
    ]
    results_Ct[t] = errs

print(" done.")

# ── Plotting ──────────────────────────────────────────────────────────────────
COLORS_N = plt.cm.viridis(np.linspace(0.15, 0.85, len(test_ns)))
COLORS_T = plt.cm.plasma(np.linspace(0.2, 0.85, len(test_t_taus)))
COLORS_S = plt.cm.cool(np.linspace(0.2, 0.8, len(SIGMAS)))
LS = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "x"]


def style_ax(ax, title):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("N (Chebyshev nodes)", fontsize=9)
    ax.set_ylabel(f"Relative error vs N={N_ref}", fontsize=9)
    ax.tick_params(labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#bbbbbb")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(N_values)
    ax.set_xticklabels([str(n) for n in N_values], fontsize=6, rotation=30)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.5)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axhline(
        THRESHOLD,
        color="#cc3333",
        linewidth=0.9,
        linestyle="--",
        alpha=0.7,
        label=r"$10^{-6}$ threshold",
    )


# ── Figure 1: A_n convergence ─────────────────────────────────────────────────
fig1 = plt.figure(figsize=(16, 12))
fig1.patch.set_facecolor("white")
gs1 = gridspec.GridSpec(
    2,
    2,
    figure=fig1,
    hspace=0.45,
    wspace=0.30,
    left=0.07,
    right=0.97,
    top=0.90,
    bottom=0.08,
)
axs1 = [fig1.add_subplot(gs1[i // 2, i % 2]) for i in range(4)]

for pi_, sigma in enumerate(SIGMAS):
    ax = axs1[pi_]
    style_ax(ax, rf"$A_n^K(E_j)$:  $\sigma = {sigma}$ GeV")
    for ni, n in enumerate(test_ns):
        for ei, E_j in enumerate(test_E_js):
            ax.plot(
                N_values,
                results_AK[(sigma, n, E_j)],
                color=COLORS_N[ni],
                linestyle=LS[ei],
                marker=MARKERS[ei],
                markersize=4,
                linewidth=1.2,
                label=rf"$n={n}$, $E_j={E_j}$ GeV",
            )
    ax.legend(
        fontsize=6.5,
        facecolor="white",
        edgecolor="#bbbbbb",
        ncol=2,
        loc="upper right",
    )

ax = axs1[2]
style_ax(ax, rf"$A_n^b(t)$: varying $t$,  $n = {test_ns[0]}$")
for ti, t in enumerate(test_t_taus):
    ax.plot(
        N_values,
        results_Ab[(test_ns[0], t)],
        color=COLORS_T[ti],
        linestyle=LS[ti],
        marker=MARKERS[ti],
        markersize=4,
        linewidth=1.2,
        label=rf"$t={t}$",
    )
ax.legend(
    fontsize=7.5, facecolor="white", edgecolor="#bbbbbb", loc="upper right"
)

ax = axs1[3]
style_ax(ax, r"$A_n^b(t)$: varying $n$,  $t = 48$")
for ni, n in enumerate(test_ns):
    ax.plot(
        N_values,
        results_Ab[(n, 48)],
        color=COLORS_N[ni],
        linestyle=LS[ni],
        marker=MARKERS[ni],
        markersize=4,
        linewidth=1.2,
        label=rf"$n={n}$",
    )
ax.legend(
    fontsize=7.5, facecolor="white", edgecolor="#bbbbbb", loc="upper right"
)

fig1.suptitle(
    r"Convergence of $A_n = (\pi/N)(F_n - F_0 T_n(x_\mathrm{min}))$  "
    r"[DCT-II, $x_k = \cos(\pi(k+\frac{1}{2})/N)$]",
    fontsize=12,
    y=0.97,
)
fig1.savefig(
    "Plots/Full_Chebyshev_convergence/convergence_A_n_dct.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig1)
print("Saved Plots/Full_Chebyshev_convergence/convergence_A_n_dct.png")

# ── Figure 2: Full result convergence ─────────────────────────────────────────
fig2 = plt.figure(figsize=(14, 6))
fig2.patch.set_facecolor("white")
gs2 = gridspec.GridSpec(
    1,
    2,
    figure=fig2,
    wspace=0.30,
    left=0.07,
    right=0.97,
    top=0.83,
    bottom=0.15,
)
axs2 = [fig2.add_subplot(gs2[0, i]) for i in range(2)]

ax = axs2[0]
style_ax(ax, r"Full $\hat{\rho}_\sigma(E_j)$ = $\sum_n c_n A_n^K$")
for si, sigma in enumerate(SIGMAS):
    for ei, E_j in enumerate(test_E_js):
        ax.plot(
            N_values,
            results_rho[(sigma, E_j)],
            color=COLORS_S[si],
            linestyle=LS[ei],
            marker=MARKERS[ei],
            markersize=4,
            linewidth=1.2,
            label=rf"$\sigma={sigma}$, $E_j={E_j}$ GeV",
        )
ax.legend(
    fontsize=6.5,
    facecolor="white",
    edgecolor="#bbbbbb",
    ncol=2,
    loc="upper right",
)

ax = axs2[1]
style_ax(ax, r"Full $C(t)$ = $\sum_n c_n A_n^b$")
for ti, t in enumerate(test_t_taus):
    ax.plot(
        N_values,
        results_Ct[t],
        color=COLORS_T[ti],
        linestyle=LS[ti],
        marker=MARKERS[ti],
        markersize=4,
        linewidth=1.2,
        label=rf"$t={t}$",
    )
ax.legend(
    fontsize=7.5, facecolor="white", edgecolor="#bbbbbb", loc="upper right"
)

fig2.suptitle(
    r"Convergence of full $\hat{\rho}_\sigma(E)$ and $C(t)$,  $N_b=$"
    + str(N_B)
    + r"  [DCT-II, $x_k = \cos(\pi(k+\frac{1}{2})/N)$]",
    fontsize=12,
    y=0.98,
)
fig2.savefig(
    "Plots/Full_Chebyshev_convergence/convergence_full_dct.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig2)
print("Saved Plots/Full_Chebyshev_convergence/convergence_full_dct.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\nSmallest N achieving relative error < {THRESHOLD:.0e}:\n")
print(f"{'Quantity':<48} {'Min N':>8}")
print("-" * 58)

for sigma in SIGMAS:
    for n in test_ns:
        for E_j in test_E_js:
            errs = results_AK[(sigma, n, E_j)]
            hit = [N_values[i] for i, e in enumerate(errs) if e < THRESHOLD]
            print(
                f"A_n^K  sigma={sigma}  n={n:>3}  E_j={E_j:.2f} GeV"
                f"{'':>3}{str(hit[0]) if hit else f'>{N_values[-1]}':>8}"
            )
print()
for n in test_ns:
    for t in test_t_taus:
        errs = results_Ab[(n, t)]
        hit = [N_values[i] for i, e in enumerate(errs) if e < THRESHOLD]
        print(
            f"A_n^b  n={n:>3}  t={t:>2}"
            f"{'':>26}{str(hit[0]) if hit else f'>{N_values[-1]}':>8}"
        )
print()
for sigma in SIGMAS:
    for E_j in test_E_js:
        errs = results_rho[(sigma, E_j)]
        hit = [N_values[i] for i, e in enumerate(errs) if e < THRESHOLD]
        print(
            f"rho_hat  sigma={sigma}  E_j={E_j:.2f} GeV"
            f"{'':>12}{str(hit[0]) if hit else f'>{N_values[-1]}':>8}"
        )
print()
for t in test_t_taus:
    errs = results_Ct[t]
    hit = [N_values[i] for i, e in enumerate(errs) if e < THRESHOLD]
    print(
        f"C(t)  t={t:>2}"
        f"{'':>38}{str(hit[0]) if hit else f'>{N_values[-1]}':>8}"
    )
