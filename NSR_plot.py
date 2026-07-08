import matplotlib.pyplot as plt
import numpy as np

two_pt_data = np.load("noise_injection_data/2pt_data_64_folded.npy")
signal = np.mean(two_pt_data, axis=0)

noise = np.std(two_pt_data, axis=0, ddof=1)

snr = signal / noise
nsr = 1 / snr
time = np.arange(len(two_pt_data[0]))

plt.scatter(time, 100 * nsr, s=10)
plt.xlabel(r"$\frac{t}{a}$")
plt.ylabel(r"$\frac{\Delta C_\text{latt}(t)}{C_\text{latt}(t)} \quad (\%)$")
plt.savefig(
    "Plots/NSR_64_folded.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()
