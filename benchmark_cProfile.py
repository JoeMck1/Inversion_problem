# benchmark.py
import cProfile

import numpy as np

from TrainingSets_Full_Chebyshev import generate_all_training_sets

muon_mass = 0.10566
e_range = np.linspace(muon_mass, 24 * muon_mass, 47)

cProfile.run(
    """
generate_all_training_sets(
    energies=e_range,
    sigmas=[0.44, 0.63],
    n_T=48,
    n_b_values=[512],        # worst case only
    n_rho_values=[10000],     # small run
    n_ch_nodes=10000,
    output_dir="benchmark_out",
    overwrite=True,
)
""",
    "profile_output.prof",
)
