# benchmark.py
import cProfile

import numpy as np

from TrainingSets_Quadrature import generate_all_training_sets

two_pt_data = np.load("2pt_data_64.npy")
muon_mass = 0.10566
e_range = np.linspace(muon_mass, 24 * muon_mass, 47)

cProfile.run(
    """
generate_all_training_sets(
    lattice_data=two_pt_data,
    energies=e_range,
    sigmas=[0.44, 0.63],
    n_b_values=[512],        # worst case only
    n_rho_values=[1000],     # small run
    n_ch_nodes=1024,
    output_dir="benchmark_out",
    overwrite=True,
)
""",
    "profile_output.prof",
)
