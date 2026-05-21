This repo contains my work on applying machine learning approaches to
the inversion problem as it appears in LQCD. Specifically, I am
recreating the approach pioneered in "Teaching to extract spectral
densities from lattice correlators to a broad audience of 
learning-machines" (Buzzicotti, De Santis, Tantalo, 2024). We then hope
to improve upon their method by applying more sophisticated machine
learning techniques.

The files in this repo are listed below, along with a short summary:

- Chebyshev_convergence_test.py contains a script produces plots which
guide/jusify the choice of the number of Chebyshev nodes used in the
integrations needed to compute the randomly generated correlators C(t)
and their accompanying smeared spectral densities.

- nn_architectures.py contains classes and functions needed to define
and train the neural networks which will be used to obtain smeared
spectral densities from real lattice generated correlators C(t).

- training_sets_generation.py contains a class which produces the
training sets needed to train the neural networks.