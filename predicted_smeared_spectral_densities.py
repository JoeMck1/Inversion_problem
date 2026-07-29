import numpy as np
import torch
from torch import nn

from nn_architectures import CorrelatorStandardiser, arcS, load_ensemble


def predict_ensemble(
    ensemble: list[nn.Module],
    standardiser: CorrelatorStandardiser,
    correlators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Predict using an ensemble of trained networks.

    Parameters
    ----------
    correlators : shape (N, n_t)

    Returns
    -------
    mean_prediction : shape (N, output_dim)
    std_prediction  : shape (N, output_dim)
    """

    correlators = standardiser.transform(correlators)

    with torch.no_grad():
        predictions = torch.stack([model(correlators) for model in ensemble])

    return predictions


if __name__ == "__main__":
    correlator_data = torch.tensor(
        np.load("noise_injection_data/2pt_data_64_folded.npy")[:, 1:],
        dtype=torch.float32,
    )

    load_dir = "trained_models/arcS_nb16_nrho50000_sigma0.44"
    ensemble, standardiser, metadata = load_ensemble(
        architecture_fn=arcS, load_dir=load_dir, device="cpu"
    )

    standardised_correlator_data = standardiser.transform(correlator_data)

    pred_rho = predict_ensemble(
        ensemble=ensemble,
        standardiser=standardiser,
        correlators=standardised_correlator_data,
    )
    save_dir = f"predicted_rhos/{metadata['architecture']}_nb{metadata['n_b']}_nrho_{metadata['n_rho']}_sigma_{metadata['sigma']}_replicas_{metadata['n_replicas']}"

    np.save(save_dir, pred_rho)
