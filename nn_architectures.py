"""
PyTorch implementation of the three neural network architectures
(arcS, arcM, arcL) from "Teaching to extract spectral densities from
lattice correlators to a broad audience of learning-machines"
(Buzzicotti, De Santis, Tantalo, 2024).

Each network solves a regression problem: given a 64-component
Euclidean correlator C(t) as input, predict a 47-component smeared
spectral density rho_hat(E) as output.

CHANGELOG (beta-NLL extension)
-------------------------------
The original paper trains with the Mean Absolute Error loss, which
gives a single point-estimate output rho_hat(E) per network. This
version adds an alternative loss, L_beta-NLL,
which instead trains the network to output both a predicted mean
mu_hat(E) and a predicted (per-energy) aleatoric variance sigma2_hat(E)

Architecture overview:

    Input (64,)
        |
    Conv1D block x3  <-- learns local/abstract features
        |
    Flatten
        |
    Fully connected x2  <-- shared trunk, produces h(X)
        |
        +---------------------------+
        |                           |
    [loss_type="mae"]        [loss_type="beta_nll"]
    single output head       two independent heads reading off the
    (256 -> 47, no           SAME h(X):
    activation)                mu_head:     256 -> 47 (no activation)
        |                      sigma2_head: 256 -> 47, then softplus
    Output (47,)                            to guarantee positivity
                                   |
                             Outputs (mu_hat(E), sigma2_hat(E)),
                                                  each (47,)
"""

import json
import os

import numpy as np
import torch
from torch import nn

# =============================================================================
# BUILDING BLOCKS
# =============================================================================


class SpectralNet(nn.Module):
    """
    Generic feed-forward convolutional network for spectral density extraction.

    All three architectures (arcS, arcM, arcL) share the same skeleton:
      - Three 1D convolutional layers with increasing numbers of feature maps
      - A shared fully-connected trunk (two dense layers of 256 neurons)
      - One or two output heads reading off the trunk's final feature
        vector h(X), depending on `probabilistic`
      - LeakyReLU activations throughout the trunk (never on the outputs)

    The only difference between arcS / arcM / arcL is the `base_maps` parameter,
    which scales the number of convolutional filters.

    Parameters
    ----------
    base_maps : int
        Number of feature maps in the first Conv1D layer.
        The subsequent layers use 2x and 4x this value.
        arcS -> 2, arcM -> 12, arcL -> 32.
    input_size : int
        Length of the input correlator vector. Default: 64.
    output_size : int
        Length of the output spectral density vector. Default: 47.
    leaky_slope : float
        Negative slope for LeakyReLU. The paper uses -0.2, but PyTorch's
        LeakyReLU convention uses a positive number for the slope magnitude,
        so we pass 0.2 here.
    probabilistic : bool
        If False (default), the network has a single output head and
        forward() returns one tensor of shape (batch, output_size) --
        exactly the original architecture, for use with the MAE loss.
        If True, the network has two independent heads -- mu_head and
        sigma2_head -- both fully connected to the same shared feature
        vector h(X), and forward() returns a (mu, sigma2) tuple, for use
        with beta_nll_loss().
    sigma2_eps : float
        Only used when probabilistic=True. A small positive floor added
        to the softplus output so sigma2_hat can never be exactly zero.
        Softplus(x) -> 0 as x -> -inf but can underflow to exactly 0.0 in
        floating point for very negative pre-activations (which happens
        routinely at the start of training, before the sigma2 head has
        learned anything sensible); an exact zero would make
        0.5*log(sigma2) in the loss diverge. Default 1e-6.
    """

    def __init__(
        self,
        base_maps: int,
        input_size: int = 64,
        output_size: int = 47,
        leaky_slope: float = 0.2,
        probabilistic: bool = False,
        sigma2_eps: float = 1e-6,
    ):
        super().__init__()

        self.probabilistic = probabilistic
        self.sigma2_eps = sigma2_eps

        # ------------------------------------------------------------------
        # CONVOLUTIONAL BLOCK
        # ------------------------------------------------------------------
        # In PyTorch, nn.Conv1d expects input of shape (batch, channels, length).
        # Our correlator starts as a 1-channel time series of length 64.
        #
        # Each convolutional layer uses:
        #   kernel_size = 3   (a filter that looks at 3 consecutive time steps)
        #   stride = 2        (the filter jumps 2 steps each time, halving the
        #                      sequence length: 64 -> 32 -> 16 -> 8)
        #   padding = 1       (adds one zero on each side so the halving is exact)
        #
        # The number of output channels (feature maps) doubles at each layer,
        # mirroring the LeNet-style design described in the paper.
        # ------------------------------------------------------------------

        maps1 = base_maps  # e.g., 2 for arcS
        maps2 = base_maps * 2  # e.g., 4 for arcS
        maps3 = base_maps * 4  # e.g., 8 for arcS

        self.conv_block = nn.Sequential(
            # Layer 1: (batch, 1, 64) -> (batch, maps1, 32)
            nn.Conv1d(
                in_channels=1,
                out_channels=maps1,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=leaky_slope),
            # Layer 2: (batch, maps1, 32) -> (batch, maps2, 16)
            nn.Conv1d(
                in_channels=maps1,
                out_channels=maps2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=leaky_slope),
            # Layer 3: (batch, maps2, 16) -> (batch, maps3, 8)
            nn.Conv1d(
                in_channels=maps2,
                out_channels=maps3,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=leaky_slope),
        )

        # After the three conv layers the signal has shape (batch, maps3, 8).
        # Flattening collapses the maps and time dimensions into one vector:
        #   flatten_size = maps3 * 8
        flatten_size = maps3 * 8  # e.g., 8*8=64 for arcS; 48*8=384 for arcM

        # ------------------------------------------------------------------
        # SHARED FULLY-CONNECTED TRUNK
        # ------------------------------------------------------------------
        # Two dense layers of 256 neurons each, both with LeakyReLU. This
        # produces the shared feature vector h(X) that whichever head(s)
        # follow will read off in full -- the trunk itself never depends
        # on `probabilistic`, only what comes after it does.
        # ------------------------------------------------------------------
        self.trunk_fc = nn.Sequential(
            nn.Flatten(),
            # Dense layer 1: flatten_size -> 256
            nn.Linear(flatten_size, 256),
            nn.LeakyReLU(negative_slope=leaky_slope),
            # Dense layer 2: 256 -> 256
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=leaky_slope),
        )

        # ------------------------------------------------------------------
        # OUTPUT HEAD(S)
        # ------------------------------------------------------------------
        # No activation on any output head: mu (or the single MAE output)
        # is left unconstrained since spectral densities can be negative.
        # sigma2's positivity constraint is applied in forward(), via
        # softplus, rather than here, since softplus is not layer-specific.
        # ------------------------------------------------------------------
        if probabilistic:
            self.mu_head = nn.Linear(256, output_size)
            self.sigma2_head = nn.Linear(256, output_size)
        else:
            self.output_head = nn.Linear(256, output_size)

        # ------------------------------------------------------------------
        # WEIGHT INITIALISATION
        # ------------------------------------------------------------------
        # The paper initialises each weight from a Gaussian with mean 0 and
        # variance 0.05 (i.e. std = sqrt(0.05) ~ 0.2236). This is applied
        # identically to every Linear/Conv1d submodule, including whichever
        # output head(s) were just created above.
        # PyTorch's default initialisation (Kaiming uniform) is different,
        # so we override it here. Biases are initialised to zero.
        # ------------------------------------------------------------------
        self._initialise_weights(std=0.2236)

    def _initialise_weights(self, std: float):
        """Gaussian weight initialisation to Linear and Conv1d layers.

        self.modules() is a method inherited from nn.Module. It returns
        an iterator over every submodule in the network e.g.
        Sequential, Conv1d, Linear, LeakyReLU, and self itself. This
        correctly reaches mu_head/sigma2_head (or output_head) whichever
        branch was constructed, since they are ordinary registered
        submodules regardless of whether they sit inside a Sequential.

        isinstance(..., (Linear, Conv1d)) returns true for the Linear
        and Conv1d submodules. We want this as these are the only
        layers with learnable weights.

        nn.init.normal_(module.weight, ...) initialises the weights by
        drawing the from a normal distribution. This replaces PyTorch's
        default Kaiming initialisation. module.weight is a tensor of
        containing the learnable weights for that layer.

        module.bias is a tensor containing the biases for the layer.
        Not every layer needs biases, so the last if statement checks
        this first. nn.init.zeros_ then initialises the biases to zero.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, 64)
            A batch of (pre-processed) correlator vectors.

        Returns
        -------
        If self.probabilistic is False:
            torch.Tensor of shape (batch_size, 47)
                Predicted smeared spectral density rho_hat(E).
        If self.probabilistic is True:
            (mu, sigma2) : tuple of two torch.Tensor, each (batch_size, 47)
                mu     -- predicted mean of rho_hat(E), unconstrained.
                sigma2 -- predicted aleatoric variance of rho_hat(E),
                          guaranteed positive via softplus (+ sigma2_eps).
        """
        # Conv1d requires a channel dimension: (batch, 64) -> (batch, 1, 64)
        x = x.unsqueeze(1)

        # Pass through the three convolutional layers
        x = self.conv_block(x)

        # Shared trunk: flatten + two FC layers -> h(X), shape (batch, 256)
        h = self.trunk_fc(x)

        if self.probabilistic:
            mu = self.mu_head(h)
            raw_sigma2 = self.sigma2_head(h)
            sigma2 = nn.functional.softplus(raw_sigma2) + self.sigma2_eps
            return mu, sigma2

        return self.output_head(h)


# =============================================================================
# THE THREE NAMED ARCHITECTURES
# =============================================================================


def arcS(
    input_size: int = 64, output_size: int = 47, probabilistic: bool = False
) -> SpectralNet:
    """
    arcS: the smallest architecture (~94,651 parameters when
    probabilistic=False; probabilistic=True adds one extra 256->47
    linear head, ~12k more parameters).
    Conv maps: 2 -> 4 -> 8.
    """
    return SpectralNet(
        base_maps=2,
        input_size=input_size,
        output_size=output_size,
        probabilistic=probabilistic,
    )


def arcM(
    input_size: int = 64, output_size: int = 47, probabilistic: bool = False
) -> SpectralNet:
    """
    arcM: the medium architecture (~180,871 parameters when
    probabilistic=False).
    Conv maps: 12 -> 24 -> 48.
    """
    return SpectralNet(
        base_maps=12,
        input_size=input_size,
        output_size=output_size,
        probabilistic=probabilistic,
    )


def arcL(
    input_size: int = 64, output_size: int = 47, probabilistic: bool = False
) -> SpectralNet:
    """
    arcL: the largest architecture (~371,311 parameters when
    probabilistic=False).
    Conv maps: 32 -> 64 -> 128.
    """
    return SpectralNet(
        base_maps=32,
        input_size=input_size,
        output_size=output_size,
        probabilistic=probabilistic,
    )


# =============================================================================
# PARAMETER COUNTING UTILITY
# =============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    model.parameters() returns an iterator over every learnable
    parameter tensor in the network (all weight matrices and bias
    vectors).

    p.requires_grad evaluates True by default for all learnable
    parameters, and False for any tensors which may have been
    explicitly forzen. The if statement checks this.

    p.numel() returns the number of scalars (i.e. elements) in a
    tensor.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# DATA PRE-PROCESSING
# =============================================================================


class CorrelatorStandardiser:
    """
    Implements the standardisation procedure described in Section IV C.

    For each time slice tau, the standardiser computes:
        mu(tau)    = mean of C(tau) over the training set
        gamma(tau) = std  of C(tau) over the training set

    Then transforms each correlator:
        C'(tau) = (C(tau) - mu(tau)) / gamma(tau)

    This ensures every component of the input has zero mean and unit variance,
    which stabilises training when the raw correlator decays exponentially.

    The mu and gamma vectors are computed from the NOISE-FREE correlators
    (before noise injection), as stated in the paper.
    """

    def __init__(self):
        self.mu: torch.Tensor | None = None
        self.gamma: torch.Tensor | None = None

    def fit(self, correlators: torch.Tensor):
        """
        Compute mu and gamma from a set of (clean) correlators.

        Parameters
        ----------
        correlators : torch.Tensor of shape (N_rho, 64)
            The clean (noise-free) correlator vectors in the training set.
        """
        # Mean and std along the sample axis (axis 0), keeping shape (64,)
        self.mu = correlators.mean(dim=0)
        self.gamma = correlators.std(dim=0)

        # Guard against zero std (shouldn't happen in practice)
        self.gamma = torch.clamp(self.gamma, min=1e-12)

    def transform(self, correlators: torch.Tensor) -> torch.Tensor:
        """
        Apply standardisation to a batch of correlators.

        Parameters
        ----------
        correlators : torch.Tensor of shape (batch, 64)

        Returns
        -------
        torch.Tensor of shape (batch, 64)
        """
        if self.mu is None or self.gamma is None:
            raise RuntimeError("Call fit() before transform().")
        return (correlators - self.mu) / self.gamma

    def fit_transform(self, correlators: torch.Tensor) -> torch.Tensor:
        """Convenience method: fit then transform."""
        self.fit(correlators)
        return self.transform(correlators)


# =============================================================================
# NOISE INJECTION
# =============================================================================


class NoiseInjector:
    """
    Draws noisy correlators from the multivariate Gaussian of Eq. (44):

        C_noisy ~ N( C, (C(a)/C_latt(a))^2 * Sigma_latt )

    where Sigma_latt is the (fixed) covariance matrix of the real lattice
    correlator, and the scalar prefactor (C(a)/C_latt(a))^2 differs for
    every sample because it depends on that sample's own C(a).

    Design note: since Sigma_latt is the SAME matrix for every sample and
    every replica -- only the scalar prefactor changes -- its decomposition
    is computed once in __init__ and reused for every draw. This means
    generating noise for 20 replicas costs 20 cheap sampling passes, not
    20 repeats of an O(n_T^3) decomposition.

    We decompose Sigma_latt via eigendecomposition rather than Cholesky.
    A sample covariance matrix estimated from N_c bootstrap/jackknife
    draws has rank at most N_c - 1, so unless N_c is comfortably larger
    than n_T it is only positive SEMI-definite: some eigenvalues are
    exactly (or, after floating-point roundoff, very slightly negative)
    zero. Cholesky requires strict positive-definiteness and raises
    LinAlgError otherwise. Eigendecomposition works for any symmetric
    matrix; we clip away the (numerically meaningless) negative
    eigenvalues, which removes floating-point noise around zero without
    inflating or otherwise altering the covariance structure the data
    actually supports. This mirrors what np.random.multivariate_normal
    does internally by default (it uses an SVD-based decomposition).
    The difference being SVD, instead of clipping negative eigenvalues
    to 0, clips them to |lambda|.

    Sampling identity used: if Sigma = Q Lambda Q^T (eigendecomposition,
    with Lambda's negative entries clipped to 0) and we set
        L = Q @ sqrt(Lambda_clipped),
    then L @ L.T = Sigma (up to the clipped directions), so
        x = mean + sqrt(scalar) * (L @ z),   z ~ N(0, I)
    has covariance scalar * Sigma, as required. For a batch of N samples
    this is implemented as a single matrix multiplication.
    """

    def __init__(
        self,
        cov_mat: np.ndarray,
        lattice_mean: np.ndarray,
        seed: int,
        eigenvalue_floor: float = 0.0,
    ):
        """
        Parameters
        ----------
        cov_mat : ndarray, shape (n_T, n_T)
            Covariance matrix Sigma_latt of the real lattice correlator,
            estimated from the bootstrap/jackknife samples.
        lattice_mean : ndarray, shape (n_T,)
            Mean lattice correlator C_latt(t). Only C_latt(a), the first
            time slice, is actually used (see Eq. 44).
        seed : int
            Seed for this injector's random number generator. Use a
            DIFFERENT seed per replica so that each replica's noise draw
            is independent, as required by the ensemble procedure
            (Section IV D).
        eigenvalue_floor : float
            Eigenvalues of cov_mat below this value are clipped up to it
            before reconstructing L. Default 0.0 simply zeroes out
            negative eigenvalues (numerical noise from the finite-sample
            covariance estimate) without adding any variance that wasn't
            already present in the data. Only raise this above 0.0 if you
            deliberately want to inflate the noise floor -- that is a
            different, explicit choice from merely fixing the
            positive-semidefiniteness issue.
        """
        cov = np.asarray(cov_mat, dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Clip away negative eigenvalues (numerical artefacts of a rank-
        # deficient sample covariance)
        eigvals_clipped = np.clip(eigvals, a_min=eigenvalue_floor, a_max=None)

        # L such that L @ L.T reconstructs the (clipped) covariance matrix
        self.L = eigvecs @ np.diag(np.sqrt(eigvals_clipped))
        self.lattice_mean_a = float(lattice_mean[0])  # C_latt(a)
        self.n_T = cov.shape[0]
        self.rng = np.random.default_rng(seed)

    def sample(self, clean_correlators: np.ndarray) -> np.ndarray:
        """
        Draw one independent noisy correlator for every row of
        clean_correlators.

        Parameters
        ----------
        clean_correlators : ndarray, shape (N, n_T)
            The clean (noise-free) correlator vectors C, e.g. loaded
            straight from load_training_set().

        Returns
        -------
        ndarray, shape (N, n_T)
            Noisy correlators C_noisy, one fresh draw per row.
        """
        clean_correlators = np.asarray(clean_correlators, dtype=np.float64)
        N = clean_correlators.shape[0]

        # Per-sample scalar prefactor (C(a)/C_latt(a))^2, shape (N,)
        C_a = clean_correlators[:, 0]
        prefactor = (C_a / self.lattice_mean_a) ** 2

        # Draw N independent standard-normal vectors, shape (N, n_T)
        z = self.rng.standard_normal(size=(N, self.n_T))

        # Correlate them via the shared Cholesky factor: (z @ L.T) has,
        # row by row, covariance Sigma_latt. Then scale each row by
        # sqrt(prefactor) to get covariance prefactor * Sigma_latt.
        correlated_noise = z @ self.L.T  # (N, n_T)
        scaled_noise = np.sqrt(prefactor)[:, None] * correlated_noise

        return clean_correlators + scaled_noise


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


def mean_absolute_error(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Mean Absolute Error (MAE) loss, as used in Eq. (48) of the paper.

    The paper's norm is the L1 norm summed over energy bins:
        |rho_hat| = sum_{j=1}^{47} |rho_hat_j|

    Then the loss is averaged over the batch:
        L(w) = (1/N_rho) * sum_i |rho_hat_pred_i(w) - rho_hat_i|

    Parameters
    ----------
    predicted : torch.Tensor of shape (batch, 47)
    target    : torch.Tensor of shape (batch, 47)

    Returns
    -------
    Scalar tensor (the loss value).
    """
    abs_diff = nn.functional.l1_loss(
        predicted, target, reduction="none"
    )  # (batch, 47)
    return abs_diff.sum(dim=1).mean()


def beta_nll_loss(
    mu: torch.Tensor,
    sigma2: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """
    beta-NLL loss (Seitzer, Tavakoli, Antic & Martius, ICLR 2022),
    generalised to the 47-dimensional spectral-density output by
    treating the 47 energies as conditionally independent Gaussians
    (diagonal covariance) given the input correlator, and summing their
    contributions -- matching the sum-over-energies convention of the
    paper's own MAE loss, Eq. (48).

        L_beta-NLL = E_{X,Y} [ sum_j sg(sigma2_j^beta) *
                        ( 0.5*log(sigma2_j) + (y_j - mu_j)^2 / (2*sigma2_j) ) ]

    where sg[.] is the stop-gradient operator (implemented below via
    .detach()). beta=0 recovers ordinary NLL. beta=1 makes the
    mean-gradient identical (up to a constant) to the unweighted MSE
    gradient, while sigma2 is still fit to the residual scale.

    Note on the diagonal-covariance assumption: this treats the
    reconstruction error at E_j and E_{j+1} as independent given X, each
    with its own private variance. For a smeared spectral density this
    is an approximation -- the smearing kernel correlates nearby
    energies, so a full 47x47 covariance (e.g. via a Cholesky-factor
    output head) would capture bin-to-bin correlations that this
    diagonal version discards. That's a bigger change than requested
    here; flagging it in case bin-to-bin correlations matter for
    downstream use of the uncertainties (e.g. propagating error into an
    integrated quantity like a spectral moment).

    Parameters
    ----------
    mu     : torch.Tensor, shape (batch, 47) -- predicted mean, mu_hat(E).
    sigma2 : torch.Tensor, shape (batch, 47) -- predicted variance,
             sigma2_hat(E). Must be strictly positive (guaranteed by
             SpectralNet's softplus + sigma2_eps when probabilistic=True).
    target : torch.Tensor, shape (batch, 47) -- true rho_hat_sigma(E).
    beta   : float in [0, 1]. Paper's recommended default is 0.5.

    Returns
    -------
    Scalar tensor: sum over the 47 energies, averaged over the batch.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")

    nll = 0.5 * torch.log(sigma2) + 0.5 * (target - mu) ** 2 / sigma2

    # Stop-gradient weighting factor: computed from sigma2, but detached
    # from the autodiff graph, so it acts as a fixed per-sample multiplier
    # rather than something the optimiser can shift the loss's minimum by
    # adjusting.
    weight = sigma2.detach() ** beta

    weighted_nll = weight * nll

    # Sum over the 47 energy bins (matching Eq. 48's |rho_hat| norm),
    # then average over the batch.
    return weighted_nll.sum(dim=1).mean()


def compute_loss(
    predictions,
    targets: torch.Tensor,
    loss_type: str = "mae",
    beta: float = 0.5,
) -> torch.Tensor:
    """
    Single dispatch point used by train_one_epoch() and validate() so
    that the training loop itself doesn't need to know which loss is
    active.

    Parameters
    ----------
    predictions : whatever model(x) returned.
        For loss_type="mae": a single torch.Tensor, shape (batch, 47).
        For loss_type="beta_nll": a (mu, sigma2) tuple, each (batch, 47)
        -- exactly what SpectralNet(probabilistic=True) returns.
    targets   : torch.Tensor, shape (batch, 47).
    loss_type : "mae" or "beta_nll".
    beta      : only used when loss_type == "beta_nll".

    Returns
    -------
    Scalar loss tensor.
    """
    if loss_type == "mae":
        return mean_absolute_error(predictions, targets)
    elif loss_type == "beta_nll":
        mu, sigma2 = predictions
        return beta_nll_loss(mu, sigma2, targets, beta=beta)
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'; expected 'mae' or 'beta_nll'."
        )


def predict_mean(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Convenience wrapper that returns just the point-estimate/mean
    prediction, regardless of whether `model` is a probabilistic
    (beta-NLL) or plain (MAE) SpectralNet. Useful for downstream
    analysis code (e.g. the ensemble error-budget procedure of Sec. V)
    that wants rho_hat_pred(E) without needing to branch on loss_type
    itself.

    Parameters
    ----------
    model : nn.Module (a SpectralNet, in either mode)
    x     : torch.Tensor of shape (batch, 64)

    Returns
    -------
    torch.Tensor of shape (batch, 47)
    """
    output = model(x)
    if isinstance(output, tuple):
        mu, _sigma2 = output
        return mu
    return output


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================


def make_lr_lambda(warmup_epochs: int = 25, decay_rate: float = 4e-4):
    """
    Returns a lambda function implementing the learning rate schedule of Eq. (49):

        eta(e) = eta(e-1) / (1 + e * decay_rate)   for e > warmup_epochs
        eta(e) = eta(0)                              for e <= warmup_epochs

    In PyTorch, LambdaLR multiplies the base lr by the value returned
    by this lambda, so we accumulate the decay as a product.

    Parameters
    ----------
    warmup_epochs : int
        Number of epochs during which the learning rate is kept constant.
    decay_rate : float
        Decay coefficient (4e-4 in the paper).

    Returns
    -------
    A callable lr_lambda(epoch) -> float  suitable for
    torch.optim.lr_scheduler.LambdaLR.
    """

    def lr_lambda(epoch: int) -> float:
        if epoch <= warmup_epochs:
            return 1.0
        # Accumulate the decay factor for all epochs after warmup
        factor = 1.0
        for e in range(warmup_epochs + 1, epoch + 1):
            factor /= 1.0 + e * decay_rate
        return factor

    return lr_lambda


# =============================================================================
# TRAINING LOOP SKELETON
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_type: str = "mae",
    beta: float = 0.5,
) -> float:
    """
    Run one epoch of mini-batch gradient descent.

    Parameters
    ----------
    model       : The neural network being trained.
    optimizer   : Adam optimiser (or any other).
    train_loader: DataLoader yielding (correlator, spectral_density) pairs.
    device      : 'cpu' or 'cuda'.
    loss_type   : "mae" or "beta_nll" -- must match model.probabilistic
                  (False for "mae", True for "beta_nll").
    beta        : only used when loss_type == "beta_nll".

    Returns
    -------
    float : Average training loss for this epoch.
    """
    model.train()  # puts model in training mode (enables dropout etc. if present)
    total_loss = 0.0

    for correlators, targets in train_loader:
        correlators = correlators.to(device)
        targets = targets.to(device)

        # Zero out gradients from the previous mini-batch
        optimizer.zero_grad()

        # Forward pass: compute predictions (tensor, or (mu, sigma2) tuple)
        predictions = model(correlators)

        # Compute the active loss
        loss = compute_loss(
            predictions, targets, loss_type=loss_type, beta=beta
        )

        # Backward pass: compute gradients of the loss w.r.t. all parameters
        loss.backward()

        # Update parameters using the gradients
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_type: str = "mae",
    beta: float = 0.5,
) -> float:
    """
    Evaluate the model on the validation set without updating weights.

    Parameters
    ----------
    model      : The neural network.
    val_loader : DataLoader for the validation set (20% of training data).
    device     : 'cpu' or 'cuda'.
    loss_type  : "mae" or "beta_nll" -- must match model.probabilistic.
    beta       : only used when loss_type == "beta_nll".

    Returns
    -------
    float : Average validation loss.
    """
    model.eval()  # disables gradient tracking (saves memory and compute)
    total_loss = 0.0

    with torch.no_grad():
        for correlators, targets in val_loader:
            correlators = correlators.to(device)
            targets = targets.to(device)

            predictions = model(correlators)
            loss = compute_loss(
                predictions, targets, loss_type=loss_type, beta=beta
            )
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_replica(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    initial_lr: float = 2e-4,
    warmup_epochs: int = 25,
    patience: int = 15,
    min_delta: float = 1e-5,
    max_epochs: int = 500,
    loss_type: str = "mae",
    beta: float = 0.5,
) -> dict:
    """
    Full training procedure for a single replica, implementing:
      - Adam optimiser with initial lr = 2e-4 (Eq. 49)
      - Learning rate decay schedule (Eq. 49)
      - Early stopping when validation loss improvement < min_delta
        for `patience` consecutive epochs (paper uses patience=15, delta=1e-5)

    Parameters
    ----------
    model         : A freshly initialised SpectralNet. Its `probabilistic`
                    flag must match `loss_type` (see train_ensemble, which
                    guarantees this automatically).
    train_loader  : DataLoader for the 80% training split.
    val_loader    : DataLoader for the 20% validation split.
    device        : torch.device.
    initial_lr    : Starting learning rate (eta_0 = 2e-4 in the paper).
    warmup_epochs : Epochs before lr decay begins (25 in the paper).
    patience      : Early stopping patience in epochs (15 in the paper).
    min_delta     : Minimum improvement to reset the patience counter.
    max_epochs    : Hard cap on training length.
    loss_type     : "mae" (original paper) or "beta_nll".
    beta          : beta-NLL hyperparameter, only used when
                    loss_type == "beta_nll" (paper's suggested default: 0.5).

    Returns
    -------
    dict with keys 'train_losses', 'val_losses', 'best_epoch'.
    """
    model.to(device)

    # Adam optimiser: combines adaptive learning rates with momentum
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    # Attach the lr schedule from Eq. (49)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(warmup_epochs)
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    no_improve_count = 0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            loss_type=loss_type,
            beta=beta,
        )
        val_loss = validate(
            model, val_loader, device, loss_type=loss_type, beta=beta
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Step the lr scheduler (adjusts lr for the next epoch)
        scheduler.step()

        # --- Early stopping logic ---
        improvement = best_val_loss - val_loss
        if improvement > min_delta:
            best_val_loss = val_loss
            no_improve_count = 0
            best_epoch = epoch
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(
                f"  Early stopping at epoch {epoch} "
                f"(best val loss {best_val_loss:.6f} at epoch {best_epoch})"
            )
            break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
    }


# =============================================================================
# ENSEMBLE TRAINING (the "broad audience of machines")
# =============================================================================


def build_replica_dataloaders(
    clean_correlators: np.ndarray,
    smeared_rho: np.ndarray,
    standardiser: CorrelatorStandardiser,
    noise_injector: NoiseInjector,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    split_seed: int = 0,
):
    """
    Turn one replica's worth of raw numpy arrays into train/val DataLoaders.

    This performs, in order, exactly the steps described in Sections IV B-D:
      1. Draw ONE fresh noisy correlator per clean pair (Eq. 44) --
         independent for every replica because noise_injector carries its
         own per-replica seed.
      2. Standardize the noisy correlators using mu, gamma that were fit
         on the CLEAN correlators (Eq. 45-47) -- these are the same for
         every replica, since they characterise the shared training set,
         not any particular noise draw.
      3. Split the (correlator, target) pairs 80/20 into train/validation
         (Section IV D).
      4. Wrap each split in a DataLoader.

    Parameters
    ----------
    clean_correlators : ndarray, shape (N_rho, n_T)
        Clean correlators C, shared across all replicas.
    smeared_rho : ndarray, shape (N_rho, n_E)
        Target smeared spectral densities rho_hat_sigma, shared across
        all replicas (same pairing as clean_correlators).
    standardiser : CorrelatorStandardiser
        Already fit on the clean correlators (fit once, outside this
        function, and reused for every replica -- see fit_standardiser
        below).
    noise_injector : NoiseInjector
        A NoiseInjector constructed with a seed UNIQUE to this replica.
    batch_size : int
        Mini-batch size (BS = 32 in the paper).
    val_fraction : float
        Fraction of pairs held out for validation / early stopping (0.2
        in the paper, i.e. an 80/20 split).
    split_seed : int
        Seed controlling which pairs land in the validation split.

    Returns
    -------
    (train_loader, val_loader) : tuple of torch.utils.data.DataLoader
    """
    # --- Step 1: fresh noise for this replica ---
    noisy_correlators = noise_injector.sample(
        clean_correlators
    )  # (N, n_T) numpy

    # --- Convert to tensors ---
    noisy_t = torch.from_numpy(noisy_correlators).float()
    targets_t = torch.from_numpy(smeared_rho).float()

    # --- Step 2: standardize using the CLEAN-data statistics ---
    standardized_t = standardiser.transform(noisy_t)

    # --- Step 3+4: split and wrap in DataLoaders ---
    dataset = torch.utils.data.TensorDataset(standardized_t, targets_t)

    n_total = len(dataset)
    n_val = round(n_total * val_fraction)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(split_seed)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


def fit_standardiser(clean_correlators: np.ndarray) -> CorrelatorStandardiser:
    """
    Fit a CorrelatorStandardiser on the clean (noise-free) correlators.

    This is done ONCE, before the replica loop, and the same fitted
    standardiser is reused for every replica -- because mu and gamma
    characterise the shared training set as a whole (Eq. 45-47), not
    any individual replica's noise draw.

    Parameters
    ----------
    clean_correlators : ndarray, shape (N_rho, n_T)

    Returns
    -------
    CorrelatorStandardiser, already fit.
    """
    standardiser = CorrelatorStandardiser()
    clean_t = torch.from_numpy(clean_correlators).float()
    standardiser.fit(clean_t)
    return standardiser


def train_ensemble(
    architecture_fn,
    clean_correlators: np.ndarray,
    smeared_rho: np.ndarray,
    cov_mat: np.ndarray,
    lattice_mean: np.ndarray,
    device: torch.device,
    n_replicas: int = 20,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    base_seed: int = 0,
    loss_type: str = "mae",
    beta: float = 0.5,
) -> tuple[list[nn.Module], CorrelatorStandardiser]:
    """
    Train an ensemble of Nr=20 replicas of the same architecture.

    Each replica starts from a different random weight initialisation
    (handled automatically by SpectralNet.__init__) AND trains on a
    freshly drawn noisy version of the SAME underlying clean pairs
    (C, rho_hat_sigma) -- matching the procedure described on page 12:
    "each machine of the ensemble is trained ... starting from the same
    unsmeared spectral densities ... but with different noisy input
    correlator vectors C_noisy."

    The spread of the resulting predictions across replicas is what
    gives the network systematic error Delta^net_sigma (Eq. 52),
    regardless of loss_type.

    NEW: loss_type selects which loss (and, correspondingly, which
    architecture variant) is used for every replica in the ensemble.
      - loss_type="mae" (default): reproduces the original paper exactly
        -- single-output SpectralNet, trained with mean_absolute_error.
      - loss_type="beta_nll": builds a two-headed SpectralNet
        (probabilistic=True) for every replica and trains with
        beta_nll_loss(..., beta=beta) instead. Each replica then
        produces its own (mu_hat, sigma2_hat) pair; see beta_nll_loss()
        and predict_mean() for how to combine these across the ensemble.

    Parameters
    ----------
    architecture_fn : One of arcS, arcM, arcL (a callable returning a
        SpectralNet; called here as architecture_fn(probabilistic=...),
        so any custom architecture_fn must accept that keyword too).
    clean_correlators : ndarray, shape (N_rho, n_T)
        Clean correlators C, as loaded from your HDF5 file (load_training_set).
    smeared_rho : ndarray, shape (N_rho, n_E)
        Target smeared spectral densities, paired with clean_correlators.
    cov_mat : ndarray, shape (n_T, n_T)
        Covariance matrix of the real lattice correlator (Eq. 44).
    lattice_mean : ndarray, shape (n_T,)
        Mean lattice correlator C_latt(t); only C_latt(a) is used.
    device : torch.device.
    n_replicas : Number of replicas (Nr = 20 in the paper).
    batch_size : Mini-batch size (BS = 32 in the paper).
    val_fraction : Held-out validation fraction (0.2 in the paper).
    base_seed : int
        Seeds are derived from this as base_seed + r for replica r, so
        every replica gets independent-but-reproducible noise and split.
    loss_type : "mae" or "beta_nll" (see above).
    beta : beta-NLL hyperparameter in [0, 1], only used when
        loss_type == "beta_nll".

    Returns
    -------
    (ensemble, standardiser) : tuple
        ensemble     : list of Nr trained nn.Module instances (each a
                       SpectralNet with .probabilistic matching loss_type).
        standardiser : the CorrelatorStandardiser fit on clean_correlators.
                       MUST be saved alongside the models and reused,
                       unchanged, to pre-process any correlator (mock,
                       validation, or real lattice data) fed to this
                       ensemble later.
    """
    probabilistic = loss_type == "beta_nll"

    # Standardization statistics come from the CLEAN correlators and are
    # shared by every replica -- fit this exactly once, outside the loop.
    standardiser = fit_standardiser(clean_correlators)

    ensemble = []

    for r in range(1, n_replicas + 1):
        print(
            f"Training replica {r}/{n_replicas} (loss_type='{loss_type}') ..."
        )

        # A distinct seed per replica ensures independent noise draws --
        # this is the piece that was previously missing.
        replica_seed = base_seed + r
        noise_injector = NoiseInjector(
            cov_mat=cov_mat, lattice_mean=lattice_mean, seed=replica_seed
        )

        train_loader, val_loader = build_replica_dataloaders(
            clean_correlators=clean_correlators,
            smeared_rho=smeared_rho,
            standardiser=standardiser,
            noise_injector=noise_injector,
            batch_size=batch_size,
            val_fraction=val_fraction,
            split_seed=replica_seed,
        )

        # Each call to architecture_fn() creates a fresh model with new
        # random weights (because _initialise_weights uses torch.randn
        # internally), with a single or two-headed output depending on
        # loss_type.
        model = architecture_fn(probabilistic=probabilistic)

        history = train_replica(
            model,
            train_loader,
            val_loader,
            device,
            loss_type=loss_type,
            beta=beta,
        )

        print(
            f"  Replica {r} finished. "
            f"Best val loss: {min(history['val_losses']):.6f} "
            f"at epoch {history['best_epoch']}"
        )

        ensemble.append(model)

    return ensemble, standardiser


# =============================================================================
# SAVING / LOADING A TRAINED ENSEMBLE
# =============================================================================


def save_ensemble(
    ensemble: list[nn.Module],
    standardiser: CorrelatorStandardiser,
    save_dir: str,
    metadata: dict | None = None,
) -> None:
    """
    Save a trained ensemble to disk so it can be reloaded later for
    use on real data.

    Two things are saved together:
      1. Each replica's weights.
      2. The standardiser's mu and gamma -- without these, any future
         correlator cannot be correctly pre-processed before being fed
         into the network.

    We save model.state_dict() rather than pickling the whole nn.Module.
    state_dict() is just an ordered dict of tensor names -> tensor
    values (weights and biases).

    IMPORTANT: metadata should include "loss_type" (and, for
    loss_type="beta_nll", "beta") so that load_ensemble() can correctly
    reconstruct probabilistic vs. non-probabilistic architectures before
    loading the saved weights back in -- see load_ensemble().

    Parameters
    ----------
    ensemble : list of trained nn.Module (all the same architecture).
    standardiser : the CorrelatorStandardiser used during training.
    save_dir : directory to write into (created if it doesn't exist).
    metadata : optional dict of extra info you want recorded alongside
        the weights, e.g. {"architecture": "arcS", "n_b": 512,
        "n_rho": 800_000, "sigma": 0.44, "loss_type": "beta_nll",
        "beta": 0.5}.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Save each replica's weights ---
    for r, model in enumerate(ensemble, start=1):
        weights_path = os.path.join(save_dir, f"replica_{r:02d}.pt")
        torch.save(model.state_dict(), weights_path)

    # --- Save the standardiser's mu and gamma ---
    standardiser_path = os.path.join(save_dir, "standardiser.pt")
    torch.save(
        {"mu": standardiser.mu, "gamma": standardiser.gamma},
        standardiser_path,
    )

    # --- Save metadata as plain JSON, so it's human-readable ---
    metadata = dict(metadata) if metadata is not None else {}
    metadata.setdefault("loss_type", "mae")
    metadata["n_replicas"] = len(ensemble)
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(ensemble)} replicas + standardiser to {save_dir}")


def load_ensemble(
    architecture_fn, load_dir: str, device: torch.device
) -> tuple[list[nn.Module], CorrelatorStandardiser, dict]:
    """
    Load a previously saved ensemble back into memory.

    Whether each replica is rebuilt as a probabilistic (two-headed) or
    plain (single-headed) SpectralNet is read automatically from
    metadata["loss_type"] (defaulting to "mae" if absent, for backward
    compatibility with ensembles saved before this option existed) --
    so the caller does not need to track this separately.

    Parameters
    ----------
    architecture_fn : the same architecture function used at training
        time (e.g. arcS) -- must match, since state_dict only contains
        weight values, not the architecture that produced them. Loading
        arcS weights into an arcM model would fail with a shape mismatch.
    load_dir : directory previously written by save_ensemble().
    device : torch.device to load the models onto.

    Returns
    -------
    (ensemble, standardiser, metadata)
        ensemble     : list of nn.Module, each in eval() mode.
        standardiser : CorrelatorStandardiser with mu, gamma restored.
        metadata     : the dict saved alongside the weights.
    """
    # --- Load metadata first, to know how many replicas to expect and
    #     whether the saved replicas are probabilistic ---
    metadata_path = os.path.join(load_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    n_replicas = metadata["n_replicas"]
    probabilistic = metadata.get("loss_type", "mae") == "beta_nll"

    # --- Rebuild the standardiser ---
    standardiser_path = os.path.join(load_dir, "standardiser.pt")
    saved_stats = torch.load(standardiser_path, map_location=device)
    standardiser = CorrelatorStandardiser()
    standardiser.mu = saved_stats["mu"].to(device)
    standardiser.gamma = saved_stats["gamma"].to(device)

    # --- Rebuild each replica ---
    ensemble = []
    for r in range(1, n_replicas + 1):
        weights_path = os.path.join(load_dir, f"replica_{r:02d}.pt")
        state_dict = torch.load(weights_path, map_location=device)

        # fresh model, correct architecture AND correct number of heads
        model = architecture_fn(probabilistic=probabilistic)
        model.load_state_dict(state_dict)  # overwrite its random weights
        model.to(device)
        model.eval()  # inference mode, not training mode

        ensemble.append(model)

    return ensemble, standardiser, metadata


# =============================================================================

if __name__ == "__main__":
    from TrainingSets_Full_Chebyshev import load_training_set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # lattice data from the 48 ensemble for the 2pi.g5.001 operator
    # we remove the irrelevant t=0 time-slice
    two_pt_data = np.load("noise_injection_data/2pt_data_64_folded.npy")[:, 1:]
    lattice_mean = np.mean(two_pt_data, axis=0)
    cov_mat = np.cov(two_pt_data, rowvar=False, bias=False)

    # One (architecture, n_b, n_rho, sigma) combination from Table IV.
    # This example reproduces the N^ref_n row: arcS, Nb=512, Nrho=800k.
    sigma = 0.44
    n_b = 32
    n_rho = 50_000

    # --- NEW: choose the loss function for this run ---
    # "mae"      : reproduces the original paper exactly (single output head).
    # "beta_nll" : trains a two-headed (mu_hat, sigma2_hat) network per
    #              replica instead; see beta_nll_loss() above. beta=0.5
    #              is the paper's (Seitzer et al.) suggested default.
    loss_type = "beta_nll"
    beta = 0.5

    data = load_training_set(
        filepath=f"Training_sets/64_data/ts_nb{n_b}.h5",
        sigma=sigma,
        n_rho=n_rho,
    )
    correlators = data["correlators"]
    smeared_rho = data["smeared_rho"]

    ensemble, standardiser = train_ensemble(
        architecture_fn=arcS,
        clean_correlators=correlators,
        smeared_rho=smeared_rho,
        cov_mat=cov_mat,
        lattice_mean=lattice_mean,
        device=device,
        n_replicas=20,
        batch_size=32,
        val_fraction=0.2,
        base_seed=0,
        loss_type=loss_type,
        beta=beta,
    )

    save_ensemble(
        ensemble,
        standardiser,
        save_dir=f"trained_models/arcS_nb{n_b}_nrho{n_rho}_sigma{sigma}_{loss_type}",
        metadata={
            "architecture": "arcS",
            "n_b": n_b,
            "n_rho": n_rho,
            "sigma": sigma,
            "loss_type": loss_type,
            "beta": beta,
        },
    )
