"""
PyTorch implementation of the three neural network architectures (arcS, arcM, arcL)
from "Teaching to extract spectral densities from lattice correlators to a broad
audience of learning-machines" (Buzzicotti, De Santis, Tantalo, 2024).

Each network solves a regression problem: given a 64-component Euclidean correlator
C(t) as input, predict a 47-component smeared spectral density rho_hat(E) as output.

Architecture overview (same structure for all three, differing only in the number
of convolutional "maps"):

    Input (64,)
        |
    Conv1D block x3  <-- learns local/abstract features from the time-series
        |
    Flatten
        |
    Fully connected x2  <-- combines features into a global prediction
        |
    Output (47,)
"""

import torch
import torch.nn as nn

# =============================================================================
# BUILDING BLOCKS
# =============================================================================


class SpectralNet(nn.Module):
    """
    Generic feed-forward convolutional network for spectral density extraction.

    All three architectures (arcS, arcM, arcL) share the same skeleton:
      - Three 1D convolutional layers with increasing numbers of feature maps
      - Two fully connected layers
      - LeakyReLU activations throughout (except the output layer)

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
    """

    def __init__(
        self,
        base_maps: int,
        input_size: int = 64,
        output_size: int = 47,
        leaky_slope: float = 0.2,
    ):
        super().__init__()

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
        # FULLY CONNECTED BLOCK
        # ------------------------------------------------------------------
        # Two dense layers of 256 neurons each, both with LeakyReLU.
        # The final output layer has NO activation function — this is
        # intentional so the network's output range is unconstrained
        # (spectral densities can be negative, as noted in the paper).
        # ------------------------------------------------------------------

        self.fc_block = nn.Sequential(
            # Flatten: (batch, maps3, 8) -> (batch, maps3*8)
            nn.Flatten(),
            # Dense layer 1: flatten_size -> 256
            nn.Linear(flatten_size, 256),
            nn.LeakyReLU(negative_slope=leaky_slope),
            # Dense layer 2: 256 -> 256
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=leaky_slope),
            # Output layer: 256 -> 47 (no activation)
            nn.Linear(256, output_size),
        )

        # ------------------------------------------------------------------
        # WEIGHT INITIALISATION
        # ------------------------------------------------------------------
        # The paper initialises each weight from a Gaussian with mean 0 and
        # variance 0.05 (i.e. std = sqrt(0.05) ≈ 0.2236).
        # PyTorch's default initialisation (Kaiming uniform) is different,
        # so we override it here.
        # Biases are initialised to zero, which is standard practice.
        # ------------------------------------------------------------------
        self._initialise_weights(std=0.2236)

    def _initialise_weights(self, std: float):
        """Apply Gaussian weight initialisation to all Linear and Conv1d layers."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, 64)
            A batch of (pre-processed) correlator vectors.

        Returns
        -------
        torch.Tensor of shape (batch_size, 47)
            Predicted smeared spectral density vectors.
        """
        # Conv1d requires a channel dimension: (batch, 64) -> (batch, 1, 64)
        x = x.unsqueeze(1)

        # Pass through the three convolutional layers
        x = self.conv_block(x)

        # Flatten and pass through the two fully connected layers + output
        x = self.fc_block(x)

        return x


# =============================================================================
# THE THREE NAMED ARCHITECTURES
# =============================================================================


def arcS(input_size: int = 64, output_size: int = 47) -> SpectralNet:
    """
    arcS: the smallest architecture (~94,651 parameters).
    Conv maps: 2 -> 4 -> 8.
    """
    return SpectralNet(
        base_maps=2, input_size=input_size, output_size=output_size
    )


def arcM(input_size: int = 64, output_size: int = 47) -> SpectralNet:
    """
    arcM: the medium architecture (~180,871 parameters).
    Conv maps: 12 -> 24 -> 48.
    """
    return SpectralNet(
        base_maps=12, input_size=input_size, output_size=output_size
    )


def arcL(input_size: int = 64, output_size: int = 47) -> SpectralNet:
    """
    arcL: the largest architecture (~371,311 parameters).
    Conv maps: 32 -> 64 -> 128.
    """
    return SpectralNet(
        base_maps=32, input_size=input_size, output_size=output_size
    )


# =============================================================================
# PARAMETER COUNTING UTILITY
# =============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
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
# LOSS FUNCTION
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

    PyTorch's nn.L1Loss with reduction='mean' does exactly this when the
    target has shape (batch, 47).

    Parameters
    ----------
    predicted : torch.Tensor of shape (batch, 47)
    target    : torch.Tensor of shape (batch, 47)

    Returns
    -------
    Scalar tensor (the loss value).
    """
    return nn.functional.l1_loss(predicted, target, reduction="mean")


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
    A callable lr_lambda(epoch) -> float  suitable for torch.optim.lr_scheduler.LambdaLR.
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
) -> float:
    """
    Run one epoch of mini-batch gradient descent.

    Parameters
    ----------
    model       : The neural network being trained.
    optimizer   : Adam optimiser (or any other).
    train_loader: DataLoader yielding (correlator, spectral_density) pairs.
    device      : 'cpu' or 'cuda'.

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

        # Forward pass: compute predictions
        predictions = model(correlators)

        # Compute MAE loss
        loss = mean_absolute_error(predictions, targets)

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
) -> float:
    """
    Evaluate the model on the validation set without updating weights.

    Parameters
    ----------
    model      : The neural network.
    val_loader : DataLoader for the validation set (20% of training data).
    device     : 'cpu' or 'cuda'.

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
            loss = mean_absolute_error(predictions, targets)
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
) -> dict:
    """
    Full training procedure for a single replica, implementing:
      - Adam optimiser with initial lr = 2e-4 (Eq. 49)
      - Learning rate decay schedule (Eq. 49)
      - Early stopping when validation loss improvement < min_delta
        for `patience` consecutive epochs (paper uses patience=15, delta=1e-5)

    Parameters
    ----------
    model         : A freshly initialised SpectralNet.
    train_loader  : DataLoader for the 80% training split.
    val_loader    : DataLoader for the 20% validation split.
    device        : torch.device.
    initial_lr    : Starting learning rate (eta_0 = 2e-4 in the paper).
    warmup_epochs : Epochs before lr decay begins (25 in the paper).
    patience      : Early stopping patience in epochs (15 in the paper).
    min_delta     : Minimum improvement to reset the patience counter.
    max_epochs    : Hard cap on training length.

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
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = validate(model, val_loader, device)

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


def train_ensemble(
    architecture_fn,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_replicas: int = 20,
) -> list[nn.Module]:
    """
    Train an ensemble of Nr=20 replicas of the same architecture.

    Each replica starts from a different random weight initialisation
    (handled automatically by SpectralNet.__init__) but is trained on
    the SAME training set. The spread of their predictions is used as
    the network systematic error Delta^net_sigma (Eq. 52).

    Parameters
    ----------
    architecture_fn : One of arcS, arcM, arcL (a callable returning a model).
    train_loader    : DataLoader for the training set.
    val_loader      : DataLoader for the validation set.
    device          : torch.device.
    n_replicas      : Number of replicas (Nr = 20 in the paper).

    Returns
    -------
    list of trained nn.Module instances.
    """
    ensemble = []

    for r in range(1, n_replicas + 1):
        print(f"Training replica {r}/{n_replicas} ...")

        # Each call to architecture_fn() creates a fresh model with new
        # random weights (because _initialise_weights uses torch.randn internally)
        model = architecture_fn()

        history = train_replica(model, train_loader, val_loader, device)

        print(
            f"  Replica {r} finished. "
            f"Best val loss: {min(history['val_losses']):.6f} "
            f"at epoch {history['best_epoch']}"
        )

        ensemble.append(model)

    return ensemble


# =============================================================================
# QUICK SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Architecture summary")
    print("=" * 60)

    for name, fn in [("arcS", arcS), ("arcM", arcM), ("arcL", arcL)]:
        net = fn()
        n_params = count_parameters(net)
        print(f"\n{name}: {n_params:,} trainable parameters")

        # Print each layer with its output shape for a single sample
        dummy = torch.zeros(1, 64)  # batch of 1 correlator
        with torch.no_grad():
            out = net(dummy)
        print(f"  Input shape:  {dummy.shape}")
        print(f"  Output shape: {out.shape}")

        # Walk through layers manually to show intermediate shapes
        x = dummy.unsqueeze(1)  # (1, 1, 64)
        print(f"  After unsqueeze:        {x.shape}")
        for i, layer in enumerate(net.conv_block):
            x = layer(x)
            print(f"  After conv_block[{i}]:    {x.shape}")
        for i, layer in enumerate(net.fc_block):
            x = layer(x)
            print(f"  After fc_block[{i}]:      {x.shape}")

    print("\n" + "=" * 60)
    print("Standardiser demo")
    print("=" * 60)
    fake_correlators = torch.randn(1000, 64).abs()  # positive fake data
    fake_correlators *= torch.exp(
        -torch.linspace(0, 5, 64)
    )  # exponential decay
    standardizer = CorrelatorStandardiser()
    standardized = standardizer.fit_transform(fake_correlators)
    print(
        f"Before: mean={fake_correlators.mean():.3f}, "
        f"std={fake_correlators.std():.3f}"
    )
    print(
        f"After:  mean={standardized.mean():.3f}, "
        f"std={standardized.std():.3f}"
    )
