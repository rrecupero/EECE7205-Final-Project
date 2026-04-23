"""
Training Script — Linear Transformer for Trajectory Prediction
================================================================
Loads preprocessed data, trains the model, evaluates, and saves results.

What this script does:
    1. Load train.pt, val.pt, test.pt from Preprocessed/
    2. Create PyTorch DataLoaders (batching + shuffling)
    3. Train the Linear Transformer with MSE loss + Adam optimizer
    4. Track training/validation loss per epoch
    5. Evaluate on test set with ADE and FDE metrics
    6. Save trained model + training curves
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (saves plots without displaying)
import matplotlib.pyplot as plt

# Import your Linear Transformer model
from linear_transformer import LinearTransformerPredictor

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Dataset" / "Preprocessed"
RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Training Hyperparameters ──
TRAIN_CONFIG = {
    # Model architecture (must match what we tested)
    "input_dim": 6,
    "output_dim": 3,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 3,
    "d_ff": 128,
    "pred_len": 10,
    "dropout": 0.1,

    # Training parameters
    "batch_size": 256,       # Number of windows per gradient update
    "learning_rate": 1e-3,  # How big each weight update step is
    "weight_decay": 1e-5,   # L2 regularization to prevent overfitting
    "epochs": 50,           # Number of full passes through training data
    "patience": 10,         # Early stopping: stop if val loss doesn't
                            # improve for this many epochs

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data():
    """
    Load preprocessed PyTorch tensors.

    What's inside each .pt file?
    ----------------------------
    A dictionary with two keys:
        "X": tensor of shape (num_windows, observe_len, input_features)
             = (num_windows, 20, 6) — the observation history
        "Y": tensor of shape (num_windows, predict_len, output_features)
             = (num_windows, 10, 3) — the future trajectory to predict

    These are already normalized (zero mean, unit std) from the
    preprocessing step, so they're ready to feed directly into the model.
    """
    print("Loading preprocessed data...")

    train_data = torch.load(DATA_DIR / "train.pt", weights_only=True)
    val_data = torch.load(DATA_DIR / "val.pt", weights_only=True)
    test_data = torch.load(DATA_DIR / "test.pt", weights_only=True)
    norm_stats = torch.load(DATA_DIR / "norm_stats.pt", weights_only=True)

    print(f"  Train: X={train_data['X'].shape}, Y={train_data['Y'].shape}")
    print(f"  Val:   X={val_data['X'].shape},  Y={val_data['Y'].shape}")
    print(f"  Test:  X={test_data['X'].shape},  Y={test_data['Y'].shape}")

    return train_data, val_data, test_data, norm_stats


def create_dataloaders(train_data, val_data, test_data, batch_size, device):
    """
    Wrap tensors in DataLoaders for batched iteration.

    What is a DataLoader?
    ---------------------
    You can't feed all 24,388 training windows to the model at once —
    that would require too much memory. Instead, we process them in
    small batches (e.g., 256 windows at a time).

    DataLoader handles:
    - Splitting data into batches of size batch_size
    - SHUFFLING training data each epoch (so the model sees different
      orderings, which prevents it from memorizing the sequence order)
    - NOT shuffling val/test data (order doesn't matter for evaluation)

    What is TensorDataset?
    ----------------------
    Pairs up X and Y tensors so that when you iterate, you get
    matching (input, target) pairs. dataset[i] returns (X[i], Y[i]).
    """
    pin = (device == "cuda")

    train_loader = DataLoader(
        TensorDataset(train_data["X"], train_data["Y"]),
        batch_size=batch_size,
        shuffle=True,   # Randomize order each epoch for better training
        num_workers=4,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        TensorDataset(val_data["X"], val_data["Y"]),
        batch_size=batch_size,
        shuffle=False,   # No need to shuffle evaluation data
        num_workers=4,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        TensorDataset(test_data["X"], test_data["Y"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin,
    )

    print(f"\n  Train batches: {len(train_loader)} "
          f"(batch_size={batch_size})")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


# =============================================================================
# 2. TRAINING LOOP
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one full pass through the training data.

    What happens in one epoch:
    --------------------------
    1. Iterate over all batches in the training set
    2. For each batch:
       a. FORWARD PASS: feed input through model, get predictions
       b. COMPUTE LOSS: compare predictions to ground truth (MSE)
       c. BACKWARD PASS: compute gradients of loss w.r.t. all weights
       d. UPDATE WEIGHTS: optimizer adjusts weights to reduce loss
       e. ZERO GRADIENTS: reset for next batch (PyTorch accumulates
          gradients by default, so we must clear them)

    model.train() vs model.eval():
    ------------------------------
    model.train() enables dropout (randomly zeroing values) and other
    training-specific behaviors. This acts as regularization — it
    forces the model to not rely on any single neuron, making it more
    robust. During evaluation, we disable dropout with model.eval()
    because we want the full model capacity for the best predictions.

    Returns the average loss across all batches.
    """
    model.train()  # Enable training mode (dropout active)
    total_loss = 0.0
    n_batches = 0

    for X_batch, Y_batch in loader:
        # Move data to device (CPU or GPU)
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        # Forward pass: input → model → predictions
        predictions = model(X_batch)  # (batch, 10, 3)

        # Compute loss: how wrong are the predictions?
        # MSE = mean of (prediction - target)² across all values
        loss = criterion(predictions, Y_batch)

        # Backward pass: compute gradients
        # PyTorch builds a computation graph during forward pass.
        # loss.backward() traverses this graph in reverse, computing
        # ∂loss/∂weight for every trainable parameter.
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients

        # Update weights: w_new = w_old - learning_rate * gradient
        # Adam optimizer also uses momentum and adaptive learning rates
        # per parameter, which is why it works better than plain SGD.
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()  # Disable gradient computation (saves memory + speeds up)
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation or test data.

    @torch.no_grad() means we don't track gradients during evaluation.
    This saves memory and computation because we're not going to call
    backward() — we just want to measure performance, not train.

    model.eval() disables dropout so we get deterministic predictions
    using the full model capacity.
    """
    model.eval()  # Disable dropout
    total_loss = 0.0
    n_batches = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        predictions = model(X_batch)
        loss = criterion(predictions, Y_batch)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train_model(model, train_loader, val_loader, config):
    """
    Full training loop with early stopping.

    What is early stopping?
    -----------------------
    During training, the model keeps getting better on training data.
    But at some point, it starts OVERFITTING — memorizing the training
    data instead of learning general patterns. When this happens,
    training loss keeps decreasing but validation loss starts INCREASING.

    Early stopping monitors validation loss and stops training when it
    hasn't improved for 'patience' epochs. This gives you the model
    that generalizes best, not the one that memorized the most.

    We save the model weights whenever validation loss reaches a new low
    (the "best" model). At the end, we load back these best weights.
    """
    device = config["device"]
    print(f"\nTraining on: {device}")

    model = model.to(device)

    # Loss function: Mean Squared Error
    # MSE = (1/N) Σ (predicted - actual)²
    # This penalizes large errors more than small ones (quadratic penalty),
    # which encourages the model to avoid big mistakes.
    criterion = nn.MSELoss()

    # Optimizer: Adam (Adaptive Moment Estimation)
    # Adam maintains per-parameter learning rates that adapt based on
    # the history of gradients. Parameters with consistently large
    # gradients get smaller learning rates (and vice versa). This makes
    # training more stable and faster than plain SGD.
    # weight_decay adds L2 regularization: penalizes large weights
    # to prevent overfitting.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Learning rate scheduler: reduce LR when val loss plateaus
    # If validation loss hasn't improved for 5 epochs, multiply LR by 0.5.
    # This helps the model make finer adjustments as it approaches
    # the optimal weights — like switching from big steps to small steps
    # when you're close to the bottom of a valley.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Tracking variables
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    print(f"\nStarting training for up to {config['epochs']} epochs...")
    print(f"  Early stopping patience: {config['patience']} epochs")
    print(f"{'─' * 70}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} "
          f"{'LR':>10} {'Time':>8} {'Status':>10}")
    print(f"{'─' * 70}")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        # Train one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluate on validation set
        val_loss = evaluate(model, val_loader, criterion, device)

        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        # Check if this is the best model so far
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            status = "★ best"
        else:
            patience_counter += 1

        print(f"{epoch:>6} {train_loss:>12.6f} {val_loss:>12.6f} "
              f"{current_lr:>10.6f} {elapsed:>7.1f}s {status:>10}")

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Early stopping check
        if patience_counter >= config["patience"]:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {config['patience']} epochs)")
            break

    print(f"{'─' * 70}")
    print(f"  Best epoch: {best_epoch} with val loss: {best_val_loss:.6f}")

    # Load best model weights
    model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, best_epoch


# =============================================================================
# 3. EVALUATION METRICS — ADE and FDE
# =============================================================================
@torch.no_grad()
def compute_metrics(model, loader, norm_stats, device):
    """
    Compute ADE and FDE on a dataset.

    These are the standard metrics in trajectory prediction literature.
    They must be computed in REAL-WORLD coordinates (meters), not in
    normalized space. So we denormalize predictions and targets first.

    ADE (Average Displacement Error):
    ----------------------------------
    The average Euclidean distance between predicted and actual positions
    across ALL predicted timesteps.

        ADE = (1/N·T) Σᵢ Σₜ √((x̂ᵢₜ - xᵢₜ)² + (ŷᵢₜ - yᵢₜ)²)

    This tells you: "on average, how far off is each predicted position?"
    An ADE of 0.5 means the model's predictions are typically 0.5 meters
    away from where the robot actually was.

    FDE (Final Displacement Error):
    --------------------------------
    The Euclidean distance between the predicted and actual positions at
    the LAST predicted timestep only.

        FDE = (1/N) Σᵢ √((x̂ᵢT - xᵢT)² + (ŷᵢT - yᵢT)²)

    This tells you: "how far off is the prediction at the end of the
    prediction horizon?" FDE is usually larger than ADE because errors
    accumulate over time — the further into the future you predict,
    the more uncertain the prediction becomes.

    Why both metrics?
    -----------------
    ADE gives the average picture. FDE gives the worst-case picture
    (at maximum prediction distance). A model might have good ADE but
    bad FDE if it's accurate for the first few steps but drifts badly
    by the end. Reporting both gives a complete picture.
    """
    model.eval()

    # Get denormalization stats for output features (x, y, yaw)
    out_mean = norm_stats["output_mean"].to(device)  # shape: (3,)
    out_std = norm_stats["output_std"].to(device)    # shape: (3,)

    all_ade = []
    all_fde = []

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        # Get predictions in normalized space
        pred = model(X_batch)  # (batch, 10, 3)

        # Denormalize: convert from normalized → real meters/radians
        # real_value = normalized_value * std + mean
        pred_real = pred * out_std + out_mean
        target_real = Y_batch * out_std + out_mean

        # Compute Euclidean distance at each timestep (x, y only — index 0, 1)
        # We exclude yaw (index 2) because displacement error is a
        # spatial metric measured in meters, not radians.
        dx = pred_real[:, :, 0] - target_real[:, :, 0]  # x error
        dy = pred_real[:, :, 1] - target_real[:, :, 1]  # y error
        distances = torch.sqrt(dx**2 + dy**2)  # (batch, 10)

        # ADE: average distance across all predicted timesteps
        ade = distances.mean(dim=1)  # (batch,)
        all_ade.append(ade)

        # FDE: distance at the LAST predicted timestep
        fde = distances[:, -1]  # (batch,)
        all_fde.append(fde)

    # Concatenate all batches and compute final metrics
    all_ade = torch.cat(all_ade).cpu().numpy()
    all_fde = torch.cat(all_fde).cpu().numpy()

    return {
        "ADE_mean": float(np.mean(all_ade)),
        "ADE_std": float(np.std(all_ade)),
        "FDE_mean": float(np.mean(all_fde)),
        "FDE_std": float(np.std(all_fde)),
    }


# =============================================================================
# 4. MEASURE INFERENCE TIME
# =============================================================================
@torch.no_grad()
def measure_inference_time(model, device, input_shape=(1, 20, 6), n_runs=100):
    """
    Measure how fast the model produces predictions.

    This is critical for your paper — you need to show that the Linear
    Transformer is faster than the Standard Transformer, especially
    as sequence length increases.

    We run the model n_runs times on a single input and measure the
    average time. We use batch_size=1 because in real-time deployment,
    the robot processes one trajectory at a time.

    Warmup runs are important because the first few runs include
    overhead from memory allocation, kernel compilation (on GPU),
    and other one-time costs that would skew the measurement.
    """
    model.eval()
    model = model.to(device)
    dummy = torch.randn(*input_shape).to(device)

    # Warmup (10 runs to stabilize)
    for _ in range(10):
        _ = model(dummy)

    # Timed runs
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_runs):
        _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) / n_runs * 1000
    return avg_ms


# =============================================================================
# 5. PLOTTING
# =============================================================================
def plot_training_curves(train_losses, val_losses, best_epoch):
    """
    Plot training and validation loss over epochs.

    What to look for:
    -----------------
    - Both curves should decrease initially (model is learning)
    - Training loss should be slightly lower than val loss (normal)
    - If val loss increases while train loss decreases: OVERFITTING
    - The star marks where we saved the best model
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", linewidth=1.5, label="Train Loss")
    ax.plot(epochs, val_losses, "r-", linewidth=1.5, label="Val Loss")
    ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
               label=f"Best Epoch ({best_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Linear Transformer — Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    print(f"  Saved: training_curves.png")


@torch.no_grad()
def plot_sample_predictions(model, test_loader, norm_stats, device, n_samples=4):
    """
    Visualize actual vs predicted trajectories for a few test samples.

    This is the most intuitive plot — you can literally SEE how well
    the model predicts. Blue = observed past, green = actual future,
    red = predicted future. If red and green overlap closely, the
    model is working well.
    """
    model.eval()
    out_mean = norm_stats["output_mean"].to(device)
    out_std = norm_stats["output_std"].to(device)
    in_mean = norm_stats["input_mean"].to(device)
    in_std = norm_stats["input_std"].to(device)

    # Get one batch
    X_batch, Y_batch = next(iter(test_loader))
    X_batch = X_batch[:n_samples].to(device)
    Y_batch = Y_batch[:n_samples].to(device)

    pred = model(X_batch)

    # Denormalize and detach from computation graph before numpy conversion
    pred_real = (pred * out_std + out_mean).detach().cpu().numpy()
    target_real = (Y_batch * out_std + out_mean).detach().cpu().numpy()
    input_real = (X_batch * in_std + in_mean).detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i in range(n_samples):
        ax = axes[i]

        # Observed trajectory (input): x, y are features 0, 1
        obs_x = input_real[i, :, 0]
        obs_y = input_real[i, :, 1]

        # Actual future
        fut_x = target_real[i, :, 0]
        fut_y = target_real[i, :, 1]

        # Predicted future
        pred_x = pred_real[i, :, 0]
        pred_y = pred_real[i, :, 1]

        ax.plot(obs_x, obs_y, "b.-", linewidth=2, markersize=4,
                label="Observed", alpha=0.7)
        ax.plot(fut_x, fut_y, "g.-", linewidth=2, markersize=6,
                label="Actual Future")
        ax.plot(pred_x, pred_y, "r.--", linewidth=2, markersize=6,
                label="Predicted Future")

        # Connect observed to future
        ax.plot([obs_x[-1], fut_x[0]], [obs_y[-1], fut_y[0]],
                "g-", alpha=0.3)
        ax.plot([obs_x[-1], pred_x[0]], [obs_y[-1], pred_y[0]],
                "r--", alpha=0.3)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Sample {i+1}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Linear Transformer — Sample Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sample_predictions.png", dpi=150)
    print(f"  Saved: sample_predictions.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Linear Transformer — Training Pipeline")
    print("=" * 60)
    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"Device: {TRAIN_CONFIG['device']}")
    print()

    # ── Step 1: Load data ──
    train_data, val_data, test_data, norm_stats = load_data()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        TRAIN_CONFIG["batch_size"],
        TRAIN_CONFIG["device"],
    )

    # ── Step 2: Create model ──
    print(f"\n{'=' * 60}")
    print("Creating Linear Transformer model")
    print(f"{'=' * 60}")

    model = LinearTransformerPredictor(
        input_dim=TRAIN_CONFIG["input_dim"],
        output_dim=TRAIN_CONFIG["output_dim"],
        d_model=TRAIN_CONFIG["d_model"],
        n_heads=TRAIN_CONFIG["n_heads"],
        n_layers=TRAIN_CONFIG["n_layers"],
        d_ff=TRAIN_CONFIG["d_ff"],
        pred_len=TRAIN_CONFIG["pred_len"],
        dropout=TRAIN_CONFIG["dropout"],
    )

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Config: {model.config}")

    # ── Step 3: Train ──
    print(f"\n{'=' * 60}")
    print("Training")
    print(f"{'=' * 60}")

    model, train_losses, val_losses, best_epoch = train_model(
        model, train_loader, val_loader, TRAIN_CONFIG
    )

    # ── Step 4: Evaluate on test set ──
    print(f"\n{'=' * 60}")
    print("Evaluating on Test Set")
    print(f"{'=' * 60}")

    device = TRAIN_CONFIG["device"]
    metrics = compute_metrics(model, test_loader, norm_stats, device)

    print(f"\n  ADE: {metrics['ADE_mean']:.4f} ± {metrics['ADE_std']:.4f} meters")
    print(f"  FDE: {metrics['FDE_mean']:.4f} ± {metrics['FDE_std']:.4f} meters")

    # ── Step 5: Measure inference time ──
    print(f"\n{'=' * 60}")
    print("Measuring Inference Time")
    print(f"{'=' * 60}")

    inference_ms = measure_inference_time(model, device)
    print(f"  Average inference time: {inference_ms:.2f} ms per prediction")
    print(f"  Throughput: {1000/inference_ms:.0f} predictions/second")

    # ── Step 6: Save everything ──
    print(f"\n{'=' * 60}")
    print("Saving Results")
    print(f"{'=' * 60}")

    # Save model weights
    model_path = RESULTS_DIR / "linear_transformer_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
        "train_config": TRAIN_CONFIG,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "inference_ms": inference_ms,
    }, model_path)
    print(f"  Saved: {model_path.name}")

    # Save metrics as JSON (human-readable)
    results = {
        "model": "LinearTransformer",
        "parameters": model.count_parameters(),
        "best_epoch": best_epoch,
        "ADE_mean": metrics["ADE_mean"],
        "ADE_std": metrics["ADE_std"],
        "FDE_mean": metrics["FDE_mean"],
        "FDE_std": metrics["FDE_std"],
        "inference_ms": inference_ms,
        "train_config": TRAIN_CONFIG,
    }
    with open(RESULTS_DIR / "linear_transformer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: linear_transformer_results.json")

    # Save training curves
    plot_training_curves(train_losses, val_losses, best_epoch)

    # Save sample prediction visualizations
    plot_sample_predictions(model, test_loader, norm_stats, device)

    # ── Final Summary ──
    print(f"\n{'=' * 60}")
    print("✅ Training Complete — Final Results")
    print(f"{'=' * 60}")
    print(f"  Model:           Linear Transformer")
    print(f"  Parameters:      {model.count_parameters():,}")
    print(f"  Best Epoch:      {best_epoch}")
    print(f"  ADE:             {metrics['ADE_mean']:.4f} m")
    print(f"  FDE:             {metrics['FDE_mean']:.4f} m")
    print(f"  Inference Time:  {inference_ms:.2f} ms")
    print(f"  All saved to:    {RESULTS_DIR.resolve()}")
