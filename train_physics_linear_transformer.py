"""
Training Script — Physics-Informed Linear Transformer
======================================================
Similar structure to the regular training script but with key differences:

1. The model returns TWO outputs: (trajectory, v_omega)
2. We use PhysicsInformedLoss (trajectory MSE + velocity + smoothness)
3. We must DENORMALIZE the initial state before the kinematic integrator
   because cos(θ) and sin(θ) only make sense in real-world radians,
   not in normalized space where θ has been shifted and scaled.
4. Loss is computed in REAL-WORLD coordinates (meters, radians)

IMPORTANT COORDINATE SUBTLETY:
-------------------------------
The preprocessed data is normalized: x_norm = (x - mean) / std
The unicycle model uses: x_{t+1} = x_t + v * cos(θ) * dt

If θ is normalized, cos(θ_norm) ≠ cos(θ_real). The cosine of a
shifted+scaled angle is physically meaningless. So we must:
    1. Denormalize the initial state before passing to the integrator
    2. The model predicts v, ω in real-world units
    3. Integration happens in real-world coordinates
    4. Denormalize targets for loss computation in real-world space

This is cleaner than trying to make physics work in normalized space.
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics_linear_transformer import (
    PhysicsLinearTransformerPredictor,
    PhysicsInformedLoss,
)

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Dataset" / "Preprocessed"
RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_CONFIG = {
    # Model architecture
    "input_dim": 6,
    "output_dim": 3,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 3,
    "d_ff": 128,
    "pred_len": 10,
    "dropout": 0.1,
    "dt": 0.1,       # timestep in seconds (10 Hz)

    # Physics loss weights
    "lambda_vel": 0.1,       # velocity bound penalty weight
    "lambda_smooth": 0.05,   # smoothness penalty weight
    "max_v": 3.0,            # max linear velocity (m/s)
    "max_omega": 2.0,        # max angular velocity (rad/s)

    # Training parameters
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 50,
    "patience": 10,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# 1. LOAD DATA
# =============================================================================
def load_data():
    """Load preprocessed tensors and normalization stats."""
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
    """Wrap tensors in DataLoaders."""
    pin = (device == "cuda")

    train_loader = DataLoader(
        TensorDataset(train_data["X"], train_data["Y"]),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=pin,
    )
    val_loader = DataLoader(
        TensorDataset(val_data["X"], val_data["Y"]),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=pin,
    )
    test_loader = DataLoader(
        TensorDataset(test_data["X"], test_data["Y"]),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=pin,
    )

    print(f"\n  Train batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


# =============================================================================
# 2. DENORMALIZATION HELPERS
# =============================================================================
def denormalize_initial_state(X_batch, norm_stats, device):
    """
    Extract and denormalize the initial state from the last observed timestep.

    WHY WE NEED THIS:
    ------------------
    The input X_batch is normalized: x_norm = (x_real - mean) / std
    The last timestep's features [0, 1, 2] are [gt_x, gt_y, gt_yaw].

    The unicycle integrator needs REAL x, y, θ because:
        x_{t+1} = x_t + v * cos(θ_t) * dt

    cos(θ_normalized) ≠ cos(θ_real)!

    Example:
        θ_real = 1.5 rad → cos(1.5) = 0.0707
        If mean=0.0, std=1.8, then θ_norm = 1.5/1.8 = 0.833
        cos(0.833) = 0.672 ← COMPLETELY WRONG!

    So we reverse the normalization: x_real = x_norm * std + mean
    """
    # Input features are: [gt_x, gt_y, gt_yaw, vx, vy, omega]
    # We need features 0, 1, 2 (position and heading)
    in_mean = norm_stats["input_mean"].to(device)  # (6,)
    in_std = norm_stats["input_std"].to(device)     # (6,)

    # Extract last timestep: (batch, 6)
    last_step_norm = X_batch[:, -1, :]

    # Denormalize all features: (batch, 6)
    last_step_real = last_step_norm * in_std + in_mean

    # Return only [x, y, θ]: (batch, 3)
    initial_state = last_step_real[:, :3]

    return initial_state


def denormalize_targets(Y_batch, norm_stats, device):
    """
    Denormalize target trajectories from normalized to real-world coordinates.

    The model's output (from the integrator) is in real-world space.
    The targets in Y_batch are normalized. We need both in the same
    space to compute loss. We denormalize the targets.
    """
    out_mean = norm_stats["output_mean"].to(device)  # (3,)
    out_std = norm_stats["output_std"].to(device)     # (3,)

    # Y_batch shape: (batch, pred_len, 3)
    # Denormalize: real = normalized * std + mean
    Y_real = Y_batch * out_std + out_mean

    return Y_real


# =============================================================================
# 3. TRAINING LOOP
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, norm_stats, device):
    """
    Train one epoch with physics-informed loss.

    Key difference from regular training:
    - We denormalize initial state before passing to model
    - Model returns (trajectory, v_omega) — two outputs
    - Loss uses trajectory, targets, AND v_omega
    - Loss is computed in real-world coordinates
    """
    model.train()
    total_loss = 0.0
    total_components = {"trajectory_mse": 0, "velocity_penalty": 0,
                        "smoothness_penalty": 0}
    n_batches = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        # Denormalize initial state for the integrator
        initial_state = denormalize_initial_state(X_batch, norm_stats, device)

        # Denormalize targets to real-world coordinates
        Y_real = denormalize_targets(Y_batch, norm_stats, device)

        # Forward pass: model predicts (v, ω) and integrates
        # trajectory is in real-world coordinates (because initial_state is real)
        trajectory, v_omega = model(X_batch, initial_state=initial_state)

        # Compute physics-informed loss in real-world space
        loss, loss_dict = criterion(trajectory, Y_real, v_omega)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: prevent exploding gradients
        # The kinematic integration can amplify gradients (each step
        # depends on the previous), so clipping is important here.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        for k in total_components:
            total_components[k] += loss_dict[k]
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in total_components.items()}

    return avg_loss, avg_components


@torch.no_grad()
def evaluate(model, loader, criterion, norm_stats, device):
    """Evaluate on validation/test data with physics-informed loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        initial_state = denormalize_initial_state(X_batch, norm_stats, device)
        Y_real = denormalize_targets(Y_batch, norm_stats, device)

        trajectory, v_omega = model(X_batch, initial_state=initial_state)
        loss, _ = criterion(trajectory, Y_real, v_omega)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train_model(model, train_loader, val_loader, norm_stats, config):
    """Full training loop with early stopping."""
    device = config["device"]
    print(f"\nTraining on: {device}")

    model = model.to(device)

    # Physics-informed loss function
    criterion = PhysicsInformedLoss(
        lambda_vel=config["lambda_vel"],
        lambda_smooth=config["lambda_smooth"],
        max_v=config["max_v"],
        max_omega=config["max_omega"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    print(f"\nStarting training for up to {config['epochs']} epochs...")
    print(f"  Physics loss weights: λ_vel={config['lambda_vel']}, "
          f"λ_smooth={config['lambda_smooth']}")
    print(f"{'─' * 80}")
    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Traj MSE':>10} "
          f"{'Vel Pen':>10} {'Smooth':>10} {'LR':>10} {'Time':>7} {'':>6}")
    print(f"{'─' * 80}")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        # Train
        train_loss, components = train_one_epoch(
            model, train_loader, optimizer, criterion, norm_stats, device
        )

        # Validate
        val_loss = evaluate(
            model, val_loader, criterion, norm_stats, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            status = "★ best"
        else:
            patience_counter += 1

        print(f"{epoch:>6} {train_loss:>10.4f} {val_loss:>10.4f} "
              f"{components['trajectory_mse']:>10.4f} "
              f"{components['velocity_penalty']:>10.6f} "
              f"{components['smoothness_penalty']:>10.6f} "
              f"{current_lr:>10.6f} {elapsed:>6.1f}s {status:>6}")

        scheduler.step(val_loss)

        if patience_counter >= config["patience"]:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"{'─' * 80}")
    print(f"  Best epoch: {best_epoch} with val loss: {best_val_loss:.4f}")

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, best_epoch


# =============================================================================
# 4. EVALUATION METRICS (ADE / FDE in real-world meters)
# =============================================================================
@torch.no_grad()
def compute_metrics(model, loader, norm_stats, device):
    """
    Compute ADE and FDE in real-world meters.

    Since the physics model already outputs in real-world coordinates,
    we just need to denormalize the targets to match.
    """
    model.eval()

    all_ade = []
    all_fde = []

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        initial_state = denormalize_initial_state(X_batch, norm_stats, device)
        Y_real = denormalize_targets(Y_batch, norm_stats, device)

        trajectory, _ = model(X_batch, initial_state=initial_state)

        # Euclidean distance on (x, y) only
        dx = trajectory[:, :, 0] - Y_real[:, :, 0]
        dy = trajectory[:, :, 1] - Y_real[:, :, 1]
        distances = torch.sqrt(dx**2 + dy**2)

        ade = distances.mean(dim=1)
        fde = distances[:, -1]

        all_ade.append(ade)
        all_fde.append(fde)

    all_ade = torch.cat(all_ade).cpu().numpy()
    all_fde = torch.cat(all_fde).cpu().numpy()

    return {
        "ADE_mean": float(np.mean(all_ade)),
        "ADE_std": float(np.std(all_ade)),
        "FDE_mean": float(np.mean(all_fde)),
        "FDE_std": float(np.std(all_fde)),
    }


# =============================================================================
# 5. INFERENCE TIME MEASUREMENT
# =============================================================================
@torch.no_grad()
def measure_inference_time(model, norm_stats, device, n_runs=100):
    """Measure average inference time per prediction."""
    model.eval()
    model = model.to(device)

    dummy_input = torch.randn(1, 20, 6).to(device)
    dummy_state = torch.randn(1, 3).to(device)

    # Warmup
    for _ in range(10):
        _ = model(dummy_input, initial_state=dummy_state)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_runs):
        _ = model(dummy_input, initial_state=dummy_state)
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    return (end - start) / n_runs * 1000


# =============================================================================
# 6. PLOTTING
# =============================================================================
def plot_training_curves(train_losses, val_losses, best_epoch):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", linewidth=1.5, label="Train Loss")
    ax.plot(epochs, val_losses, "r-", linewidth=1.5, label="Val Loss")
    ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
               label=f"Best Epoch ({best_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Physics-Informed Loss")
    ax.set_title("Physics-Informed Linear Transformer — Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "physics_training_curves.png", dpi=150)
    print(f"  Saved: physics_training_curves.png")


@torch.no_grad()
def plot_sample_predictions(model, test_loader, norm_stats, device, n_samples=4):
    """Visualize actual vs predicted trajectories."""
    model.eval()

    in_mean = norm_stats["input_mean"].to(device)
    in_std = norm_stats["input_std"].to(device)

    X_batch, Y_batch = next(iter(test_loader))
    X_batch = X_batch[:n_samples].to(device)
    Y_batch = Y_batch[:n_samples].to(device)

    initial_state = denormalize_initial_state(X_batch, norm_stats, device)
    Y_real = denormalize_targets(Y_batch, norm_stats, device)

    trajectory, v_omega = model(X_batch, initial_state=initial_state)

    # Denormalize input for plotting observed trajectory
    input_real = (X_batch * in_std + in_mean).cpu().numpy()
    pred_np = trajectory.cpu().numpy()
    target_np = Y_real.cpu().numpy()
    v_omega_np = v_omega.cpu().numpy()

    # Plot trajectories
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for i in range(n_samples):
        ax = axes_flat[i]

        obs_x = input_real[i, :, 0]
        obs_y = input_real[i, :, 1]
        fut_x = target_np[i, :, 0]
        fut_y = target_np[i, :, 1]
        pred_x = pred_np[i, :, 0]
        pred_y = pred_np[i, :, 1]

        ax.plot(obs_x, obs_y, "b.-", linewidth=2, markersize=4,
                label="Observed", alpha=0.7)
        ax.plot(fut_x, fut_y, "g.-", linewidth=2, markersize=6,
                label="Actual Future")
        ax.plot(pred_x, pred_y, "r.--", linewidth=2, markersize=6,
                label="Predicted (Physics)")

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

    fig.suptitle("Physics-Informed Linear Transformer — Sample Predictions",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "physics_sample_predictions.png", dpi=150)
    print(f"  Saved: physics_sample_predictions.png")

    # Also plot predicted velocities for one sample
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    t_steps = np.arange(TRAIN_CONFIG["pred_len"]) * TRAIN_CONFIG["dt"]

    ax1.plot(t_steps, v_omega_np[0, :, 0], "b.-", label="Predicted v")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Linear Velocity (m/s)")
    ax1.set_title("Predicted Linear Velocity")
    ax1.axhline(y=TRAIN_CONFIG["max_v"], color="r", linestyle="--",
                alpha=0.5, label=f"Max ({TRAIN_CONFIG['max_v']} m/s)")
    ax1.axhline(y=-TRAIN_CONFIG["max_v"], color="r", linestyle="--", alpha=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_steps, v_omega_np[0, :, 1], "g.-", label="Predicted ω")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular Velocity (rad/s)")
    ax2.set_title("Predicted Angular Velocity")
    ax2.axhline(y=TRAIN_CONFIG["max_omega"], color="r", linestyle="--",
                alpha=0.5, label=f"Max ({TRAIN_CONFIG['max_omega']} rad/s)")
    ax2.axhline(y=-TRAIN_CONFIG["max_omega"], color="r", linestyle="--",
                alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.suptitle("Physics Model — Predicted Velocity Commands", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "physics_velocity_predictions.png", dpi=150)
    print(f"  Saved: physics_velocity_predictions.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Physics-Informed Linear Transformer — Training Pipeline")
    print("=" * 60)
    print(f"Results: {RESULTS_DIR}")
    print(f"Device:  {TRAIN_CONFIG['device']}")
    print()

    # ── Load data ──
    train_data, val_data, test_data, norm_stats = load_data()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        TRAIN_CONFIG["batch_size"],
        TRAIN_CONFIG["device"],
    )

    # ── Create model ──
    print(f"\n{'=' * 60}")
    print("Creating Physics-Informed Linear Transformer")
    print(f"{'=' * 60}")

    model = PhysicsLinearTransformerPredictor(
        input_dim=TRAIN_CONFIG["input_dim"],
        output_dim=TRAIN_CONFIG["output_dim"],
        d_model=TRAIN_CONFIG["d_model"],
        n_heads=TRAIN_CONFIG["n_heads"],
        n_layers=TRAIN_CONFIG["n_layers"],
        d_ff=TRAIN_CONFIG["d_ff"],
        pred_len=TRAIN_CONFIG["pred_len"],
        dropout=TRAIN_CONFIG["dropout"],
        dt=TRAIN_CONFIG["dt"],
    )

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Config: {model.config}")

    # ── Train ──
    print(f"\n{'=' * 60}")
    print("Training")
    print(f"{'=' * 60}")

    model, train_losses, val_losses, best_epoch = train_model(
        model, train_loader, val_loader, norm_stats, TRAIN_CONFIG
    )

    # ── Evaluate ──
    print(f"\n{'=' * 60}")
    print("Evaluating on Test Set")
    print(f"{'=' * 60}")

    device = TRAIN_CONFIG["device"]
    metrics = compute_metrics(model, test_loader, norm_stats, device)

    print(f"\n  ADE: {metrics['ADE_mean']:.4f} ± {metrics['ADE_std']:.4f} meters")
    print(f"  FDE: {metrics['FDE_mean']:.4f} ± {metrics['FDE_std']:.4f} meters")

    # ── Inference time ──
    print(f"\n{'=' * 60}")
    print("Measuring Inference Time")
    print(f"{'=' * 60}")

    inference_ms = measure_inference_time(model, norm_stats, device)
    print(f"  Average inference time: {inference_ms:.2f} ms per prediction")
    print(f"  Throughput: {1000/inference_ms:.0f} predictions/second")

    # ── Save everything ──
    print(f"\n{'=' * 60}")
    print("Saving Results")
    print(f"{'=' * 60}")

    # Save model
    model_path = RESULTS_DIR / "physics_linear_transformer_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.config,
        "train_config": TRAIN_CONFIG,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "inference_ms": inference_ms,
    }, model_path)
    print(f"  Saved: {model_path.name}")

    # Save metrics JSON
    results = {
        "model": "PhysicsLinearTransformer",
        "parameters": model.count_parameters(),
        "best_epoch": best_epoch,
        "ADE_mean": metrics["ADE_mean"],
        "ADE_std": metrics["ADE_std"],
        "FDE_mean": metrics["FDE_mean"],
        "FDE_std": metrics["FDE_std"],
        "inference_ms": inference_ms,
        "train_config": TRAIN_CONFIG,
    }
    with open(RESULTS_DIR / "physics_linear_transformer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: physics_linear_transformer_results.json")

    # Plots
    plot_training_curves(train_losses, val_losses, best_epoch)
    plot_sample_predictions(model, test_loader, norm_stats, device)

    # ── Compare with regular Linear Transformer ──
    print(f"\n{'=' * 60}")
    print("Comparison with Regular Linear Transformer")
    print(f"{'=' * 60}")

    regular_results_path = RESULTS_DIR / "linear_transformer_results.json"
    if regular_results_path.exists():
        with open(regular_results_path) as f:
            regular = json.load(f)

        print(f"\n  {'Metric':<25} {'Regular':>12} {'Physics':>12} {'Diff':>12}")
        print(f"  {'─' * 61}")
        print(f"  {'ADE (m)':<25} {regular['ADE_mean']:>12.4f} "
              f"{metrics['ADE_mean']:>12.4f} "
              f"{metrics['ADE_mean'] - regular['ADE_mean']:>+12.4f}")
        print(f"  {'FDE (m)':<25} {regular['FDE_mean']:>12.4f} "
              f"{metrics['FDE_mean']:>12.4f} "
              f"{metrics['FDE_mean'] - regular['FDE_mean']:>+12.4f}")
        print(f"  {'Inference (ms)':<25} {regular['inference_ms']:>12.2f} "
              f"{inference_ms:>12.2f} "
              f"{inference_ms - regular['inference_ms']:>+12.2f}")
        print(f"  {'Parameters':<25} {regular['parameters']:>12,} "
              f"{model.count_parameters():>12,} "
              f"{model.count_parameters() - regular['parameters']:>+12,}")
    else:
        print("  Regular model results not found. Run train_linear_transformer.py first.")

    # ── Final Summary ──
    print(f"\n{'=' * 60}")
    print("✅ Training Complete — Final Results")
    print(f"{'=' * 60}")
    print(f"  Model:           Physics-Informed Linear Transformer")
    print(f"  Parameters:      {model.count_parameters():,}")
    print(f"  Best Epoch:      {best_epoch}")
    print(f"  ADE:             {metrics['ADE_mean']:.4f} m")
    print(f"  FDE:             {metrics['FDE_mean']:.4f} m")
    print(f"  Inference Time:  {inference_ms:.2f} ms")
    print(f"  All saved to:    {RESULTS_DIR.resolve()}")
