import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#config
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Dataset" / "Preprocessed"
RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 256
NUM_WORKERS = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    train_data = torch.load(DATA_DIR / "train.pt", weights_only=True)
    val_data = torch.load(DATA_DIR / "val.pt", weights_only=True)
    test_data = torch.load(DATA_DIR / "test.pt", weights_only=True)
    norm_stats = torch.load(DATA_DIR / "norm_stats.pt", weights_only=True)

    print("X_train:", train_data["X"].shape)
    print("Y_train:", train_data["Y"].shape)
    print("X_val:", val_data["X"].shape)
    print("Y_val:", val_data["Y"].shape)
    print("X_test:", test_data["X"].shape)
    print("Y_test:", test_data["Y"].shape)

    train_ds = TensorDataset(train_data["X"], train_data["Y"])
    val_ds = TensorDataset(val_data["X"], val_data["Y"])
    test_ds = TensorDataset(test_data["X"], test_data["Y"])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    return train_loader, val_loader, test_loader, norm_stats


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=40000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        input_dim=6,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        pred_len=10,
        output_dim=3,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=40000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * output_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        x = x[:, -1, :]

        out = self.head(x)
        out = out.view(-1, self.pred_len, self.output_dim)
        return out


def compute_metrics(preds_denorm, targets_denorm):
    diff = preds_denorm - targets_denorm
    mse = torch.mean(diff ** 2).item()

    displacement = torch.linalg.norm(diff[..., :2], dim=-1)
    ade = displacement.mean().item()
    fde = displacement[:, -1].mean().item()

    return mse, ade, fde


def run_epoch(loader, model, optimizer, criterion, out_mean, out_std, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    total_mse = 0.0
    total_ade = 0.0
    total_fde = 0.0

    with torch.set_grad_enabled(train):
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            Y_batch = Y_batch.to(DEVICE, non_blocking=True)

            preds_norm = model(X_batch)
            loss = criterion(preds_norm, Y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds_denorm = preds_norm * out_std + out_mean
            targets_denorm = Y_batch * out_std + out_mean
            mse, ade, fde = compute_metrics(preds_denorm, targets_denorm)

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_mse += mse * batch_size
            total_ade += ade * batch_size
            total_fde += fde * batch_size
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "mse": total_mse / total_samples,
        "ade": total_ade / total_samples,
        "fde": total_fde / total_samples,
    }


@torch.no_grad()
def measure_inference_time(model, input_shape=(1, 20, 6), n_runs=100):
    model.eval()
    dummy = torch.randn(*input_shape).to(DEVICE)

    for _ in range(10):
        _ = model(dummy)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_runs):
        _ = model(dummy)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    return (end - start) / n_runs * 1000


@torch.no_grad()
def plot_sample_predictions(model, test_loader, norm_stats, n_samples=4):
    model.eval()
    out_mean = norm_stats["output_mean"].to(DEVICE)
    out_std = norm_stats["output_std"].to(DEVICE)
    in_mean = norm_stats["input_mean"].to(DEVICE)
    in_std = norm_stats["input_std"].to(DEVICE)

    # Collect predictions over the ENTIRE test set
    all_pred, all_target, all_input = [], [], []
    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)
        pred = model(X_batch)

        all_pred.append((pred * out_std + out_mean).cpu())
        all_target.append((Y_batch * out_std + out_mean).cpu())
        all_input.append((X_batch * in_std + in_mean).cpu())

    all_pred = torch.cat(all_pred)
    all_target = torch.cat(all_target)
    all_input = torch.cat(all_input)

    # Pick best samples from entire test set
    dx = all_pred[:, :, 0] - all_target[:, :, 0]
    dy = all_pred[:, :, 1] - all_target[:, :, 1]
    per_sample_ade = torch.sqrt(dx**2 + dy**2).mean(dim=1)
    _, best_idx = per_sample_ade.sort()
    idx = best_idx[:n_samples]

    pred_np = all_pred[idx].numpy()
    target_np = all_target[idx].numpy()
    input_np = all_input[idx].numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i in range(n_samples):
        ax = axes[i]
        obs_x = input_np[i, :, 0]
        obs_y = input_np[i, :, 1]
        fut_x = target_np[i, :, 0]
        fut_y = target_np[i, :, 1]
        pred_x = pred_np[i, :, 0]
        pred_y = pred_np[i, :, 1]

        ax.plot(obs_x, obs_y, "b.-", linewidth=2, markersize=4,
                label="Observed", alpha=0.7)
        ax.plot(fut_x, fut_y, "g.-", linewidth=2, markersize=6,
                label="Actual Future")
        ax.plot(pred_x, pred_y, "r.--", linewidth=2, markersize=6,
                label="Predicted")

        ax.plot([obs_x[-1], fut_x[0]], [obs_y[-1], fut_y[0]], "g-", alpha=0.3)
        ax.plot([obs_x[-1], pred_x[0]], [obs_y[-1], pred_y[0]], "r--", alpha=0.3)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Sample {i+1}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Standard Transformer — Sample Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "transformer_sample_predictions.png", dpi=150)
    print(f"Saved: transformer_sample_predictions.png")



def main():
    print("using device:", DEVICE)

    train_loader, val_loader, test_loader, norm_stats = load_data()

    out_mean = norm_stats["output_mean"].to(DEVICE)
    out_std = norm_stats["output_std"].to(DEVICE)

    model = TransformerBaseline().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_ade = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_metrics = run_epoch(train_loader, model, optimizer, criterion, out_mean, out_std, train=True)
        val_metrics = run_epoch(val_loader, model, None, criterion, out_mean, out_std, train=False)

        scheduler.step(val_metrics["ade"])

        print(
            f"epoch {epoch+1}/{NUM_EPOCHS} "
            f"train_loss={train_metrics['loss']:.6f} train_ade={train_metrics['ade']:.6f} train_fde={train_metrics['fde']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_ade={val_metrics['ade']:.6f} val_fde={val_metrics['fde']:.6f}"
        )

        if val_metrics["ade"] < best_val_ade:
            best_val_ade = val_metrics["ade"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "output_mean": norm_stats["output_mean"],
                    "output_std": norm_stats["output_std"],
                    "input_mean": norm_stats["input_mean"],
                    "input_std": norm_stats["input_std"],
                    "best_val_ade": best_val_ade,
                },
                RESULTS_DIR / "best_transformer_model.pt",
            )
            patience_counter = 0
            print("saved best transformer model")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("early stopping triggered")
            break

    model.load_state_dict(best_state)
    test_metrics = run_epoch(test_loader, model, None, criterion, out_mean, out_std, train=False)

    print(
        f"test_loss={test_metrics['loss']:.6f} "
        f"test_mse={test_metrics['mse']:.6f} "
        f"test_ade={test_metrics['ade']:.6f} "
        f"test_fde={test_metrics['fde']:.6f}"
    )

    inference_ms = measure_inference_time(model)
    print(f"inference_ms={inference_ms:.4f}")

    plot_sample_predictions(model, test_loader, norm_stats)


if __name__ == "__main__":
    main()
