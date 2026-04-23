import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

#config
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Dataset" / "Preprocessed"
RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 256
NUM_WORKERS = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 5
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


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, pred_len=10, output_dim=3, dropout=0.2):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * output_dim),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        out = self.head(h)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def main():
    print("using device:", DEVICE)

    train_loader, val_loader, test_loader, norm_stats = load_data()

    out_mean = norm_stats["output_mean"].to(DEVICE)
    out_std = norm_stats["output_std"].to(DEVICE)

    model = LSTMBaseline().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
                RESULTS_DIR / "best_lstm_model.pt",
            )
            patience_counter = 0
            print("saved best lstm model")
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


if __name__ == "__main__":
    main()
