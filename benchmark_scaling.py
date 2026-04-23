import time
import torch
from linear_transformer import LinearTransformerPredictor
from train_transformer import TransformerBaseline
from train_lstm import LSTMBaseline
from physics_linear_transformer import PhysicsLinearTransformerPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"

seq_lengths = [100, 500, 1000, 2000, 4000, 8000, 16000, 32000]
n_runs = 100

N_LAYERS = 3
D_MODEL = 64
N_HEADS = 4
D_FF = 128

lin_model = LinearTransformerPredictor(
    input_dim=6, output_dim=3, d_model=D_MODEL,
    n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
    pred_len=10, dropout=0.1,
).to(device).eval()

std_model = TransformerBaseline(
    input_dim=6, d_model=D_MODEL, nhead=N_HEADS,
    num_layers=N_LAYERS, dim_feedforward=D_FF,
    dropout=0.2, pred_len=10, output_dim=3,
).to(device).eval()

lstm_model = LSTMBaseline(
    input_dim=6, hidden_dim=D_MODEL, num_layers=N_LAYERS,
    pred_len=10, output_dim=3, dropout=0.2,
).to(device).eval()

phys_model = PhysicsLinearTransformerPredictor(
    input_dim=6, output_dim=3, d_model=D_MODEL,
    n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
    pred_len=10, dropout=0.1, dt=0.1,
).to(device).eval()

# Physics model returns (trajectory, v_omega), so wrap it
class PhysicsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        traj, _ = self.model(x)
        return traj

models = {
    "linear": lin_model,
    "standard": std_model,
    "lstm": lstm_model,
    "physics": PhysicsWrapper(phys_model).to(device).eval(),
}

print(f"device: {device}")
print(f"layers: {N_LAYERS}, d_model: {D_MODEL}, heads: {N_HEADS}")

header = f"{'T':>8}"
for name in models:
    header += f" {name + '_ms':>14}"
print(header)
print("-" * (8 + 15 * len(models)))

with torch.no_grad():
    for T in seq_lengths:
        dummy = torch.randn(1, T, 6, device=device)

        times = {}
        for name, model in models.items():
            # warmup
            for _ in range(20):
                model(dummy)

            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(n_runs):
                model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times[name] = (time.time() - start) / n_runs * 1000

        row = f"{T:>8}"
        for name in models:
            row += f" {times[name]:>14.3f}"
        print(row)
