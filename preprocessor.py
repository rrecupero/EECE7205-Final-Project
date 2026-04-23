"""
NCLT Preprocessing Pipeline — Step 3
======================================
Takes processed CSVs from Step 2 (Outputdata/) and produces
PyTorch-ready sliding window datasets for trajectory prediction.

Pipeline:
    1. Load processed CSVs
    2. Downsample from ~100 Hz to target Hz
    3. Remove outliers in velocity
    4. Normalize features
    5. Create sliding windows (observe T → predict N)
    6. Split by session into train/val/test
    7. Save as PyTorch tensors + normalization stats

Input:  Dataset/Outputdata/processed_<date>.csv
Output: Dataset/Preprocessed/
            ├── train.pt
            ├── val.pt
            ├── test.pt
            ├── norm_stats.pt    (mean/std for denormalization later)
            └── config.json      (all preprocessing hyperparameters)
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "Dataset" / "Outputdata"
PREPROCESS_DIR = PROJECT_ROOT / "Dataset" / "Preprocessed"
PREPROCESS_DIR.mkdir(exist_ok=True)

# ── Preprocessing Hyperparameters ──
CONFIG = {
    # Which features to use as INPUT to the model
    # These are what the model "observes" from the past
    "input_features": ["gt_x", "gt_y", "gt_yaw", "vx", "vy", "omega"],

    # Which features to PREDICT (the output / labels)
    # These are what the model tries to forecast into the future
    "output_features": ["gt_x", "gt_y", "gt_yaw"],

    # Downsampling: reduce from ~100 Hz to this rate
    # 10 Hz = one sample every 0.1 seconds
    "target_hz": 10,

    # Sliding window sizes (in timesteps AFTER downsampling)
    # observe 20 steps at 10 Hz = observe 2.0 seconds of history
    # predict 10 steps at 10 Hz = predict 1.0 second into the future
    "observe_len": 20,
    "predict_len": 10,

    # How many steps to slide the window forward each time
    # stride=1 means maximum overlap (most training samples)
    # stride=5 means less overlap (fewer but more independent samples)
    "window_stride": 5,

    # Outlier removal: clip velocities beyond these thresholds
    # The Segway moves at ~1-2 m/s; anything above 5 m/s is suspicious
    "max_linear_vel": 5.0,   # m/s
    "max_angular_vel": 3.0,  # rad/s

    # Train/val/test split — which sessions go where
    # Split by FULL SESSIONS to avoid data leakage
    "train_sessions": ["2012-01-08", "2012-02-18", "2013-02-23"],
    "val_sessions": ["2012-01-22"],
    "test_sessions": ["2013-04-05"],
}


# =============================================================================
# STEP 1: LOAD PROCESSED DATA
# =============================================================================
def load_processed_sessions() -> dict:
    """
    Load the cleaned CSVs from Step 2.

    Why not load the raw data again?
    ---------------------------------
    Step 2 already did the heavy lifting: extracting tars, aligning
    timestamps between ground truth and odometry, computing velocities,
    and dropping bad rows. The processed CSVs are the clean output of
    all that work. Loading them directly saves time and ensures
    consistency — every script downstream uses the same cleaned data.
    """
    print("=" * 60)
    print("STEP 1: Loading processed data from Step 2")
    print("=" * 60)

    all_sessions = (
        CONFIG["train_sessions"]
        + CONFIG["val_sessions"]
        + CONFIG["test_sessions"]
    )
    # Remove duplicates while preserving order
    all_sessions = list(dict.fromkeys(all_sessions))

    sessions = {}
    for session in all_sessions:
        path = OUTPUT_DIR / f"processed_{session}.csv"
        if not path.exists():
            print(f"  [WARN] {path.name} not found — skipping")
            continue

        df = pd.read_csv(path)
        print(f"  Loaded {session}: {len(df):>10,} rows")
        sessions[session] = df

    print()
    return sessions


# =============================================================================
# STEP 2: DOWNSAMPLE
# =============================================================================
def downsample(df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    """
    Reduce the sampling rate from ~100 Hz to target_hz.

    Why downsample?
    ---------------
    At 100 Hz, consecutive rows are only 10 ms apart. The robot barely
    moves in 10 ms — maybe 0.01 meters. This means:

    1. Consecutive rows are almost identical, which wastes computation.
       A neural network processing 100 nearly-identical rows learns
       nothing more than processing 10 meaningfully-different rows.

    2. Your sliding window would need to be huge. To observe 2 seconds
       of history at 100 Hz, you'd need a window of 200 timesteps.
       At 10 Hz, that's only 20 timesteps — much more manageable for
       attention mechanisms (especially the O(T²) standard Transformer).

    3. The finite-difference velocities are noisy at 100 Hz because
       position changes between consecutive rows are tiny compared to
       measurement noise. At 10 Hz, each step has ~10x more displacement,
       giving a much better signal-to-noise ratio.

    How we downsample:
    ------------------
    We compute the original sampling rate from the data, then keep
    every Nth row where N = original_rate / target_rate. For example,
    if the data is at 100 Hz and we want 10 Hz, we keep every 10th row.

    We DON'T average or interpolate — we just subsample. This is simpler
    and preserves the exact ground truth values. Averaging would smooth
    out the trajectory, which could hide real dynamics like sharp turns.
    """
    # Estimate the original sampling rate from the data
    dt_median = np.median(np.diff(df["time_s"].values))
    original_hz = 1.0 / dt_median

    # How many rows to skip between kept rows
    skip = max(1, int(round(original_hz / target_hz)))

    df_down = df.iloc[::skip].reset_index(drop=True)

    # Verify the new rate
    new_dt = np.median(np.diff(df_down["time_s"].values))
    actual_hz = 1.0 / new_dt

    print(f"    Downsample: {original_hz:.0f} Hz → {actual_hz:.1f} Hz "
          f"(keep every {skip}th row) | "
          f"{len(df):,} → {len(df_down):,} rows")
    return df_down


# =============================================================================
# STEP 3: REMOVE OUTLIERS
# =============================================================================
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unrealistic velocity values.

    Why remove outliers?
    --------------------
    Finite-difference velocity computation (v = Δx/Δt) amplifies noise.
    If there's a timestamp glitch (e.g., two rows with nearly the same
    timestamp but different positions), the computed velocity can be
    absurdly large — like 100 m/s for a robot that walks at 1.5 m/s.

    These outliers are rare but devastating for training:
    - They create huge loss spikes that destabilize gradient descent
    - Normalization gets skewed (mean/std become unreliable)
    - The model might learn to predict extreme velocities

    We clip velocities to physically reasonable bounds. The Segway
    robot's max speed is ~3-4 m/s, so anything above 5 m/s is
    definitely sensor noise, not real motion.

    We remove the entire ROW rather than just clipping the value,
    because if the velocity is wrong, the position at that timestep
    is likely problematic too (temporal discontinuity).
    """
    max_v = CONFIG["max_linear_vel"]
    max_w = CONFIG["max_angular_vel"]

    rows_before = len(df)

    # Create a mask: True for rows we want to KEEP
    speed = np.sqrt(df["vx"]**2 + df["vy"]**2)
    keep_mask = (speed <= max_v) & (df["omega"].abs() <= max_w)

    df_clean = df[keep_mask].reset_index(drop=True)
    removed = rows_before - len(df_clean)

    if removed > 0:
        print(f"    Outliers:   removed {removed} rows "
              f"({removed/rows_before*100:.2f}%) "
              f"with speed > {max_v} m/s or ω > {max_w} rad/s")
    else:
        print(f"    Outliers:   none found (all velocities within bounds)")

    return df_clean


# =============================================================================
# STEP 4: NORMALIZE
# =============================================================================
def compute_norm_stats(sessions: dict) -> dict:
    """
    Compute normalization statistics (mean and std) from TRAINING data only.

    Why normalize?
    --------------
    Neural networks learn by adjusting weights using gradients. If one
    feature (like x position) ranges from -500 to +500 and another
    (like angular velocity) ranges from -1 to +1, the gradients for
    the large feature dominate training. The network essentially ignores
    the small features because their gradients are tiny in comparison.

    Normalization rescales all features to roughly the same range
    (zero mean, unit standard deviation). After normalization, x and ω
    both range from roughly -3 to +3, so the network treats them equally.

    Why compute stats from TRAINING data only?
    -------------------------------------------
    This is a critical ML principle. If you compute mean/std using the
    test data too, you're "peeking" at the test set — the model indirectly
    learns information about data it's supposed to never have seen. This
    is called DATA LEAKAGE and it makes your reported metrics artificially
    better than real-world performance.

    In deployment, you wouldn't have access to future data to compute
    statistics. You'd use stats from your training data to normalize
    new incoming data. We simulate this by computing stats only from
    training sessions.

    Returns: dict with 'mean' and 'std' arrays for input and output features.
    """
    print("\n  Computing normalization stats from training data only...")

    # Stack all training session data
    train_dfs = []
    for session_name in CONFIG["train_sessions"]:
        if session_name in sessions:
            train_dfs.append(sessions[session_name])

    train_data = pd.concat(train_dfs, ignore_index=True)

    # Compute stats for input features
    input_cols = CONFIG["input_features"]
    input_mean = train_data[input_cols].mean().values.astype(np.float32)
    input_std = train_data[input_cols].std().values.astype(np.float32)

    # Avoid division by zero (if a feature is constant, std=0)
    input_std = np.where(input_std < 1e-8, 1.0, input_std)

    # Compute stats for output features
    output_cols = CONFIG["output_features"]
    output_mean = train_data[output_cols].mean().values.astype(np.float32)
    output_std = train_data[output_cols].std().values.astype(np.float32)
    output_std = np.where(output_std < 1e-8, 1.0, output_std)

    stats = {
        "input_mean": input_mean,
        "input_std": input_std,
        "output_mean": output_mean,
        "output_std": output_std,
        "input_features": input_cols,
        "output_features": output_cols,
    }

    print(f"    Input features:  {input_cols}")
    print(f"    Input mean:      {np.array2string(input_mean, precision=3)}")
    print(f"    Input std:       {np.array2string(input_std, precision=3)}")
    print(f"    Output features: {output_cols}")
    print(f"    Output mean:     {np.array2string(output_mean, precision=3)}")
    print(f"    Output std:      {np.array2string(output_std, precision=3)}")

    return stats


def normalize_df(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Apply z-score normalization: x_normalized = (x - mean) / std

    After this, each feature has approximately mean=0 and std=1.
    We normalize input and output features using their respective stats.
    """
    df = df.copy()

    # Normalize input features
    for i, col in enumerate(CONFIG["input_features"]):
        df[col] = (df[col] - stats["input_mean"][i]) / stats["input_std"][i]

    # Normalize output features (if they aren't already normalized as input)
    for i, col in enumerate(CONFIG["output_features"]):
        if col not in CONFIG["input_features"]:
            df[col] = (df[col] - stats["output_mean"][i]) / stats["output_std"][i]

    return df


# =============================================================================
# STEP 5: CREATE SLIDING WINDOWS
# =============================================================================
def create_windows(df: pd.DataFrame) -> tuple:
    """
    Slice the trajectory into overlapping input-output window pairs.

    What is a sliding window?
    -------------------------
    Imagine a trajectory with 1000 timesteps. We take a "window" of
    size (observe_len + predict_len) = 30 and slide it across:

    Window 1: timesteps  0-19 (input)  → timesteps 20-29 (target)
    Window 2: timesteps  5-24 (input)  → timesteps 25-34 (target)
    Window 3: timesteps 10-29 (input)  → timesteps 30-39 (target)
    ...

    Each window becomes one training sample. The "stride" controls how
    far the window moves each step:
    - stride=1: maximum overlap, maximum samples (but highly correlated)
    - stride=5: less overlap, fewer but more independent samples

    Why overlapping?
    ----------------
    If we used non-overlapping windows (stride=30), a 1000-step trajectory
    gives only 33 samples. With stride=5, we get ~194 samples from the
    same data. More training samples = better model generalization.

    The overlap creates correlated samples (nearby windows share most of
    their data), but this is standard practice and works well in practice
    as long as your train/test split is at the SESSION level (which ours is).

    What gets returned?
    -------------------
    X: array of shape (num_windows, observe_len, num_input_features)
       Each X[i] is one observation sequence — 20 timesteps × 6 features

    Y: array of shape (num_windows, predict_len, num_output_features)
       Each Y[i] is the corresponding future trajectory — 10 timesteps × 3 features
    """
    obs_len = CONFIG["observe_len"]
    pred_len = CONFIG["predict_len"]
    stride = CONFIG["window_stride"]
    total_len = obs_len + pred_len

    input_cols = CONFIG["input_features"]
    output_cols = CONFIG["output_features"]

    # Extract numpy arrays for speed (DataFrame indexing is slow in loops)
    input_data = df[input_cols].values.astype(np.float32)
    output_data = df[output_cols].values.astype(np.float32)

    n_rows = len(df)
    if n_rows < total_len:
        print(f"    [WARN] Only {n_rows} rows, need at least {total_len} — skipping")
        return np.array([]), np.array([])

    # Calculate window start indices
    starts = np.arange(0, n_rows - total_len + 1, stride)

    # Pre-allocate arrays (much faster than appending in a loop)
    X = np.zeros((len(starts), obs_len, len(input_cols)), dtype=np.float32)
    Y = np.zeros((len(starts), pred_len, len(output_cols)), dtype=np.float32)

    for i, start in enumerate(starts):
        X[i] = input_data[start : start + obs_len]
        Y[i] = output_data[start + obs_len : start + total_len]

    return X, Y


# =============================================================================
# STEP 6: FULL PREPROCESSING PIPELINE
# =============================================================================
def preprocess_session(df: pd.DataFrame, stats: dict, session_name: str) -> tuple:
    """
    Run the complete preprocessing pipeline on one session:
    downsample → remove outliers → normalize → create windows.
    """
    print(f"\n  Processing {session_name}...")

    # Step 2: Downsample
    df = downsample(df, CONFIG["target_hz"])

    # Step 3: Remove outliers
    df = remove_outliers(df)

    # Step 4: Normalize
    df = normalize_df(df, stats)

    # Step 5: Create sliding windows
    X, Y = create_windows(df)

    if len(X) > 0:
        print(f"    Windows:    {len(X):,} samples | "
              f"X shape: {X.shape} | Y shape: {Y.shape}")
    return X, Y


def build_split(sessions: dict, session_names: list,
                stats: dict, split_name: str) -> tuple:
    """
    Process all sessions for a given split (train/val/test) and
    concatenate their windows.
    """
    all_X, all_Y = [], []

    for name in session_names:
        if name not in sessions:
            print(f"  [WARN] Session {name} not available for {split_name}")
            continue
        X, Y = preprocess_session(sessions[name], stats, name)
        if len(X) > 0:
            all_X.append(X)
            all_Y.append(Y)

    if not all_X:
        return np.array([]), np.array([])

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    return X, Y


# =============================================================================
# STEP 7: SAVE EVERYTHING
# =============================================================================
def save_dataset(X: np.ndarray, Y: np.ndarray, split_name: str):
    """
    Save as a PyTorch .pt file.

    Why .pt files?
    --------------
    PyTorch's native tensor format loads directly into GPU memory
    without any parsing or conversion. When you start training,
    loading a .pt file takes milliseconds vs seconds for CSV parsing.
    It also preserves the exact dtype (float32) and shape.
    """
    path = PREPROCESS_DIR / f"{split_name}.pt"
    torch.save({
        "X": torch.from_numpy(X),
        "Y": torch.from_numpy(Y),
    }, path)
    print(f"  Saved {split_name}.pt: X={X.shape}, Y={Y.shape}, "
          f"size={path.stat().st_size / 1e6:.1f} MB")


def save_norm_stats(stats: dict):
    """
    Save normalization statistics.

    Why save these?
    ---------------
    After your model makes a prediction, the output is in NORMALIZED
    space (values near 0 with std ~1). To convert back to real-world
    meters and radians, you need to "denormalize":

        real_value = normalized_value * std + mean

    So you need the exact same mean/std that were used during training.
    These stats bridge the gap between model output and real-world units.
    They're also needed to normalize any new data during inference.
    """
    path = PREPROCESS_DIR / "norm_stats.pt"
    torch.save({
        "input_mean": torch.from_numpy(stats["input_mean"]),
        "input_std": torch.from_numpy(stats["input_std"]),
        "output_mean": torch.from_numpy(stats["output_mean"]),
        "output_std": torch.from_numpy(stats["output_std"]),
        "input_features": stats["input_features"],
        "output_features": stats["output_features"],
    }, path)
    print(f"  Saved norm_stats.pt")


def save_config():
    """
    Save all preprocessing hyperparameters as JSON.

    Why save the config?
    --------------------
    Reproducibility. If you come back in 3 weeks and want to know
    "what window size did I use?" or "which sessions were in the
    training set?", this file has the answer. It's also useful for
    your paper's methodology section — you can report exact numbers.
    """
    path = PREPROCESS_DIR / "config.json"
    with open(path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"  Saved config.json")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"Project root:    {PROJECT_ROOT}")
    print(f"Input from:      {OUTPUT_DIR}")
    print(f"Output to:       {PREPROCESS_DIR}")
    print()

    # ── Step 1: Load processed data ──
    sessions = load_processed_sessions()
    if not sessions:
        print("❌ No processed data found! Run dataexplorer.py first.")
        exit(1)

    # ── Step 2-3: Downsample + remove outliers (done per session below) ──
    # We need to downsample and clean BEFORE computing norm stats,
    # otherwise the stats include outliers and high-frequency noise

    print("=" * 60)
    print("STEP 2-3: Downsample + Remove Outliers (pre-normalization)")
    print("=" * 60)

    cleaned_sessions = {}
    for name, df in sessions.items():
        print(f"\n  Cleaning {name}...")
        df = downsample(df, CONFIG["target_hz"])
        df = remove_outliers(df)
        cleaned_sessions[name] = df

    # ── Step 4: Compute normalization stats from TRAINING data ──
    print(f"\n{'=' * 60}")
    print("STEP 4: Compute normalization stats")
    print("=" * 60)
    norm_stats = compute_norm_stats(cleaned_sessions)

    # ── Step 5-6: Normalize + create windows for each split ──
    print(f"\n{'=' * 60}")
    print("STEP 5-6: Normalize + Create sliding windows")
    print("=" * 60)

    # --- TRAIN ---
    print(f"\n── TRAIN (sessions: {CONFIG['train_sessions']}) ──")
    X_train, Y_train = build_split(
        cleaned_sessions, CONFIG["train_sessions"], norm_stats, "train"
    )

    # --- VAL ---
    print(f"\n── VALIDATION (sessions: {CONFIG['val_sessions']}) ──")
    X_val, Y_val = build_split(
        cleaned_sessions, CONFIG["val_sessions"], norm_stats, "val"
    )

    # --- TEST ---
    print(f"\n── TEST (sessions: {CONFIG['test_sessions']}) ──")
    X_test, Y_test = build_split(
        cleaned_sessions, CONFIG["test_sessions"], norm_stats, "test"
    )

    # ── Step 7: Save everything ──
    print(f"\n{'=' * 60}")
    print("STEP 7: Saving preprocessed data")
    print("=" * 60)

    save_dataset(X_train, Y_train, "train")
    save_dataset(X_val, Y_val, "val")
    save_dataset(X_test, Y_test, "test")
    save_norm_stats(norm_stats)
    save_config()

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("✅ Preprocessing complete!")
    print(f"{'=' * 60}")
    print(f"  Train:      {len(X_train):>8,} windows")
    print(f"  Validation: {len(X_val):>8,} windows")
    print(f"  Test:       {len(X_test):>8,} windows")
    print(f"  Total:      {len(X_train)+len(X_val)+len(X_test):>8,} windows")
    print(f"\n  Each input window:  {CONFIG['observe_len']} steps × "
          f"{len(CONFIG['input_features'])} features "
          f"({CONFIG['observe_len']/CONFIG['target_hz']:.1f}s of history)")
    print(f"  Each output window: {CONFIG['predict_len']} steps × "
          f"{len(CONFIG['output_features'])} features "
          f"({CONFIG['predict_len']/CONFIG['target_hz']:.1f}s into future)")
    print(f"\n  Saved to: {PREPROCESS_DIR.resolve()}")
