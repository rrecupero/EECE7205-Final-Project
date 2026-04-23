"""
NCLT Dataset Loader & Explorer
================================
Step 2: Load NCLT data, align timestamps, visualize trajectories.

Your directory structure:
    Dataset/
    ├── Groundtruth/
    │   ├── groundtruth_2012-01-08.csv
    │   ├── groundtruth_2012-01-15.csv
    │   └── groundtruth_2012-01-22.csv
    └── sensor_dataset/
        ├── 2012-01-08_sen.tar.gz
        ├── 2012-01-15_sen.tar.gz
        └── 2012-01-22_sen.tar.gz
    SRC/
    └── dataexplorer.py   <-- this script lives here
"""

import os
import tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

# =============================================================================
# CONFIG
# =============================================================================
# Since this script is in SRC/, go one level up to find Dataset/
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
GT_DIR = DATASET_DIR / "Groundtruth"
SENSOR_DIR = DATASET_DIR / "sensor_dataset"

# Sessions you downloaded (just the date strings)
SESSIONS = [
    "2012-01-08",
    "2012-01-22",
    "2012-02-18",
    "2013-02-23",
    "2013-04-05",
]

# Directory to save plots
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Directory to save processed output data
OUTPUT_DIR = DATASET_DIR / "Outputdata"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# 1. EXTRACT TAR FILES
# =============================================================================
def extract_tar(session: str):
    """
    Extract a session's sen.tar.gz into a folder inside sensor_dataset/.

    What's happening here:
    - The tar.gz file is a compressed archive (like a .zip) containing
      all the sensor CSV files for that session.
    - We extract it into a folder named by the session date so we can
      access the CSVs inside (especially odometry_mu_100hz.csv).

    After extraction, the structure becomes:
        sensor_dataset/
        ├── 2012-01-08_sen.tar.gz
        └── 2012-01-08/          <-- extracted folder
            ├── odometry_mu_100hz.csv
            ├── odometry_cov_100hz.csv
            ├── gps.csv
            ├── gps_rtk.csv
            └── ... (other sensor files)
    """
    tar_path = SENSOR_DIR / f"{session}_sen.tar.gz"
    extract_dir = SENSOR_DIR / session

    # Skip if already extracted
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  [SKIP] {session} sensor data already extracted")
        return True

    if not tar_path.exists():
        print(f"  [WARN] {tar_path.name} not found — skipping")
        return False

    print(f"  [EXTRACTING] {tar_path.name}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        # Extract into a session-named subfolder to keep things organized
        tar.extractall(path=extract_dir)
    print(f"  [DONE] Extracted to {extract_dir}")
    return True


def extract_all_sessions():
    """Extract tar files for all sessions."""
    print("=" * 60)
    print("STEP 1: Extracting sensor tar files")
    print("=" * 60)
    for session in SESSIONS:
        extract_tar(session)
    print()


# =============================================================================
# 2. LOAD GROUND TRUTH
# =============================================================================
def load_ground_truth(session: str) -> pd.DataFrame:
    """
    Load groundtruth_<date>.csv for a session.

    What is ground truth?
    ---------------------
    This is the ACCURATE position of the robot, computed OFFLINE by fusing
    RTK-GPS (centimeter-level GPS) with lidar scan matching. Think of it as
    the "answer key" — it tells us where the robot actually was at each moment.

    File format (no header row):
        Column 0: timestamp in microseconds (e.g., 1326044406516741)
        Column 1: x position in meters (forward direction)
        Column 2: y position in meters (lateral direction)
        Column 3: z position in meters (vertical direction)
        Column 4: roll in radians (tilt around forward axis)
        Column 5: pitch in radians (tilt around lateral axis)
        Column 6: yaw in radians (heading / compass direction)

    The coordinate frame is a local frame — (0,0,0) is roughly where the
    robot started that day's run. Positive x is roughly north, positive y
    is roughly west (but this varies by session).
    """
    gt_path = GT_DIR / f"groundtruth_{session}.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    df = pd.read_csv(
        gt_path,
        header=None,
        names=["timestamp", "x", "y", "z", "roll", "pitch", "yaw"],
        comment="%",  # Skip comment lines starting with %
        low_memory=False,
    )

    # The CSV may have a text header row or other non-numeric junk mixed in.
    # Force all columns to numeric — any non-numeric values become NaN.
    rows_before = len(df)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that couldn't be converted (header rows, comments, etc.)
    df = df.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"  [CLEAN] Dropped {rows_dropped} non-numeric rows from ground truth")

    # Convert timestamp from microseconds to seconds (relative to start)
    # This makes plotting and windowing much easier
    df["time_s"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 1e6

    duration = df["time_s"].iloc[-1]
    print(f"  Ground truth: {len(df):>8,} rows | "
          f"duration = {duration:.1f}s ({duration/60:.1f} min)")
    return df


# =============================================================================
# 3. LOAD ODOMETRY
# =============================================================================
def find_odometry_file(session: str) -> Path:
    """
    Find odometry_mu_100hz.csv inside the extracted sensor directory.

    Why do we need to search?
    -------------------------
    Different tar files may extract with slightly different folder structures.
    Some extract directly into the target folder, others create a subfolder.
    This function walks the directory tree to find the file regardless of
    the exact nesting structure.
    """
    extract_dir = SENSOR_DIR / session

    # Search recursively for the odometry file
    for root, dirs, files in os.walk(extract_dir):
        if "odometry_mu_100hz.csv" in files:
            return Path(root) / "odometry_mu_100hz.csv"

    raise FileNotFoundError(
        f"odometry_mu_100hz.csv not found in {extract_dir}. "
        f"Contents: {list(extract_dir.rglob('*'))[:10]}"
    )


def load_odometry(session: str) -> pd.DataFrame:
    """
    Load odometry_mu_100hz.csv from the extracted sensor directory.

    What is odometry? (recap)
    -------------------------
    This is the robot's SELF-ESTIMATED position based on wheel encoders + IMU.
    It's what the robot THINKS its position is. It drifts over time because
    small measurement errors in wheel rotation accumulate.

    "mu" (μ) = mean of the pose estimate (the best guess)
    "100hz" = sampled at 100 times per second

    Same format as ground truth:
        timestamp (μs), x, y, z, roll, pitch, yaw

    Key difference from ground truth:
    - Odometry is available in REAL-TIME on the robot
    - Ground truth is computed OFFLINE and wouldn't be available during deployment
    - Odometry drifts; ground truth doesn't (or drifts much less)
    """
    odom_path = find_odometry_file(session)
    print(f"  Found odometry at: {odom_path.relative_to(SENSOR_DIR)}")

    df = pd.read_csv(
        odom_path,
        header=None,
        names=["timestamp", "x", "y", "z", "roll", "pitch", "yaw"],
        comment="%",  # Skip comment lines starting with %
        low_memory=False,
    )

    # Force all columns to numeric, drop any non-numeric rows
    rows_before = len(df)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"  [CLEAN] Dropped {rows_dropped} non-numeric rows from odometry")

    df["time_s"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 1e6

    duration = df["time_s"].iloc[-1]
    rate = len(df) / duration if duration > 0 else 0
    print(f"  Odometry:     {len(df):>8,} rows | "
          f"duration = {duration:.1f}s | rate ≈ {rate:.0f} Hz")
    return df


# =============================================================================
# 4. ALIGN TIMESTAMPS
# =============================================================================
def align_timestamps(gt: pd.DataFrame, odom: pd.DataFrame) -> pd.DataFrame:
    """
    Align ground truth and odometry by nearest timestamp.

    Why do we need alignment?
    -------------------------
    Ground truth and odometry are recorded at DIFFERENT rates and timestamps:
    - Odometry: 100 Hz (every 10 ms)
    - Ground truth: variable rate (depends on when GPS + lidar fusion converges)

    Their timestamps won't match exactly. For example:
        GT timestamp:   1326044406.500000
        Nearest odom:   1326044406.500312  (0.3 ms off)

    We use "nearest neighbor" matching: for each ground truth timestamp,
    we find the odometry reading with the closest timestamp.

    How np.searchsorted works:
    --------------------------
    Given a sorted array [10, 20, 30, 40, 50] and a query value 27,
    searchsorted returns index 2 (where 27 would be inserted to keep
    the array sorted). We then check whether index 2 (value 30) or
    index 1 (value 20) is actually closer to 27. Answer: 30 is 3 away,
    20 is 7 away, so we pick index 2.

    Returns a DataFrame with both GT and odometry columns aligned row-by-row.
    """
    gt_times = gt["timestamp"].values
    odom_times = odom["timestamp"].values

    # For each GT timestamp, find insertion point in sorted odometry times
    indices = np.searchsorted(odom_times, gt_times, side="left")
    indices = np.clip(indices, 0, len(odom_times) - 1)

    # Check if the previous index is actually closer
    prev_indices = np.clip(indices - 1, 0, len(odom_times) - 1)
    diff_curr = np.abs(odom_times[indices] - gt_times)
    diff_prev = np.abs(odom_times[prev_indices] - gt_times)
    indices = np.where(diff_prev < diff_curr, prev_indices, indices)

    # Build aligned dataframe
    aligned = pd.DataFrame({
        "timestamp": gt_times,
        "time_s": gt["time_s"].values,
        # Ground truth columns (the "answer key")
        "gt_x": gt["x"].values,
        "gt_y": gt["y"].values,
        "gt_z": gt["z"].values,
        "gt_roll": gt["roll"].values,
        "gt_pitch": gt["pitch"].values,
        "gt_yaw": gt["yaw"].values,
        # Odometry columns (the robot's self-estimate, aligned to GT times)
        "odom_x": odom["x"].values[indices],
        "odom_y": odom["y"].values[indices],
        "odom_z": odom["z"].values[indices],
        "odom_roll": odom["roll"].values[indices],
        "odom_pitch": odom["pitch"].values[indices],
        "odom_yaw": odom["yaw"].values[indices],
    })

    # Quality check: how close were the matched timestamps?
    time_diffs_ms = np.abs(odom_times[indices] - gt_times) / 1e3
    print(f"  Alignment:    {len(aligned):>8,} matched pairs | "
          f"median Δt = {np.median(time_diffs_ms):.1f} ms | "
          f"max Δt = {np.max(time_diffs_ms):.1f} ms")

    return aligned


# =============================================================================
# 5. COMPUTE DERIVED FEATURES (velocities via finite differences)
# =============================================================================
def compute_velocities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute linear and angular velocities from ground truth positions.

    Why finite differences?
    -----------------------
    The NCLT dataset gives us positions (x, y) and heading (yaw) but NOT
    velocities directly. For your trajectory prediction models, velocity
    is a crucial input feature — it tells the model not just WHERE the
    robot is, but HOW FAST and IN WHAT DIRECTION it's moving.

    Finite difference approximation:
        vx ≈ (x[t+1] - x[t]) / (time[t+1] - time[t])
        vy ≈ (y[t+1] - y[t]) / dt
        ω  ≈ (yaw[t+1] - yaw[t]) / dt

    This is a first-order numerical derivative. It's noisy (amplifies
    high-frequency noise in the position data) but good enough for our
    purposes. In Step 3 (preprocessing), we'll smooth this further.

    Yaw wraparound problem:
    -----------------------
    Yaw is an angle in [-π, π]. If the robot is heading at yaw = 3.1 rad
    and turns slightly to yaw = -3.1 rad, the naive difference is -6.2 rad
    (almost a full rotation!) when the actual change was only ~0.08 rad.

    We fix this with arctan2(sin(Δyaw), cos(Δyaw)), which always gives
    the shortest angular path between two angles.
    """
    dt = np.diff(df["time_s"].values)
    dt = np.clip(dt, 1e-6, None)  # avoid division by zero

    dx = np.diff(df["gt_x"].values)
    dy = np.diff(df["gt_y"].values)
    dyaw = np.diff(df["gt_yaw"].values)

    # Handle yaw wraparound: get shortest angular difference
    dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))

    vx = dx / dt
    vy = dy / dt
    omega = dyaw / dt

    # First row has no velocity (need two points to compute a derivative)
    df = df.copy()
    df["vx"] = np.concatenate([[np.nan], vx])
    df["vy"] = np.concatenate([[np.nan], vy])
    df["omega"] = np.concatenate([[np.nan], omega])

    # Drop the first row (NaN) and reset index
    df = df.iloc[1:].reset_index(drop=True)

    speed = np.sqrt(df["vx"]**2 + df["vy"]**2)
    print(f"  Velocities:   vx  [{df['vx'].min():>7.2f}, {df['vx'].max():>7.2f}] m/s | "
          f"ω [{df['omega'].min():>7.2f}, {df['omega'].max():>7.2f}] rad/s | "
          f"mean speed = {speed.mean():.2f} m/s")
    return df


# =============================================================================
# 6. LOAD A FULL SESSION (orchestrates steps 2-5)
# =============================================================================
def load_session(session: str) -> pd.DataFrame:
    """
    Full pipeline for one session: load GT → load odom → align → velocities.

    This function ties together everything above. For each session date,
    it produces a single clean DataFrame where every row is one timestep
    containing:
        - time_s: time in seconds from start
        - gt_x, gt_y, gt_z, gt_roll, gt_pitch, gt_yaw: accurate position
        - odom_x, odom_y, ..., odom_yaw: robot's self-estimate
        - vx, vy, omega: computed velocities from ground truth
    """
    print(f"\n{'─' * 60}")
    print(f"Loading session: {session}")
    print(f"{'─' * 60}")

    gt = load_ground_truth(session)
    odom = load_odometry(session)
    aligned = align_timestamps(gt, odom)
    aligned = compute_velocities(aligned)
    return aligned


# =============================================================================
# 7. VISUALIZATION
# =============================================================================
def plot_trajectory_2d(sessions: dict):
    """
    Plot all sessions' ground truth trajectories in 2D.

    What to look for:
    - Do the sessions follow similar paths? (NCLT dataset covers the same
      campus loop, so they should overlap roughly)
    - Are there different route variations?
    - Circle markers = start, Square markers = end
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sessions)))

    for (name, df), color in zip(sessions.items(), colors):
        ax.plot(df["gt_x"], df["gt_y"], color=color, alpha=0.7,
                linewidth=0.5, label=name)
        ax.scatter(df["gt_x"].iloc[0], df["gt_y"].iloc[0],
                   color=color, marker="o", s=80, zorder=5,
                   edgecolors="black", label=f"{name} start")
        ax.scatter(df["gt_x"].iloc[-1], df["gt_y"].iloc[-1],
                   color=color, marker="s", s=80, zorder=5,
                   edgecolors="black", label=f"{name} end")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("NCLT Ground Truth Trajectories (2D)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "trajectories_2d.png", dpi=150)
    print(f"  Saved: {PLOTS_DIR / 'trajectories_2d.png'}")


def plot_odometry_vs_gt(df: pd.DataFrame, session: str):
    """
    Compare odometry vs ground truth for a single session.

    What to look for:
    - LEFT PLOT: The red (odometry) and blue (ground truth) paths start
      together but diverge over time. This is odometry drift in action.
    - RIGHT PLOT: The position error (distance between odom and GT)
      generally increases over time — this is the cumulative effect of
      small wheel encoder errors compounding.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].plot(df["gt_x"], df["gt_y"], "b-", alpha=0.7,
                 linewidth=0.8, label="Ground Truth")
    axes[0].plot(df["odom_x"], df["odom_y"], "r-", alpha=0.5,
                 linewidth=0.8, label="Odometry")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title(f"{session}: GT vs Odometry")
    axes[0].legend()
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    pos_err = np.sqrt(
        (df["gt_x"] - df["odom_x"])**2 + (df["gt_y"] - df["odom_y"])**2
    )
    axes[1].plot(df["time_s"], pos_err, "k-", linewidth=0.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position Error (m)")
    axes[1].set_title(f"{session}: Odometry Drift Over Time")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"odom_vs_gt_{session}.png", dpi=150)
    print(f"  Saved: {PLOTS_DIR / f'odom_vs_gt_{session}.png'}")


def plot_velocity_distributions(df: pd.DataFrame, session: str):
    """
    Histograms of computed velocities.

    What to look for:
    - vx and vy should be centered near 0 with tails out to ~1-3 m/s
      (the Segway moves at walking/jogging speed)
    - omega should be centered near 0 (going straight) with tails
      out to ~1-2 rad/s (turning)
    - Any extreme outliers (e.g., 100 m/s) indicate bad data or
      timestamp issues that need cleaning
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(df["vx"].dropna(), bins=100, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("vx (m/s)")
    axes[0].set_title("Linear Velocity X")

    axes[1].hist(df["vy"].dropna(), bins=100, color="coral", edgecolor="none")
    axes[1].set_xlabel("vy (m/s)")
    axes[1].set_title("Linear Velocity Y")

    axes[2].hist(df["omega"].dropna(), bins=100, color="seagreen", edgecolor="none")
    axes[2].set_xlabel("ω (rad/s)")
    axes[2].set_title("Angular Velocity")

    fig.suptitle(f"{session}: Velocity Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"velocities_{session}.png", dpi=150)
    print(f"  Saved: {PLOTS_DIR / f'velocities_{session}.png'}")


def plot_state_timeseries(df: pd.DataFrame, session: str):
    """
    Plot all 6 state variables over time.

    This is your most complete view of one session. Each subplot shows
    one state variable evolving over the full duration of the run.

    What to look for:
    - x and y: should show the robot traversing a path (smooth, no jumps)
    - yaw: should wrap around if the robot does full loops
    - vx, vy: should be noisy but bounded (no crazy spikes)
    - omega: spikes indicate sharp turns
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)

    axes[0, 0].plot(df["time_s"], df["gt_x"], linewidth=0.5)
    axes[0, 0].set_ylabel("X (m)")
    axes[0, 0].set_title("Position X")

    axes[0, 1].plot(df["time_s"], df["gt_y"], linewidth=0.5)
    axes[0, 1].set_ylabel("Y (m)")
    axes[0, 1].set_title("Position Y")

    axes[1, 0].plot(df["time_s"], df["gt_yaw"], linewidth=0.5, color="green")
    axes[1, 0].set_ylabel("Yaw (rad)")
    axes[1, 0].set_title("Heading")

    axes[1, 1].plot(df["time_s"], df["vx"], linewidth=0.3, color="steelblue")
    axes[1, 1].set_ylabel("vx (m/s)")
    axes[1, 1].set_title("Velocity X")

    axes[2, 0].plot(df["time_s"], df["vy"], linewidth=0.3, color="coral")
    axes[2, 0].set_ylabel("vy (m/s)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_title("Velocity Y")

    axes[2, 1].plot(df["time_s"], df["omega"], linewidth=0.3, color="seagreen")
    axes[2, 1].set_ylabel("ω (rad/s)")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_title("Angular Velocity")

    fig.suptitle(f"{session}: Full State Time Series", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"timeseries_{session}.png", dpi=150)
    print(f"  Saved: {PLOTS_DIR / f'timeseries_{session}.png'}")


def save_processed_data(sessions: dict):
    """
    Save the processed, aligned DataFrames to CSV files in Outputdata/.

    Why save?
    ---------
    The raw data loading, timestamp alignment, and velocity computation
    take time — especially as you add more sessions. By saving the
    processed output, you only need to run this script once. All future
    scripts (preprocessing, windowing, training) can load directly from
    these clean CSVs instead of re-processing the raw data every time.

    What gets saved per session:
    - Every row is one aligned timestep
    - Columns: time_s, gt_x, gt_y, gt_z, gt_roll, gt_pitch, gt_yaw,
               odom_x, odom_y, ..., odom_yaw, vx, vy, omega
    - All values are numeric (no strings, no NaNs)

    Also saves a combined CSV with all sessions stacked together,
    with a 'session' column so you can identify which session each
    row came from. This is useful for train/val/test splitting by session.
    """
    print(f"\n{'=' * 60}")
    print("SAVING PROCESSED DATA")
    print(f"{'=' * 60}")

    all_dfs = []
    for name, df in sessions.items():
        # Save individual session
        out_path = OUTPUT_DIR / f"processed_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path.name}  ({len(df):,} rows)")

        # Prepare for combined file
        df_copy = df.copy()
        df_copy["session"] = name
        all_dfs.append(df_copy)

    # Save combined file with all sessions
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = OUTPUT_DIR / "all_sessions_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"  Saved: {combined_path.name}  ({len(combined):,} rows, "
          f"{len(sessions)} sessions)")

    # Save a quick metadata summary
    summary_path = OUTPUT_DIR / "dataset_summary.txt"
    with open(summary_path, "w") as f:
        f.write("NCLT Dataset - Processed Data Summary\n")
        f.write("=" * 50 + "\n\n")
        for name, df in sessions.items():
            speed = np.sqrt(df["vx"]**2 + df["vy"]**2)
            f.write(f"Session: {name}\n")
            f.write(f"  Samples:      {len(df):,}\n")
            f.write(f"  Duration:     {df['time_s'].iloc[-1]/60:.1f} min\n")
            f.write(f"  Mean speed:   {speed.mean():.2f} m/s\n")
            f.write(f"  Max speed:    {speed.max():.2f} m/s\n")
            f.write(f"  X range:      [{df['gt_x'].min():.1f}, {df['gt_x'].max():.1f}] m\n")
            f.write(f"  Y range:      [{df['gt_y'].min():.1f}, {df['gt_y'].max():.1f}] m\n\n")
        f.write(f"Total samples:  {sum(len(d) for d in sessions.values()):,}\n")
        f.write(f"Total sessions: {len(sessions)}\n")
    print(f"  Saved: {summary_path.name}")
    print()


def print_session_summary(sessions: dict):
    """Print a summary table of all loaded sessions."""
    print(f"\n{'=' * 75}")
    print(f"{'SESSION SUMMARY':^75}")
    print(f"{'=' * 75}")
    print(f"{'Session':<15} {'Samples':>10} {'Duration':>14} "
          f"{'Mean Speed':>12} {'Max Speed':>12}")
    print(f"{'─' * 75}")
    for name, df in sessions.items():
        speed = np.sqrt(df["vx"]**2 + df["vy"]**2)
        duration_min = df['time_s'].iloc[-1] / 60
        print(f"{name:<15} {len(df):>10,} {duration_min:>11.1f} min "
              f"{speed.mean():>10.2f} m/s {speed.max():>10.2f} m/s")
    print(f"{'─' * 75}")
    total = sum(len(df) for df in sessions.values())
    print(f"{'TOTAL':<15} {total:>10,}")
    print()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"Project root:   {PROJECT_ROOT}")
    print(f"Dataset dir:    {DATASET_DIR}")
    print(f"Groundtruth:    {GT_DIR}")
    print(f"Sensor data:    {SENSOR_DIR}")
    print(f"Plots output:   {PLOTS_DIR}")
    print()

    # ── Step 1: Extract tar files ──
    extract_all_sessions()

    # ── Step 2: Load all sessions ──
    sessions = {}
    for session_name in SESSIONS:
        try:
            sessions[session_name] = load_session(session_name)
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")

    if not sessions:
        print("\n❌ No sessions loaded!")
        print("   Check that your files match the expected naming:")
        print(f"   - Groundtruth: {GT_DIR}/groundtruth_<date>.csv")
        print(f"   - Sensors:     {SENSOR_DIR}/<date>_sen.tar.gz")
        exit(1)

    # ── Step 3: Print summary ──
    print_session_summary(sessions)

    # ── Step 4: Save processed data ──
    save_processed_data(sessions)

    # ── Step 5: Visualize ──
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # 5a. All trajectories overlaid
    plot_trajectory_2d(sessions)

    # 5b. Detailed plots for each session
    for session_name, df in sessions.items():
        print(f"\n  Plotting {session_name}...")
        plot_odometry_vs_gt(df, session_name)
        plot_velocity_distributions(df, session_name)
        plot_state_timeseries(df, session_name)

    # ── Done ──
    print(f"\n{'=' * 60}")
    print("✅ Data loading and exploration complete!")
    print(f"   Sessions loaded: {len(sessions)}")
    print(f"   Total samples:   {sum(len(d) for d in sessions.values()):,}")
    print(f"   Processed data:  {OUTPUT_DIR.resolve()}")
    print(f"   Plots saved to:  {PLOTS_DIR.resolve()}")
    print(f"{'=' * 60}")
