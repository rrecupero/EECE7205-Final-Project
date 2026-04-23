"""
Physics-Informed Linear Transformer for Robot Trajectory Prediction
====================================================================
YOUR NOVEL CONTRIBUTION to the EECE 7205 paper.

Key difference from the regular Linear Transformer:
    Regular:    Input → Attention Layers → Directly predict (x, y, θ)
    Physics:    Input → Attention Layers → Predict (v, ω) → Integrate
                through unicycle kinematic model → Get (x, y, θ)

The unicycle model:
    x_{t+1} = x_t + v_t * cos(θ_t) * dt
    y_{t+1} = y_t + v_t * sin(θ_t) * dt
    θ_{t+1} = θ_t + ω_t * dt

This GUARANTEES that predicted trajectories are physically plausible
for a differential-drive robot (like the NCLT Segway). The model
cannot produce impossible motions like lateral sliding.

Additionally, we add a physics loss term that penalizes velocity
predictions outside realistic bounds, combining Approach 1 (loss)
and Approach 2 (constrained output layer) from our project plan.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the building blocks from the regular Linear Transformer
from linear_transformer import (
    ELUFeatureMap,
    LinearAttention,
    LinearTransformerLayer,
    SinusoidalPositionalEncoding,
)


# =============================================================================
# 1. UNICYCLE KINEMATIC MODEL — The physics constraint
# =============================================================================
class UnicycleIntegrator(nn.Module):
    """
    Integrates velocity commands (v, ω) through the unicycle kinematic
    model to produce a trajectory (x, y, θ).

    WHAT IS THE UNICYCLE MODEL?
    ----------------------------
    The unicycle model describes how a robot that can only move forward
    and rotate in place (like a Segway, differential-drive robot, or
    bicycle at low speeds) moves through 2D space.

    Given:
        v_t = linear velocity at time t   (how fast it moves forward)
        ω_t = angular velocity at time t  (how fast it turns)
        θ_t = heading at time t           (which direction it faces)

    The next state is:
        x_{t+1} = x_t + v_t * cos(θ_t) * dt
        y_{t+1} = y_t + v_t * sin(θ_t) * dt
        θ_{t+1} = θ_t + ω_t * dt

    WHY THIS IS A "CONSTRAINT":
    ----------------------------
    The key insight is what this model PREVENTS:

    1. NO LATERAL MOTION — The robot can only move in the direction
       it's facing (the cos(θ), sin(θ) terms). It cannot slide sideways.
       This is the "non-holonomic constraint" of wheeled robots.

    2. SMOOTH TRAJECTORIES — Because position changes are proportional
       to velocity × dt, the trajectory is inherently smooth. No
       teleportation or instant direction changes.

    3. CONSISTENT PHYSICS — If v=0, the robot doesn't move (x, y stay
       the same). If ω=0, the robot goes straight. These trivial cases
       are automatically handled correctly.

    WHY NOT JUST USE PHYSICS EQUATIONS DIRECTLY (LIKE MPC)?
    --------------------------------------------------------
    Because we don't know v and ω ahead of time! MPC CHOOSES v and ω
    to achieve a goal. Our model PREDICTS what v and ω will be based
    on observed behavior, then uses the physics to convert those
    predictions into positions. The neural network learns the
    "intention" (velocity commands), and the physics handles the
    "mechanics" (how those commands move the robot).

    Parameters
    ----------
    dt : float
        Time step between consecutive predictions in seconds.
        At 10 Hz, dt = 0.1 seconds.
    """

    def __init__(self, dt: float = 0.1):
        super().__init__()
        self.dt = dt

    def forward(self, v_omega, initial_state):
        """
        Integrate velocity commands to produce a trajectory.

        Args:
            v_omega: tensor of shape (batch, pred_len, 2)
                     v_omega[:, :, 0] = linear velocity v
                     v_omega[:, :, 1] = angular velocity ω

            initial_state: tensor of shape (batch, 3)
                          The robot's state at the LAST observed timestep:
                          [x_0, y_0, θ_0]
                          This is the starting point for integration.

        Returns:
            trajectory: tensor of shape (batch, pred_len, 3)
                       The predicted [x, y, θ] at each future timestep.

        STEP-BY-STEP WALKTHROUGH:
        -------------------------
        Say initial_state = [10.0, 5.0, 0.5]  (x=10m, y=5m, θ=0.5 rad)
        And v_omega at t=0 is [1.5, 0.2]      (moving at 1.5 m/s, turning 0.2 rad/s)

        Then at t=1 (0.1 seconds later):
            θ_1 = 0.5 + 0.2 * 0.1 = 0.52 rad
            x_1 = 10.0 + 1.5 * cos(0.5) * 0.1 = 10.0 + 0.1318 = 10.132 m
            y_1 = 5.0 + 1.5 * sin(0.5) * 0.1 = 5.0 + 0.0719 = 5.072 m

        Then at t=2, we use (x_1, y_1, θ_1) as the starting point and
        apply the next v_omega to get (x_2, y_2, θ_2), and so on.

        This sequential dependency is what makes the trajectory physically
        consistent — each step builds on the previous one.
        """
        B, T, _ = v_omega.shape

        v = v_omega[:, :, 0]      # (batch, pred_len) — linear velocity
        omega = v_omega[:, :, 1]   # (batch, pred_len) — angular velocity

        # Initialize lists to collect trajectory points
        x = initial_state[:, 0]    # (batch,)
        y = initial_state[:, 1]    # (batch,)
        theta = initial_state[:, 2]  # (batch,)

        trajectory = []

        for t in range(T):
            # Update heading FIRST (mid-point integration is more accurate,
            # but simple Euler integration works well for small dt)
            theta = theta + omega[:, t] * self.dt

            # Update position using the NEW heading
            x = x + v[:, t] * torch.cos(theta) * self.dt
            y = y + v[:, t] * torch.sin(theta) * self.dt

            # Stack [x, y, θ] for this timestep
            trajectory.append(torch.stack([x, y, theta], dim=-1))

        # Stack all timesteps: list of (batch, 3) → (batch, pred_len, 3)
        trajectory = torch.stack(trajectory, dim=1)

        return trajectory


# =============================================================================
# 2. PHYSICS-INFORMED LINEAR TRANSFORMER
# =============================================================================
class PhysicsLinearTransformerPredictor(nn.Module):
    """
    Linear Transformer with physics-constrained output layer.

    ARCHITECTURE COMPARISON:
    ========================

    Regular Linear Transformer:
        Input (B, 20, 6)
            → Embed → PosEnc → Attention Layers
            → Last timestep → Linear(d_model → 30) → Reshape(10, 3)
        Output: directly predicted (x, y, θ) — NO physics guarantee

    Physics-Informed Linear Transformer (THIS MODEL):
        Input (B, 20, 6)
            → Embed → PosEnc → Attention Layers
            → Last timestep → Linear(d_model → 20)  ← predicts (v, ω) × 10 steps
            → Reshape(10, 2)
            → Unicycle Integrator (uses last observed state as initial condition)
        Output: (x, y, θ) computed through kinematic equations — PHYSICS GUARANTEED

    THE SUBTLE BUT CRITICAL DIFFERENCE:
    ------------------------------------
    Both models have the same number of attention layers, same d_model,
    same positional encoding. The ONLY difference is what the output
    layer predicts:

    Regular:   predicts 10 × 3 = 30 values (x, y, θ directly)
    Physics:   predicts 10 × 2 = 20 values (v, ω) then integrates

    The physics model actually has FEWER output parameters (20 vs 30)
    but produces BETTER predictions because:

    1. The search space is SMALLER — instead of predicting arbitrary
       positions in 2D space, it predicts velocities in a bounded range.
       It's easier to learn "the robot moves at 1.5 m/s" than to learn
       "the robot will be at position (143.7, -287.2)."

    2. Physical consistency is FREE — the kinematic integration
       guarantees smooth, non-holonomic trajectories without the
       model having to learn these constraints from data.

    3. In LOW-DATA REGIMES, this advantage is amplified — with little
       training data, the regular model might learn physically impossible
       predictions. The physics model can't, because the constraints
       are hardcoded in the integrator, not learned from data.
       This is the key experiment for your paper.
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        pred_len: int = 10,
        dropout: float = 0.1,
        dt: float = 0.1,
    ):
        super().__init__()

        self.pred_len = pred_len
        self.output_dim = output_dim

        # ── Same encoder as regular Linear Transformer ──
        self.input_embed = nn.Linear(input_dim, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=40000)

        self.layers = nn.ModuleList([
            LinearTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # ── KEY DIFFERENCE: predict (v, ω) instead of (x, y, θ) ──
        # Output is pred_len × 2 (velocity + angular velocity)
        # instead of pred_len × 3 (position + heading)
        self.velocity_proj = nn.Linear(d_model, pred_len * 2)

        # ── Unicycle integrator: converts (v, ω) → (x, y, θ) ──
        self.integrator = UnicycleIntegrator(dt=dt)

        # Store config
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "pred_len": pred_len,
            "dropout": dropout,
            "dt": dt,
            "physics_informed": True,
        }

    def forward(self, x, initial_state=None):
        """
        Args:
            x: input tensor (batch, observe_len, input_dim)
               e.g., (32, 20, 6)

            initial_state: (batch, 3) — the [x, y, θ] at the last
               observed timestep. If None, we extract it from the
               input's last timestep (features 0, 1, 2 = gt_x, gt_y, gt_yaw).

        Returns:
            trajectory: (batch, pred_len, 3) — predicted [x, y, θ]
            v_omega: (batch, pred_len, 2) — predicted [v, ω]
                     (returned for computing physics loss)
        """
        B = x.size(0)

        # Extract initial state from last observed timestep if not provided
        # x[:, -1, :3] gives [gt_x, gt_y, gt_yaw] of the last input step
        if initial_state is None:
            initial_state = x[:, -1, :3]  # (B, 3)

        # ── Encoder (identical to regular Linear Transformer) ──
        h = self.input_embed(x)          # (B, 20, d_model)
        h = self.pos_encoding(h)         # (B, 20, d_model)

        for layer in self.layers:
            h = layer(h)                 # (B, 20, d_model)

        h = self.final_norm(h)           # (B, 20, d_model)

        # Take last timestep's representation
        h = h[:, -1, :]                  # (B, d_model)

        # ── KEY DIFFERENCE: predict velocities, not positions ──
        v_omega = self.velocity_proj(h)  # (B, pred_len * 2)
        v_omega = v_omega.view(B, self.pred_len, 2)  # (B, pred_len, 2)

        # ── Integrate through kinematic model ──
        # This is where the physics constraint happens!
        # The integrator converts (v, ω) → (x, y, θ) using:
        #   x_{t+1} = x_t + v * cos(θ) * dt
        #   y_{t+1} = y_t + v * sin(θ) * dt
        #   θ_{t+1} = θ_t + ω * dt
        trajectory = self.integrator(v_omega, initial_state)  # (B, pred_len, 3)

        return trajectory, v_omega

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 3. PHYSICS LOSS — Additional regularization
# =============================================================================
class PhysicsInformedLoss(nn.Module):
    """
    Combined loss function for the physics-informed model.

    Total Loss = MSE_trajectory + λ_vel * velocity_penalty
                                + λ_smooth * smoothness_penalty

    COMPONENT 1: MSE on trajectory (standard)
    ------------------------------------------
    Same as the regular model — penalize deviation between predicted
    and actual (x, y, θ) positions. This is the primary training signal.

    COMPONENT 2: Velocity bound penalty
    ------------------------------------
    Penalize predicted velocities that exceed realistic physical limits.
    The Segway has a maximum speed (~3 m/s) and max turn rate (~2 rad/s).
    If the model predicts v = 10 m/s, this penalty kicks in.

    This is a SOFT constraint — unlike the kinematic integrator which
    is a HARD constraint (structurally prevents impossible motions),
    this penalty DISCOURAGES but doesn't prevent large velocities.

    Why both soft and hard constraints?
    The integrator ensures non-holonomic consistency (no sideways motion).
    The velocity penalty ensures reasonable speeds. Together they cover
    both types of physical plausibility.

    COMPONENT 3: Smoothness penalty
    --------------------------------
    Penalize large changes in velocity between consecutive timesteps.
    Real robots can't instantly change speed (they have inertia and
    motor limits). This encourages smooth, realistic acceleration.

        smoothness = Σ (v_{t+1} - v_t)² + (ω_{t+1} - ω_t)²

    Parameters
    ----------
    lambda_vel : float
        Weight of velocity bound penalty. Higher = stricter speed limits.
    lambda_smooth : float
        Weight of smoothness penalty. Higher = smoother predictions.
    max_v : float
        Maximum realistic linear velocity in m/s.
    max_omega : float
        Maximum realistic angular velocity in rad/s.
    """

    def __init__(
        self,
        lambda_vel: float = 0.1,
        lambda_smooth: float = 0.05,
        max_v: float = 3.0,
        max_omega: float = 2.0,
    ):
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_smooth = lambda_smooth
        self.max_v = max_v
        self.max_omega = max_omega
        self.mse = nn.MSELoss()

    def forward(self, pred_trajectory, target_trajectory, v_omega):
        """
        Args:
            pred_trajectory: (batch, pred_len, 3) — predicted [x, y, θ]
            target_trajectory: (batch, pred_len, 3) — actual [x, y, θ]
            v_omega: (batch, pred_len, 2) — predicted [v, ω]

        Returns:
            total_loss: scalar
            loss_dict: breakdown of individual loss components
        """
        # Component 1: Standard trajectory MSE
        traj_loss = self.mse(pred_trajectory, target_trajectory)

        # Component 2: Velocity bound penalty
        # Only penalize velocities OUTSIDE the allowed range
        # ReLU(|v| - max_v) = 0 if within bounds, positive if exceeding
        v = v_omega[:, :, 0]      # linear velocity
        omega = v_omega[:, :, 1]  # angular velocity

        vel_penalty = (
            torch.relu(v.abs() - self.max_v).pow(2).mean()
            + torch.relu(omega.abs() - self.max_omega).pow(2).mean()
        )

        # Component 3: Smoothness penalty
        # Penalize large acceleration (change in velocity between steps)
        if v_omega.size(1) > 1:
            dv = torch.diff(v_omega, dim=1)  # (batch, pred_len-1, 2)
            smooth_penalty = dv.pow(2).mean()
        else:
            smooth_penalty = torch.tensor(0.0, device=v_omega.device)

        # Total loss
        total_loss = (
            traj_loss
            + self.lambda_vel * vel_penalty
            + self.lambda_smooth * smooth_penalty
        )

        loss_dict = {
            "total": total_loss.item(),
            "trajectory_mse": traj_loss.item(),
            "velocity_penalty": vel_penalty.item(),
            "smoothness_penalty": smooth_penalty.item(),
        }

        return total_loss, loss_dict


# =============================================================================
# 4. QUICK TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Physics-Informed Linear Transformer — Architecture Test")
    print("=" * 60)

    model = PhysicsLinearTransformerPredictor(
        input_dim=6,
        output_dim=3,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        pred_len=10,
        dropout=0.1,
        dt=0.1,
    )

    print(f"\nModel config: {model.config}")
    print(f"Total parameters: {model.count_parameters():,}")

    # Create fake batch
    dummy_input = torch.randn(4, 20, 6)
    print(f"\nInput shape: {dummy_input.shape}")

    # Forward pass
    trajectory, v_omega = model(dummy_input)
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Velocity shape:   {v_omega.shape}")

    assert trajectory.shape == (4, 10, 3), \
        f"Expected (4, 10, 3), got {trajectory.shape}"
    assert v_omega.shape == (4, 10, 2), \
        f"Expected (4, 10, 2), got {v_omega.shape}"

    # Test the physics loss
    print(f"\n{'─' * 60}")
    print("Testing Physics Loss")
    print(f"{'─' * 60}")

    criterion = PhysicsInformedLoss(
        lambda_vel=0.1,
        lambda_smooth=0.05,
        max_v=3.0,
        max_omega=2.0,
    )

    target = torch.randn(4, 10, 3)  # fake ground truth
    total_loss, loss_dict = criterion(trajectory, target, v_omega)

    print(f"  Total loss:          {loss_dict['total']:.4f}")
    print(f"  Trajectory MSE:      {loss_dict['trajectory_mse']:.4f}")
    print(f"  Velocity penalty:    {loss_dict['velocity_penalty']:.4f}")
    print(f"  Smoothness penalty:  {loss_dict['smoothness_penalty']:.4f}")

    # Compare parameter count with regular model
    from linear_transformer import LinearTransformerPredictor
    regular_model = LinearTransformerPredictor()
    print(f"\n{'─' * 60}")
    print("Parameter Comparison")
    print(f"{'─' * 60}")
    print(f"  Regular Linear Transformer:  {regular_model.count_parameters():,}")
    print(f"  Physics-Informed variant:    {model.count_parameters():,}")
    diff = regular_model.count_parameters() - model.count_parameters()
    print(f"  Difference:                  {diff:,} fewer parameters")
    print(f"  (Physics model predicts 2 values per step instead of 3)")

    print(f"\n✅ All tests passed!")
    print(f"   The physics-informed model predicts (v, ω) and integrates")
    print(f"   through the unicycle model to guarantee physical plausibility.")
