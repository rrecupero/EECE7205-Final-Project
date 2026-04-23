"""
Linear Transformer for Robot Trajectory Prediction
====================================================
Your contribution to the EECE 7205 project.

Architecture overview:
    Input (20 steps × 6 features)
        ↓
    [Input Embedding]     — project 6 → d_model dimensions
        ↓
    [Positional Encoding] — add time-position information
        ↓
    [Linear Attention × N layers] — the core: O(T) attention
        ↓
    [Output Projection]   — project to 10 steps × 3 features
        ↓
    Output (10 steps × 3 features)

Key reference: Katharopoulos et al., "Transformers are RNNs" (ICML 2020)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. FEATURE MAP φ(x) — The heart of Linear Attention
# =============================================================================
class ELUFeatureMap(nn.Module):
    """
    The feature map φ(x) = elu(x) + 1 used in linear attention.

    WHY DO WE NEED THIS?
    ---------------------
    In standard attention, softmax guarantees that attention weights are
    positive and sum to 1 (a valid probability distribution). When we
    remove softmax, we need another way to ensure the attention weights
    behave reasonably.

    The ELU+1 feature map does this:
    - elu(x) + 1 is ALWAYS positive (elu bottoms out at -1, so +1 makes it ≥ 0)
    - Positive feature maps guarantee positive attention weights
    - It's smooth and differentiable (good for gradient-based training)

    The formula:
        elu(x) = x           if x > 0
                 α(eˣ - 1)   if x ≤ 0     (where α = 1.0)

    So elu(x) + 1:
        = x + 1              if x > 0    (linearly grows)
        = eˣ                 if x ≤ 0    (exponentially approaches 0, never negative)

    OTHER OPTIONS:
    - Random Fourier Features (RFF): better theoretical approximation of
      softmax, but more complex and slower
    - ReLU: simpler but has dead neurons (output=0 kills gradients)
    - We use ELU+1 because it's the standard choice from the original
      Katharopoulos et al. paper and works well in practice.
    """

    def forward(self, x):
        return F.elu(x) + 1


# =============================================================================
# 2. LINEAR ATTENTION — O(T) instead of O(T²)
# =============================================================================
class LinearAttention(nn.Module):
    """
    Linear attention mechanism with O(T) complexity.

    STANDARD ATTENTION (what your teammate's Transformer uses):
    -----------------------------------------------------------
    Attention(Q, K, V) = softmax(Q @ Kᵀ / √d_k) @ V

    Step by step:
    1. Q @ Kᵀ produces a (T × T) attention matrix     → O(T² · d)
    2. softmax normalizes each row                      → O(T²)
    3. Multiply by V to get output                      → O(T² · d)
    Total: O(T² · d) — quadratic in sequence length T

    For T=20 (your current setup), T²=400. Not bad.
    For T=200 (longer sequences), T²=40,000. Getting expensive.
    For T=2000 (very long trajectories), T²=4,000,000. Impractical.

    LINEAR ATTENTION (what you're building):
    ----------------------------------------
    Attention(Q, K, V) = φ(Q) @ (φ(K)ᵀ @ V) / (φ(Q) @ (φ(K)ᵀ @ 1))

    The KEY INSIGHT is the order of operations (associativity trick):

    Standard:  (φ(Q) @ φ(K)ᵀ) @ V    → first compute T×T matrix, then multiply by V
    Linear:     φ(Q) @ (φ(K)ᵀ @ V)   → first compute d×d matrix, then multiply by Q

    Why this changes complexity:
    - φ(K)ᵀ @ V is shape (d × d), independent of T     → O(T · d²)
    - φ(Q) @ result is shape (T × d)                    → O(T · d²)
    Total: O(T · d²) — LINEAR in sequence length T!

    Since d (model dimension) is fixed (e.g., 64), this is dramatically
    faster for long sequences. The 73% speedup in your abstract comes
    from this exact difference.

    THE CATCH:
    ----------
    Standard softmax attention can create SHARP attention patterns
    (strongly focus on one or two timesteps). Linear attention can only
    create SMOOTH attention patterns because the kernel approximation
    can't represent sharp peaks as well. This is why Linear Transformers
    sometimes lose a bit of accuracy — they're "blurrier" in which
    past timesteps they attend to.

    For trajectory prediction, this is usually fine because robot motion
    is smooth — you don't need to sharply attend to one specific past
    moment. The motion dynamics are gradual.

    Parameters
    ----------
    d_model : int
        The internal dimension of the model (e.g., 64)
    n_heads : int
        Number of attention heads. Each head independently learns
        different aspects of the temporal relationships.
        d_model must be divisible by n_heads.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # dimension per head

        # These linear layers project the input into Q, K, V
        # Q = "what am I looking for?"
        # K = "what do I contain?"
        # V = "what information do I provide?"
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final projection after combining all heads
        self.W_out = nn.Linear(d_model, d_model)

        # The feature map φ
        self.feature_map = ELUFeatureMap()

    def forward(self, x, causal=True):
        """
        Args:
            x: input tensor of shape (batch, seq_len, d_model)
            causal: if True, each position can only attend to past positions.
                    This is important for prediction — the model shouldn't
                    "cheat" by looking at future timesteps.

        Returns:
            output: tensor of shape (batch, seq_len, d_model)
        """
        B, T, _ = x.shape
        H = self.n_heads
        D = self.d_head

        # Project input to Q, K, V and reshape for multi-head attention
        # Shape: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_head)
        Q = self.W_q(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = self.W_k(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        V = self.W_v(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

        # Apply feature map: φ(Q), φ(K)
        # This is what makes it "linear" — we replace softmax with φ
        Q = self.feature_map(Q)  # (B, H, T, D), all values ≥ 0
        K = self.feature_map(K)  # (B, H, T, D), all values ≥ 0

        if causal:
            # CAUSAL linear attention: each position only sees past + current
            # We must compute this step-by-step (can't use the matrix trick
            # directly because future keys must be excluded)
            output = self._causal_linear_attention(Q, K, V)
        else:
            # NON-CAUSAL: every position attends to every other position
            # This is where the associativity trick shines
            output = self._noncausal_linear_attention(Q, K, V)

        # Reshape back: (B, H, T, D) → (B, T, H*D) = (B, T, d_model)
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Final linear projection
        return self.W_out(output)

    def _noncausal_linear_attention(self, Q, K, V):
        """
        Non-causal linear attention using the associativity trick.

        Instead of: (Q @ Kᵀ) @ V     → T×T intermediate, O(T²)
        We compute: Q @ (Kᵀ @ V)     → D×D intermediate, O(T·D²)

        The denominator normalizes so attention weights "sum to 1":
            output_i = (φ(Q_i) @ S) / (φ(Q_i) @ z)
        where:
            S = Σⱼ φ(K_j)ᵀ ⊗ V_j     (the "summary" matrix, shape D×D)
            z = Σⱼ φ(K_j)             (normalization vector, shape D)
        """
        # S = φ(K)ᵀ @ V — the "summary" of all key-value pairs
        # Shape: (B, H, D, D)
        KV = torch.einsum("bhnd,bhnm->bhdm", K, V)

        # z = sum of all φ(K) — for normalization
        # Shape: (B, H, D)
        z = K.sum(dim=2)

        # Numerator: Q @ S → (B, H, T, D)
        numerator = torch.einsum("bhnd,bhdm->bhnm", Q, KV)

        # Denominator: Q @ z → (B, H, T, 1)
        denominator = torch.einsum("bhnd,bhd->bhn", Q, z).unsqueeze(-1)

        # Avoid division by zero
        denominator = denominator.clamp(min=1e-6)

        return numerator / denominator

    def _causal_linear_attention(self, Q, K, V):
        """
        Causal linear attention — each position only attends to past.

        For trajectory prediction, this is essential. When predicting the
        future, the model at timestep t should only use information from
        timesteps 0, 1, ..., t (not t+1, t+2, ...).

        We use a cumulative sum approach:
        At each timestep t, we maintain a running "summary" S_t that
        accumulates all past key-value pairs:
            S_t = Σⱼ₌₀ᵗ φ(K_j)ᵀ ⊗ V_j

        Then the output at timestep t is:
            output_t = φ(Q_t) @ S_t / (φ(Q_t) @ z_t)

        We compute S_t efficiently using cumulative sums (cumsum).

        This is still O(T) because at each step we just ADD one more
        key-value pair to the running sum, rather than recomputing
        attention over all past positions from scratch.
        """
        B, H, T, D = Q.shape

        # Build KV pairs at each timestep: φ(K_t) ⊗ V_t
        # Shape: (B, H, T, D, D) — at each timestep, a D×D outer product
        KV = torch.einsum("bhnd,bhnm->bhndm", K, V)

        # Cumulative sum along time axis gives S_t = Σⱼ₌₀ᵗ KV_j
        # Shape: (B, H, T, D, D)
        S = KV.cumsum(dim=2)

        # Cumulative sum of keys for normalization: z_t = Σⱼ₌₀ᵗ φ(K_j)
        # Shape: (B, H, T, D)
        z = K.cumsum(dim=2)

        # Output at each timestep: φ(Q_t) @ S_t
        # Shape: (B, H, T, D)
        numerator = torch.einsum("bhnd,bhndm->bhnm", Q, S)

        # Normalization: φ(Q_t) · z_t
        # Shape: (B, H, T, 1)
        denominator = torch.einsum("bhnd,bhnd->bhn", Q, z).unsqueeze(-1)
        denominator = denominator.clamp(min=1e-6)

        return numerator / denominator


# =============================================================================
# 3. TRANSFORMER ENCODER LAYER (with Linear Attention)
# =============================================================================
class LinearTransformerLayer(nn.Module):
    """
    One layer of the Linear Transformer encoder.

    Each layer has two sub-components:

    1. Linear Attention — lets each timestep gather information from
       other timesteps (temporal reasoning)

    2. Feed-Forward Network (FFN) — processes each timestep independently
       through two linear layers with a ReLU in between. This adds
       non-linear transformation capacity. Think of attention as
       "communication between timesteps" and FFN as "processing
       within each timestep."

    Both sub-components use RESIDUAL CONNECTIONS and LAYER NORMALIZATION:

    Residual connection: output = input + sublayer(input)
        Why? Lets gradients flow directly through the network during
        backpropagation. Without this, deep networks (many layers)
        have vanishing gradients and fail to train.

    Layer normalization: normalize the values at each layer
        Why? Keeps the magnitudes stable across layers. Without this,
        values can explode or vanish as they pass through many layers.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model:  internal dimension (e.g., 64)
            n_heads:  number of attention heads (e.g., 4)
            d_ff:     feed-forward hidden dimension (typically 2-4x d_model)
            dropout:  randomly zero out this fraction of values during
                      training to prevent overfitting
        """
        super().__init__()

        # Linear attention sub-layer
        self.attention = LinearAttention(d_model, n_heads)

        # Feed-forward network: two linear layers with ReLU
        # d_model → d_ff → d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # Layer norms (one for each sub-layer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass with Pre-Norm architecture:
            x → LayerNorm → Attention → Dropout → + residual
            → LayerNorm → FFN → Dropout → + residual

        Pre-Norm (normalize BEFORE the sublayer) is more stable than
        Post-Norm (normalize AFTER) for training deeper models.
        """
        # Sub-layer 1: Linear Attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.attention(x, causal=True)
        x = self.dropout1(x)
        x = residual + x

        # Sub-layer 2: Feed-Forward Network with residual connection
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x

        return x


# =============================================================================
# 4. POSITIONAL ENCODING — Tell the model about time order
# =============================================================================
class SinusoidalPositionalEncoding(nn.Module):
    """
    Add sinusoidal position information to the input embeddings.

    WHY IS THIS NEEDED?
    --------------------
    Attention treats its input as a SET, not a SEQUENCE. If you shuffle
    the timesteps, the attention output is the same (just shuffled).
    But for trajectory prediction, ORDER MATTERS — the robot was at
    position A BEFORE position B. Positional encoding injects this
    ordering information.

    HOW IT WORKS:
    -------------
    For each position t and dimension i, we add:
        PE(t, 2i)   = sin(t / 10000^(2i/d_model))
        PE(t, 2i+1) = cos(t / 10000^(2i/d_model))

    Different dimensions oscillate at different frequencies:
    - Low dimensions (i ≈ 0): high frequency, captures fine time differences
    - High dimensions (i ≈ d_model): low frequency, captures coarse time structure

    This is the same encoding used in the original "Attention is All You Need"
    paper (Vaswani et al., 2017). It's deterministic (no learned parameters)
    and can generalize to sequence lengths not seen during training.
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        # Pre-compute the positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # The division term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions

        # Register as buffer (not a trainable parameter, but moves to GPU with model)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: shape (batch, seq_len, d_model)
        Returns:
            x + positional encoding (same shape)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


# =============================================================================
# 5. FULL LINEAR TRANSFORMER MODEL
# =============================================================================
class LinearTransformerPredictor(nn.Module):
    """
    Complete Linear Transformer for robot trajectory prediction.

    Data flow:
        Raw input (B, 20, 6)
            ↓
        Input embedding: Linear(6 → d_model)         → (B, 20, d_model)
            ↓
        Add positional encoding                       → (B, 20, d_model)
            ↓
        N × LinearTransformerLayer                    → (B, 20, d_model)
            ↓
        Take last timestep's representation           → (B, d_model)
            ↓
        Output projection: Linear(d_model → pred_len × 3)  → (B, 30)
            ↓
        Reshape                                       → (B, 10, 3)

    Why "take last timestep"?
    -------------------------
    After the attention layers process all 20 input timesteps, the
    representation at the LAST timestep has had the chance to attend
    to ALL previous timesteps (via causal attention). It therefore
    contains a "summary" of the entire observed trajectory. We use
    this summary to predict the future.

    An alternative is to use ALL timestep representations and project
    them, but the "last timestep" approach is simpler and standard.
    """

    def __init__(
        self,
        input_dim: int = 6,       # number of input features per timestep
        output_dim: int = 3,      # number of output features per timestep
        d_model: int = 64,        # internal model dimension
        n_heads: int = 4,         # number of attention heads
        n_layers: int = 3,        # number of transformer layers
        d_ff: int = 128,          # feed-forward hidden dimension
        pred_len: int = 10,       # number of future timesteps to predict
        dropout: float = 0.1,     # dropout rate
    ):
        super().__init__()

        self.pred_len = pred_len
        self.output_dim = output_dim

        # ── Input embedding: project raw features to model dimension ──
        # Why not feed 6 features directly into attention?
        # Attention works better in higher dimensions because Q, K, V
        # projections need enough "room" to encode different aspects
        # of the input. 6 dimensions is too cramped.
        self.input_embed = nn.Linear(input_dim, d_model)

        # ── Positional encoding ──
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=40000)

        # ── Stack of Linear Transformer layers ──
        # More layers = more capacity to learn complex patterns,
        # but also more parameters and slower training.
        # 3 layers is a good starting point for this problem size.
        self.layers = nn.ModuleList([
            LinearTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # ── Final layer norm ──
        self.final_norm = nn.LayerNorm(d_model)

        # ── Output projection: model dimension → prediction ──
        # Maps d_model → pred_len * output_dim
        # e.g., 64 → 10 * 3 = 30, then reshape to (10, 3)
        self.output_proj = nn.Linear(d_model, pred_len * output_dim)

        # Store hyperparameters for reference
        self.config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "pred_len": pred_len,
            "dropout": dropout,
        }

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch, observe_len, input_dim)
               e.g., (32, 20, 6) — a batch of 32 windows, each with
               20 timesteps of 6 features

        Returns:
            predictions: shape (batch, pred_len, output_dim)
               e.g., (32, 10, 3) — for each window in the batch,
               10 predicted future timesteps with 3 features each
        """
        B = x.size(0)

        # Step 1: Embed input features into model dimension
        # (B, 20, 6) → (B, 20, 64)
        x = self.input_embed(x)

        # Step 2: Add positional encoding
        # (B, 20, 64) → (B, 20, 64) (values change, shape stays same)
        x = self.pos_encoding(x)

        # Step 3: Pass through all Linear Transformer layers
        # Each layer refines the representations using attention + FFN
        # (B, 20, 64) → (B, 20, 64) through each layer
        for layer in self.layers:
            x = layer(x)

        # Step 4: Final layer norm
        x = self.final_norm(x)

        # Step 5: Take the last timestep's representation
        # The last position has attended to all previous positions,
        # so it contains a compressed summary of the full trajectory
        # (B, 20, 64) → (B, 64)
        x = x[:, -1, :]

        # Step 6: Project to prediction
        # (B, 64) → (B, 30) → (B, 10, 3)
        x = self.output_proj(x)
        x = x.view(B, self.pred_len, self.output_dim)

        return x

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 6. QUICK TEST — Verify shapes and forward pass work
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Linear Transformer — Architecture Test")
    print("=" * 60)

    # Create model with default hyperparameters
    model = LinearTransformerPredictor(
        input_dim=6,
        output_dim=3,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        pred_len=10,
        dropout=0.1,
    )

    print(f"\nModel config: {model.config}")
    print(f"Total parameters: {model.count_parameters():,}")

    # Create a fake batch to test the forward pass
    # Shape: (batch=4, observe_len=20, input_features=6)
    dummy_input = torch.randn(4, 20, 6)
    print(f"\nInput shape:  {dummy_input.shape}")

    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Verify output shape matches expectations
    assert output.shape == (4, 10, 3), \
        f"Expected (4, 10, 3), got {output.shape}"

    print(f"\n✅ Forward pass successful!")
    print(f"   Input:  (batch=4, observe=20, features=6)")
    print(f"   Output: (batch=4, predict=10, features=3)")

    # Print model architecture
    print(f"\n{'─' * 60}")
    print("Model Architecture:")
    print(f"{'─' * 60}")
    print(model)
