"""Actor-critic networks for the substitution (ACS) environment.

Both architectures are "dual ring" actor-critics: they split the presentation
into its two relators ("rings"), run a small transformer with self-attention
within each ring and cross-attention between them, and emit a Categorical
policy over the packed S-move action index plus a scalar value.

- ``RelativeDualRingActorCritic`` — relative-position attention, treating each
  relator as cyclic. This is the architecture used by ``ppo_ac_s.py``.
- ``DualRingActorCritic`` — absolute (fixed sinusoidal) positional encoding.
  Kept here as an alternative.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import distrax
from flax.linen.initializers import orthogonal, constant

jax.config.update("jax_default_matmul_precision", "float32")


def _stable_change_of_variables_mask(mask1, mask2):
    """Mask finite change-of-variables windows with nonempty complements."""
    B, L = mask1.shape
    lengths = jnp.stack([mask1.sum(axis=1), mask2.sum(axis=1)], axis=1)
    starts = jnp.arange(L)[None, :, None]
    z_lens = (jnp.arange(L) + 1)[None, None, :]
    channel_masks = []
    for branch in range(4):
        iso_relator = branch % 2
        rel_len = lengths[:, iso_relator][:, None, None]
        valid = (starts < rel_len) & (z_lens < rel_len)
        channel_masks.extend([valid, valid])
    return jnp.stack(channel_masks, axis=-1)


# ---------------------------------------------------------------------------
# Relative-position attention (cyclic) building blocks
# ---------------------------------------------------------------------------
class RelativeSelfAttention(nn.Module):
    num_heads: int
    head_dim: int
    max_len: int  # fixed sequence length L

    def setup(self):
        # Precompute static cyclic distance base matrix [L,L]
        idxs = jnp.arange(self.max_len)
        self.base_rel_dist = (idxs[None, :] - idxs[:, None])  # [L,L]

    @nn.compact
    def __call__(self, x, mask):
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        assert L == self.max_len, f"Expected length {self.max_len}, got {L}"

        # === QKV projections ===
        qkv = nn.Dense(3 * H * Dh)(x).reshape(B, L, 3, H, Dh)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.squeeze(2).transpose(0, 2, 1, 3)  # [B,H,L,Dh]
        k = k.squeeze(2).transpose(0, 2, 1, 3)
        v = v.squeeze(2).transpose(0, 2, 1, 3)

        # === Base attention scores ===
        content_scores = jnp.einsum("bhid,bhjd->bhij", q, k)  # [B,H,L,L]

        # === Relative position embeddings per head ===
        # Learnable embedding: [H,L,Dh]
        rel_emb = self.param("rel_emb", nn.initializers.normal(stddev=0.02), (H, L, Dh))

        # Compute effective lengths per batch
        lengths = jnp.maximum(mask.sum(axis=1), 1)  # [B]

        # Compute cyclic distances for each batch: [B,L,L]
        rel_dist = self.base_rel_dist[None, :, :] % lengths[:, None, None]

        # Gather embeddings for each batch & head
        # rel_emb: [H,L,Dh] -> [1,H,1,L,Dh] to prepare for gather
        rel_emb_exp = rel_emb[None, :, None, :, :]  # [1,H,1,L,Dh]
        rel_dist_exp = rel_dist[:, None, :, :, None]  # [B,1,L,L,1]

        # Gather: [B,H,L,L,Dh]
        rel_emb_full = jnp.take_along_axis(rel_emb_exp, rel_dist_exp, axis=3)

        # Compute relative scores: [B,H,L,L]
        rel_scores = jnp.einsum("bhid,bhijd->bhij", q, rel_emb_full)

        # Combine content and relative scores
        scores = content_scores + rel_scores

        # Apply mask
        scores = jnp.where(mask[:, None, None, :], scores, -1e9)

        # === Attention ===
        attn = nn.softmax(scores / jnp.sqrt(Dh), axis=-1)
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, H * Dh)

        return nn.Dense(D)(out)


class RelativeCrossAttention(nn.Module):
    num_heads: int
    head_dim: int
    max_len: int

    @nn.compact
    def __call__(self, q_seq, kv_seq, q_mask=None, kv_mask=None):
        B, Lq, D = q_seq.shape
        Lv = kv_seq.shape[1]
        H = self.num_heads
        D_head = self.head_dim

        # projections
        q = nn.Dense(H * D_head)(q_seq).reshape(B, Lq, H, D_head).transpose(0, 2, 1, 3)
        kv = nn.Dense(2 * H * D_head)(kv_seq).reshape(B, Lv, 2, H, D_head)
        k, v = jnp.split(kv, 2, axis=2)
        k = k.squeeze(2).transpose(0, 2, 1, 3)
        v = v.squeeze(2).transpose(0, 2, 1, 3)

        # dot product attention only (no relative bias here)
        scores = jnp.einsum("bhid,bhjd->bhij", q, k)

        if kv_mask is not None:
            scores = jnp.where(kv_mask[:, None, None, :], scores, -1e9)
        if q_mask is not None:
            scores = jnp.where(q_mask[:, None, :, None], scores, -1e9)

        attn = nn.softmax(scores / jnp.sqrt(D_head), axis=-1)
        out = jnp.einsum("bhij,bhjd->bhid", attn, v).transpose(0, 2, 1, 3).reshape(B, Lq, H * D_head)
        return nn.Dense(D)(out)


class RelativeDualRingBlock(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    max_len: int

    @nn.compact
    def __call__(self, x1, x2, mask1, mask2):
        # Normalize inputs first
        x1_norm = nn.LayerNorm()(x1)
        x2_norm = nn.LayerNorm()(x2)

        # Self-attention (residual)
        sa1 = RelativeSelfAttention(self.num_heads, self.head_dim, self.max_len)(x1_norm, mask1)
        sa2 = RelativeSelfAttention(self.num_heads, self.head_dim, self.max_len)(x2_norm, mask2)
        x1 = x1 + sa1
        x2 = x2 + sa2

        # Normalize after self-attention for cross-attention inputs
        x1_norm2 = nn.LayerNorm()(x1)
        x2_norm2 = nn.LayerNorm()(x2)

        # Cross-attention (residual), symmetrical treatment
        ca1 = RelativeCrossAttention(self.num_heads, self.head_dim, self.max_len)(
            x1_norm2, x2_norm2, q_mask=mask1, kv_mask=mask2
        )
        ca2 = RelativeCrossAttention(self.num_heads, self.head_dim, self.max_len)(
            x2_norm2, x1_norm2, q_mask=mask2, kv_mask=mask1
        )
        x1 = x1 + ca1
        x2 = x2 + ca2

        # MLP block with residual connection and LayerNorm inside
        def mlp_block(x):
            residual = x
            x_norm = nn.LayerNorm()(x)
            x = nn.Dense(self.mlp_dim)(x_norm)
            x = nn.gelu(x)
            x = nn.Dense(residual.shape[-1])(x)
            return x + residual

        x1 = mlp_block(x1)
        x2 = mlp_block(x2)

        return x1, x2


class RelativeDualRingActorCritic(nn.Module):
    activation: str = "tanh"
    num_layers: int = 2
    num_heads: int = 4
    head_dim: int = 8
    mlp_dim: int = 32
    embedding_dim: int = num_heads * head_dim
    vocab_size: int = 5
    max_len: int = 24
    stable_ac_moves: bool = False
    change_of_variables_moves: bool = False
    ac45_moves: bool = False

    @nn.compact
    def __call__(self, input_seq):
        B, L = input_seq.shape
        L_half = L // 2

        # Split rings and build masks
        r1_raw, r2_raw = input_seq[:, :L_half], input_seq[:, L_half:]
        mask1 = (r1_raw != 0)  # [B, L_half]
        mask2 = (r2_raw != 0)  # [B, L_half]

        # Semantic mask for (i,j)
        r1_exp = r1_raw[:, :, None]  # [B, L_half, 1]
        r2_exp = r2_raw[:, None, :]  # [B, 1, L_half]
        r1_broadcast = jnp.broadcast_to(r1_exp, (B, L_half, L_half))
        r2_broadcast = jnp.broadcast_to(r2_exp, (B, L_half, L_half))

        mask_j0 = (r1_broadcast == -r2_broadcast)
        mask_j1 = (r1_broadcast ==  r2_broadcast)
        semantic_mask = jnp.stack([mask_j0, mask_j1], axis=-1)  # [B, L_half, L_half, 2]

        # Padding masks for i and j positions
        mask1_b = mask1[:, :, None]  # [B, L_half, 1]
        mask2_b = mask2[:, None, :]  # [B, 1, L_half]
        padding_mask = mask1_b & mask2_b  # [B, L_half, L_half]

        # Combine semantic mask with padding mask to get valid (i,j) actions per type
        base_mask = semantic_mask & padding_mask[..., None]  # [B, L_half, L_half, 2]
        final_mask = jnp.broadcast_to(base_mask[:,:,:,None,:], (B,L_half,L_half,2,2))
        final_mask = final_mask.reshape(B, -1)  # [B, total_action_dim]

        # === Embeddings ===
        embed = nn.Embed(self.vocab_size, self.embedding_dim, name="shared_embed")
        x1 = embed(r1_raw.astype(jnp.int32) + 2)  # [B, L_half, D]
        x2 = embed(r2_raw.astype(jnp.int32) + 2)  # [B, L_half, D]

        # Transformer blocks
        for _ in range(self.num_layers):
            x1, x2 = RelativeDualRingBlock(
                self.num_heads, self.head_dim, self.mlp_dim, self.max_len
            )(x1, x2, mask1, mask2)

        # === Value head ===
        def masked_mean(x, mask):
            mask = mask.astype(jnp.float32)
            return (x * mask[:, :, None]).sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1e-6)

        pooled1 = masked_mean(x1, mask1)
        pooled2 = masked_mean(x2, mask2)
        joint = jnp.concatenate([pooled1, pooled2], axis=-1)

        act_fn = nn.gelu if self.activation == "gelu" else nn.tanh

        critic = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)))(joint)
        critic = act_fn(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)))(critic)
        critic = act_fn(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0))(critic)

        # === Actor head ===
        x1_exp = jnp.expand_dims(x1, axis=2)  # [B, L_half, 1, D]
        x2_exp = jnp.expand_dims(x2, axis=1)  # [B, 1, L_half, D]

        x1_b = jnp.broadcast_to(x1_exp, (B, L_half, L_half, x1.shape[-1]))
        x2_b = jnp.broadcast_to(x2_exp, (B, L_half, L_half, x2.shape[-1]))

        x_joint = jnp.concatenate([x1_b, x2_b], axis=-1)  # [B, L_half, L_half, 2D]

        x_joint = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)))(x_joint)
        x_joint = act_fn(x_joint)

        # Output logits for substitution actions (j_type * k1 * k2)
        logits = nn.Dense(4, kernel_init=orthogonal(0.01))(x_joint)  # [B, L_half, L_half, 4] k1, k2, ij

        # flatten to [B, total_action_dim]; sample = ((k1 * L_half + k2) * 4) + (i * 2 + j), so j is the last index
        logits_flat = logits.reshape(B, -1)

        # Mask invalid logits with -1e9
        logits_flat = jnp.where(final_mask.reshape(B, -1), logits_flat, -1e9)

        cov_enabled = self.change_of_variables_moves or self.stable_ac_moves
        ac45_enabled = self.ac45_moves or self.stable_ac_moves
        if cov_enabled:
            stable_logits = nn.Dense(8, kernel_init=orthogonal(0.01))(x_joint)
            stable_mask = _stable_change_of_variables_mask(mask1, mask2)
            stable_logits = jnp.where(stable_mask, stable_logits, -1e9)
            logits_flat = jnp.concatenate(
                [logits_flat, stable_logits.reshape(B, -1)],
                axis=-1,
            )
        if ac45_enabled:
            generator_logits = nn.Dense(3, kernel_init=orthogonal(0.01))(joint)
            logits_flat = jnp.concatenate([logits_flat, generator_logits], axis=-1)

        # Distribution over all actions (i, j, j_type, k1, k2)
        pi = distrax.Categorical(logits=logits_flat)

        return pi, jnp.squeeze(critic, axis=-1)


# ---------------------------------------------------------------------------
# Absolute (fixed sinusoidal) positional encoding building blocks
# ---------------------------------------------------------------------------
def fixed_positional_encoding(length, dim):
    position = jnp.arange(length)[:, None]
    div_term = jnp.exp(jnp.arange(0, dim, 2) * -(jnp.log(10000.0) / dim))
    pe = jnp.zeros((length, dim))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe


class AbsoluteSelfAttention(nn.Module):
    num_heads: int
    head_dim: int
    max_len: int

    @nn.compact
    def __call__(self, x, mask):
        B, L, D = x.shape
        H = self.num_heads
        D_head = self.head_dim
        x = x + fixed_positional_encoding(L, D)

        qkv = nn.Dense(3 * H * D_head)(x).reshape(B, L, 3, H, D_head)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2).transpose(0, 2, 1, 3), k.squeeze(2).transpose(0, 2, 1, 3), v.squeeze(2).transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhid,bhjd->bhij", q, k)

        if mask is not None:
            scores = jnp.where(mask[:, None, None, :], scores, -1e9)

        attn = nn.softmax(scores / jnp.sqrt(D_head), axis=-1)
        out = jnp.einsum("bhij,bhjd->bhid", attn, v).transpose(0, 2, 1, 3).reshape(B, L, H * D_head)

        return nn.Dense(D)(out)


class AbsoluteCrossAttention(nn.Module):
    num_heads: int
    head_dim: int
    max_len: int

    @nn.compact
    def __call__(self, q_seq, kv_seq, q_mask=None, kv_mask=None):
        B, Lq, D = q_seq.shape
        Lv = kv_seq.shape[1]
        H = self.num_heads
        D_head = self.head_dim

        # Add fixed sinusoidal positional encodings
        q_seq = q_seq + fixed_positional_encoding(Lq, D)
        kv_seq = kv_seq + fixed_positional_encoding(Lv, D)

        q = nn.Dense(H * D_head)(q_seq).reshape(B, Lq, H, D_head).transpose(0, 2, 1, 3)
        kv = nn.Dense(2 * H * D_head)(kv_seq)  # One layer, two outputs
        kv = kv.reshape(B, Lv, 2, H, D_head)
        k, v = jnp.split(kv, 2, axis=2)  # Split along the 2-axis
        k = k.squeeze(2).transpose(0, 2, 1, 3)
        v = v.squeeze(2).transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhid,bhjd->bhij", q, k)

        if kv_mask is not None:
            scores = jnp.where(kv_mask[:, None, None, :], scores, -1e9)
        if q_mask is not None:
            scores = jnp.where(q_mask[:, None, :, None], scores, -1e9)

        attn = nn.softmax(scores / jnp.sqrt(D_head), axis=-1)
        out = jnp.einsum("bhij,bhjd->bhid", attn, v).transpose(0, 2, 1, 3).reshape(B, Lq, H * D_head)
        return nn.Dense(D)(out)


class AbsoluteDualRingBlock(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    max_len: int

    @nn.compact
    def __call__(self, x1, x2, mask1, mask2):
        # Normalize inputs first
        x1_norm = nn.LayerNorm()(x1)
        x2_norm = nn.LayerNorm()(x2)

        # Self-attention (residual)
        sa1 = AbsoluteSelfAttention(self.num_heads, self.head_dim, self.max_len)(x1_norm, mask1)
        sa2 = AbsoluteSelfAttention(self.num_heads, self.head_dim, self.max_len)(x2_norm, mask2)
        x1 = x1 + sa1
        x2 = x2 + sa2

        # Normalize after self-attention for cross-attention inputs
        x1_norm2 = nn.LayerNorm()(x1)
        x2_norm2 = nn.LayerNorm()(x2)

        # Cross-attention (residual), symmetrical treatment
        ca1 = AbsoluteCrossAttention(self.num_heads, self.head_dim, self.max_len)(
            x1_norm2, x2_norm2, q_mask=mask1, kv_mask=mask2
        )
        ca2 = AbsoluteCrossAttention(self.num_heads, self.head_dim, self.max_len)(
            x2_norm2, x1_norm2, q_mask=mask2, kv_mask=mask1
        )
        x1 = x1 + ca1
        x2 = x2 + ca2

        # MLP block with residual connection and LayerNorm inside
        def mlp_block(x):
            residual = x
            x_norm = nn.LayerNorm()(x)
            x = nn.Dense(self.mlp_dim)(x_norm)
            x = nn.gelu(x)
            x = nn.Dense(residual.shape[-1])(x)
            return x + residual

        x1 = mlp_block(x1)
        x2 = mlp_block(x2)

        return x1, x2


class DualRingActorCritic(nn.Module):
    activation: str = "tanh"
    num_layers: int = 2
    num_heads: int = 4
    head_dim: int = 8
    mlp_dim: int = 32
    embedding_dim: int = num_heads * head_dim
    vocab_size: int = 5
    max_len: int = 32
    stable_ac_moves: bool = False
    change_of_variables_moves: bool = False
    ac45_moves: bool = False

    @nn.compact
    def __call__(self, input_seq):
        B, L = input_seq.shape
        L_half = L // 2

        # Split rings and build masks
        r1_raw, r2_raw = input_seq[:, :L_half], input_seq[:, L_half:]
        mask1 = (r1_raw != 0)  # [B, L_half]
        mask2 = (r2_raw != 0)  # [B, L_half]

        # Semantic mask for (i,j)
        r1_exp = r1_raw[:, :, None]  # [B, L_half, 1]
        r2_exp = r2_raw[:, None, :]  # [B, 1, L_half]
        r1_broadcast = jnp.broadcast_to(r1_exp, (B, L_half, L_half))
        r2_broadcast = jnp.broadcast_to(r2_exp, (B, L_half, L_half))

        mask_j0 = (r1_broadcast == -r2_broadcast)
        mask_j1 = (r1_broadcast ==  r2_broadcast)
        semantic_mask = jnp.stack([mask_j0, mask_j1], axis=-1)  # [B, L_half, L_half, 2]

        # Padding masks for i and j positions
        mask1_b = mask1[:, :, None]  # [B, L_half, 1]
        mask2_b = mask2[:, None, :]  # [B, 1, L_half]
        padding_mask = mask1_b & mask2_b  # [B, L_half, L_half]

        # Combine semantic mask with padding mask to get valid (i,j) actions per type
        base_mask = semantic_mask & padding_mask[..., None]  # [B, L_half, L_half, 2]
        final_mask = jnp.broadcast_to(base_mask[:,:,:,None,:], (B,L_half,L_half,2,2))
        final_mask = final_mask.reshape(B, -1)  # [B, total_action_dim]

        # === Embeddings ===
        embed = nn.Embed(self.vocab_size, self.embedding_dim, name="shared_embed")
        x1 = embed(r1_raw.astype(jnp.int32) + 2)  # [B, L_half, D]
        x2 = embed(r2_raw.astype(jnp.int32) + 2)  # [B, L_half, D]

        # Transformer blocks
        for _ in range(self.num_layers):
            x1, x2 = AbsoluteDualRingBlock(
                self.num_heads, self.head_dim, self.mlp_dim, self.max_len
            )(x1, x2, mask1, mask2)

        # === Value head ===
        def masked_mean(x, mask):
            mask = mask.astype(jnp.float32)
            return (x * mask[:, :, None]).sum(axis=1) / (mask.sum(axis=1, keepdims=True) + 1e-6)

        pooled1 = masked_mean(x1, mask1)
        pooled2 = masked_mean(x2, mask2)
        joint = jnp.concatenate([pooled1, pooled2], axis=-1)

        act_fn = nn.gelu if self.activation == "gelu" else nn.tanh

        critic = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)))(joint)
        critic = act_fn(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)))(critic)
        critic = act_fn(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0))(critic)

        # === Actor head ===
        x1_exp = jnp.expand_dims(x1, axis=2)  # [B, L_half, 1, D]
        x2_exp = jnp.expand_dims(x2, axis=1)  # [B, 1, L_half, D]

        x1_b = jnp.broadcast_to(x1_exp, (B, L_half, L_half, x1.shape[-1]))
        x2_b = jnp.broadcast_to(x2_exp, (B, L_half, L_half, x2.shape[-1]))

        x_joint = jnp.concatenate([x1_b, x2_b], axis=-1)  # [B, L_half, L_half, 2D]

        x_joint = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)))(x_joint)
        x_joint = act_fn(x_joint)

        # Output logits for substitution actions (j_type * k1 * k2)
        logits = nn.Dense(4, kernel_init=orthogonal(0.01))(x_joint)  # [B, L_half, L_half, 4] k1, k2, ij

        # flatten to [B, total_action_dim]; sample = ((k1 * L_half + k2) * 4) + (i * 2 + j), so j is the last index
        logits_flat = logits.reshape(B, -1)

        # Mask invalid logits with -1e9
        logits_flat = jnp.where(final_mask.reshape(B, -1), logits_flat, -1e9)

        cov_enabled = self.change_of_variables_moves or self.stable_ac_moves
        ac45_enabled = self.ac45_moves or self.stable_ac_moves
        if cov_enabled:
            stable_logits = nn.Dense(8, kernel_init=orthogonal(0.01))(x_joint)
            stable_mask = _stable_change_of_variables_mask(mask1, mask2)
            stable_logits = jnp.where(stable_mask, stable_logits, -1e9)
            logits_flat = jnp.concatenate(
                [logits_flat, stable_logits.reshape(B, -1)],
                axis=-1,
            )
        if ac45_enabled:
            generator_logits = nn.Dense(3, kernel_init=orthogonal(0.01))(joint)
            logits_flat = jnp.concatenate([logits_flat, generator_logits], axis=-1)

        # Distribution over all actions (i, j, j_type, k1, k2)
        pi = distrax.Categorical(logits=logits_flat)

        return pi, jnp.squeeze(critic, axis=-1)
