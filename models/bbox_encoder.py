"""
BboxEncoder — encodes a padded set of 3D bounding boxes into token embeddings
that can be concatenated with CLIP spatial features in the DiT cross-attention.

Input  : (B, N, 7) boxes  [x, y, z, dim_x, dim_y, dim_z, theta]
         (B, N)    mask   True = real box, False = padding
Output : (B, N, context_dim=1024) tokens

Architecture:
  Linear(7→128) → LayerNorm → ReLU
  → 2-layer Transformer (self-attention between boxes)
  → Linear(128→context_dim)

Padding tokens are zeroed out before returning.
"""

import torch
import torch.nn as nn
import math


class BboxEncoder(nn.Module):
    """
    Parameters
    ----------
    context_dim : int
        Output dimension — must match CLIP feature dim (1024 for ViT-L/14).
    n_transformer_layers : int
        Number of self-attention layers between boxes.
    hidden_dim : int
        Internal hidden dimension.
    max_bboxes : int
        Maximum number of boxes per sample (for positional embedding).
    """

    def __init__(
        self,
        context_dim: int = 1024,
        n_transformer_layers: int = 2,
        hidden_dim: int = 128,
        max_bboxes: int = 8,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.max_bboxes  = max_bboxes

        # ── Input projection ───────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── Learnable positional embedding for box order ───────────────────
        self.pos_embed = nn.Embedding(max_bboxes, hidden_dim)

        # ── Self-attention between boxes ───────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
        )

        # ── Output projection to CLIP dim ──────────────────────────────────
        self.out_proj = nn.Linear(hidden_dim, context_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        bboxes: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        bboxes : (B, N, 7) — padded 3D box parameters
        mask   : (B, N)    — bool, True = valid box

        Returns
        -------
        tokens : (B, N, context_dim)
                 Padding positions are zeroed out.
        """
        B, N, _ = bboxes.shape
        pos_ids = torch.arange(N, device=bboxes.device)  # (N,)

        # ── Input projection ───────────────────────────────────────────────
        h = self.input_proj(bboxes)                        # (B, N, hidden_dim)
        h = h + self.pos_embed(pos_ids).unsqueeze(0)       # (B, N, hidden_dim)

        # ── Self-attention (key_padding_mask: True = ignore) ───────────────
        # nn.TransformerEncoder expects src_key_padding_mask where True = pad
        src_key_padding_mask = ~mask  # (B, N)  True = padding position

        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # (B, N, hidden_dim)

        # ── Output projection ──────────────────────────────────────────────
        tokens = self.out_proj(h)                          # (B, N, context_dim)

        # Zero-out padding token embeddings so they don't pollute attention
        tokens = tokens * mask.unsqueeze(-1).float()       # (B, N, context_dim)

        return tokens


# ── Null (empty-scene) token for CFG dropout ──────────────────────────────────

class NullBboxContext(nn.Module):
    """
    Trainable null embedding for classifier-free guidance.
    When we drop the bbox condition, we replace all bbox tokens with this.
    """

    def __init__(self, max_bboxes: int = 8, context_dim: int = 1024):
        super().__init__()
        self.null = nn.Parameter(torch.zeros(1, max_bboxes, context_dim))
        nn.init.normal_(self.null, std=0.02)

    def expand(self, batch_size: int) -> torch.Tensor:
        return self.null.expand(batch_size, -1, -1)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, N = 4, 8
    enc   = BboxEncoder()
    boxes = torch.randn(B, N, 7)
    mask  = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :3] = True   # 3 real boxes, rest padding

    out = enc(boxes, mask)
    print(f"Input  : {boxes.shape}")
    print(f"Mask   : {mask.shape}  ({mask.sum(dim=1).tolist()} real boxes per item)")
    print(f"Output : {out.shape}")
    # Padding positions should be zero
    print(f"Pad token norm (should be 0): {out[:, 5:].norm().item():.6f}")
