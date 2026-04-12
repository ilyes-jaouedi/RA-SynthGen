"""
DiT (Diffusion Transformer) for RA-SynthGen.

Image encoder: DINOv2 ViT-L/14 (frozen) — replaces CLIP.
  DINOv2 is self-supervised with a dense spatial objective, giving
  geometrically precise patch features vs CLIP's globally-pooled semantic ones.
  Same output shape: (B, 703, 1024) for 266x518 input with patch_size=14.

Cross-attention context:
  [DINOv2 tokens  (B, N_dino, 1024)]  calibration-aware spatial bias (hard mask)
  [Bbox tokens    (B, N_bbox, 1024)]  Gaussian spatial bias centred at projected
                                       (r_bin, az_bin) of each detection

Two calibration-aware attention biases
  1. CLIP/DINOv2 mask (256, N_dino): pre-computed from calibration projection,
     stored as {-1, 0} binary, multiplied by a learnable scale.
     -scale  = radar patch is geometrically blocked from this image patch
      0      = permitted

  2. Bbox Gaussian (B, 256, N_bbox): per-sample, computed live from raw (x,y,z).
     bias(i,j) = -dist_patch² / (2 * geo_bbox_scale²)
     where dist is the distance in 16x16 patch-grid space between radar patch
     (i,j) and the projected position of the bbox.
     Padded / out-of-grid boxes get zero bias.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn

from transformers import Dinov2Model
from .bbox_encoder import BboxEncoder, NullBboxContext

# ── Configuration ──────────────────────────────────────────────────────────────
RADAR_SIZE       = 256
PATCH_SIZE       = 16
N_PATCHES        = (RADAR_SIZE // PATCH_SIZE) ** 2   # 256
GRID             = RADAR_SIZE // PATCH_SIZE           # 16
HIDDEN_DIM       = 384
NUM_HEADS        = 6
NUM_LAYERS       = 6
DINO_MODEL_NAME  = "facebook/dinov2-large"
CONTEXT_DIM      = 1024
MAX_BBOXES       = 8

# Radar geometry constants (same as psf_prior.py)
AZ_MIN_DEG  = -75.0
AZ_MAX_DEG  =  75.0
R_MAX_M     = 103.0

_HERE      = os.path.dirname(os.path.abspath(__file__))
MASK_PATH  = os.path.join(_HERE, "..", "assets", "geometry_soft_bias.npy")


# ── adaLN helper ──────────────────────────────────────────────────────────────

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ── Sinusoidal time embedding ──────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        self.register_buffer("freqs", freqs.float())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1) * self.freqs.view(1, -1)
        return torch.cat([t.sin(), t.cos()], dim=1)


# ── DiTBlock ──────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, ctx_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden, heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn2 = nn.MultiheadAttention(
            hidden, heads, kdim=ctx_dim, vdim=ctx_dim, batch_first=True
        )

        self.norm3 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden * 4, hidden),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, 6 * hidden, bias=True),
        )

    def forward(self, x, t_emb, context, attn_mask=None):
        s_msa, sc_msa, s_ca, sc_ca, s_mlp, sc_mlp = self.adaLN(t_emb).chunk(6, dim=1)

        xn = modulate(self.norm1(x), s_msa, sc_msa)
        x  = x + self.attn1(xn, xn, xn)[0]

        xn = modulate(self.norm2(x), s_ca, sc_ca)
        x  = x + self.attn2(xn, context, context, attn_mask=attn_mask)[0]

        xn = modulate(self.norm3(x), s_mlp, sc_mlp)
        x  = x + self.mlp(xn)
        return x


# ── DiT model ─────────────────────────────────────────────────────────────────

class DiT(nn.Module):
    def __init__(
        self,
        input_size: int = RADAR_SIZE,
        patch_size: int = PATCH_SIZE,
        in_channels: int = 1,
        hidden: int = HIDDEN_DIM,
        depth: int = NUM_LAYERS,
        heads: int = NUM_HEADS,
        context_dim: int = CONTEXT_DIM,
        max_bboxes: int = MAX_BBOXES,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.num_heads   = heads
        num_patches      = (input_size // patch_size) ** 2   # 256
        self.grid        = input_size // patch_size          # 16

        # ── Null context for DINOv2 dropout ───────────────────────────────
        self.null_dino_embed = nn.Parameter(torch.randn(1, context_dim))

        # ── Calibration geometry mask for DINOv2 cross-attention ──────────
        # Stored as {-1, 0} binary; multiplied by learnable scale at runtime.
        # -scale  → radar patch blocked from attending to this image token
        #  0      → permitted (no additive shift)
        # Clamped to [-1e4, 0] to stay fp16-safe.
        if os.path.exists(MASK_PATH):
            print(f">> Loading geometry calibration mask: {MASK_PATH}")
            raw_mask = np.load(MASK_PATH).astype(np.float32)   # (256, N_dino)
            binary   = np.where(raw_mask < -1.0, -1.0, 0.0).astype(np.float32)
            self.register_buffer("dino_attn_bias", torch.from_numpy(binary))
            self.n_dino_tokens  = raw_mask.shape[1]
            # Learnable scale: init=1e4 (hard mask), can soften during training
            self.geo_mask_scale = nn.Parameter(torch.tensor(1e4))
        else:
            print(f"!! Geometry mask not found — using global attention.")
            self.register_buffer("dino_attn_bias", None)
            self.n_dino_tokens  = 703
            self.geo_mask_scale = nn.Parameter(torch.tensor(0.0))

        # ── Bbox Gaussian attention bias ───────────────────────────────────
        # Pre-compute patch centre coordinates in the 16x16 patch grid.
        # patch_coords[k] = (row_centre, col_centre) for patch k in [0,15]
        rows = torch.arange(self.grid, dtype=torch.float32) + 0.5
        cols = torch.arange(self.grid, dtype=torch.float32) + 0.5
        grid_r, grid_az = torch.meshgrid(rows, cols, indexing="ij")
        self.register_buffer(
            "patch_coords",
            torch.stack([grid_r.flatten(), grid_az.flatten()], dim=1),
        )   # (256, 2)

        # Learnable Gaussian sigma in patch units.
        # Init = 2.0 → ~1 patch FWHM, attention drops to 50% at ~2 patches away.
        self.geo_bbox_scale = nn.Parameter(torch.tensor(2.0))

        # ── Patchify ───────────────────────────────────────────────────────
        self.x_embedder = nn.Conv2d(
            in_channels, hidden, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden))

        # ── Timestep ──────────────────────────────────────────────────────
        self.t_embedder = nn.Sequential(
            TimeEmbedding(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, heads, context_dim) for _ in range(depth)]
        )

        # ── Final head ────────────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_ada  = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.final_proj = nn.Linear(hidden, patch_size * patch_size * in_channels)

        # ── Refinement ────────────────────────────────────────────────────
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1),
        )

        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def unpatchify(self, x):
        c, p = self.in_channels, self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ── Attention mask builder ────────────────────────────────────────────────

    def _build_attn_mask(
        self,
        B: int,
        n_dino: int,
        n_bbox: int,
        bboxes_raw: torch.Tensor,    # (B, N_bbox, 7)  [x,y,z,...]
        bbox_mask: torch.Tensor,     # (B, N_bbox) bool  True=real
        use_dino_bias: bool = True,
    ) -> torch.Tensor:
        """
        Returns (B * num_heads, N_radar, n_dino + n_bbox) additive logit bias.

        DINOv2 part  : binary calibration mask * learnable scale, fp16-safe.
        Bbox part    : per-sample Gaussian centred at projected (r_bin, az_bin),
                       zero for padded / inactive boxes.
        """
        device = self.patch_coords.device
        dtype  = self.patch_coords.dtype

        # ── DINOv2 calibration bias: (1, 256, n_dino) → expand to (B, ...) ──
        if use_dino_bias and self.dino_attn_bias is not None:
            scale      = self.geo_mask_scale.abs().clamp(max=1e4)
            dino_bias  = (self.dino_attn_bias[:, :n_dino] * scale).clamp(min=-1e4)
            dino_bias  = dino_bias.unsqueeze(0).expand(B, -1, -1)   # (B, 256, n_dino)
        else:
            dino_bias = torch.zeros(B, N_PATCHES, n_dino, device=device, dtype=dtype)

        # ── Bbox Gaussian bias: (B, 256, n_bbox) ─────────────────────────────
        x_b = bboxes_raw[:, :, 0]   # (B, N_bbox)
        y_b = bboxes_raw[:, :, 1]

        r_m    = torch.sqrt(x_b ** 2 + y_b ** 2).clamp(min=1e-3)     # (B, N_bbox)
        az_deg = torch.atan2(y_b, x_b) * (180.0 / math.pi)

        # Bin coords [0, 255]
        r_bin  = (r_m  / R_MAX_M * RADAR_SIZE).clamp(0, RADAR_SIZE - 1)
        az_bin = ((az_deg - AZ_MIN_DEG) / (AZ_MAX_DEG - AZ_MIN_DEG) * RADAR_SIZE
                  ).clamp(0, RADAR_SIZE - 1)

        # Continuous patch-grid coordinates [0, 16]
        r_patch  = r_bin  / self.grid    # (B, N_bbox)
        az_patch = az_bin / self.grid

        # bbox_proj: (B, N_bbox, 2)
        bbox_proj = torch.stack([r_patch, az_patch], dim=-1)

        # patch_coords: (256, 2) → (1, 256, 1, 2)
        # bbox_proj:               (B,   1, N_bbox, 2)
        diff    = (self.patch_coords.unsqueeze(0).unsqueeze(2)
                   - bbox_proj.unsqueeze(1))                          # (B, 256, N_bbox, 2)
        dist_sq = diff.pow(2).sum(-1)                                 # (B, 256, N_bbox)

        sigma_sq    = self.geo_bbox_scale.abs().pow(2).clamp(min=0.25)
        bbox_bias   = -dist_sq / (2.0 * sigma_sq)                    # (B, 256, N_bbox)

        # Zero out padded / inactive bbox slots
        valid      = bbox_mask.float().unsqueeze(1)                   # (B, 1, N_bbox)
        bbox_bias  = bbox_bias * valid

        # ── Combine and expand for multi-head attention ───────────────────────
        # (B, 256, n_dino + n_bbox)
        full_bias = torch.cat([dino_bias, bbox_bias], dim=2)

        # MHA expects (B * num_heads, tgt_len, src_len)
        full_bias = (full_bias
                     .unsqueeze(1)
                     .expand(-1, self.num_heads, -1, -1)
                     .reshape(B * self.num_heads, N_PATCHES, n_dino + n_bbox))
        return full_bias

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,             # (B, 1, 256, 256)
        t: torch.Tensor,             # (B,)
        dino_ctx: torch.Tensor,      # (B, N_dino, 1024)
        bbox_ctx: torch.Tensor,      # (B, N_bbox, 1024)
        bboxes_raw: torch.Tensor,    # (B, N_bbox, 7)  raw box params
        bbox_mask: torch.Tensor,     # (B, N_bbox) bool
        drop_clip: bool = False,     # alias kept for API compat
        drop_bbox: bool = False,
    ) -> torch.Tensor:
        B = x.shape[0]

        # ── CFG dropout ───────────────────────────────────────────────────
        use_dino_bias = True
        if drop_clip:                                   # drop image condition
            n_dino   = dino_ctx.shape[1]
            dino_ctx = self.null_dino_embed.unsqueeze(0).expand(B, n_dino, -1)
            use_dino_bias = False                       # spatial bias also dropped

        _bbox_mask = bbox_mask
        if drop_bbox:
            bbox_ctx   = torch.zeros_like(bbox_ctx)
            _bbox_mask = torch.zeros_like(bbox_mask)    # Gaussian bias zeroed too

        # ── Context + calibration mask ─────────────────────────────────────
        n_dino = dino_ctx.shape[1]
        n_bbox = bbox_ctx.shape[1]
        context   = torch.cat([dino_ctx, bbox_ctx], dim=1)   # (B, N_dino+N_bbox, 1024)
        attn_mask = self._build_attn_mask(
            B, n_dino, n_bbox, bboxes_raw, _bbox_mask, use_dino_bias,
        )   # (B*heads, 256, N_dino+N_bbox)

        # ── Patchify ──────────────────────────────────────────────────────
        x     = self.x_embedder(x).flatten(2).transpose(1, 2)  # (B, 256, hidden)
        x     = x + self.pos_embed
        t_emb = self.t_embedder(t)                              # (B, hidden)

        # ── Transformer ───────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x, t_emb, context, attn_mask=attn_mask)

        # ── Final head ─────────────────────────────────────────────────────
        shift, scale = self.final_ada(t_emb).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.unpatchify(self.final_proj(x))     # (B, 1, 256, 256)

        return x + self.refine(x)


# ── Spatial encoder: DINOv2 ViT-L/14 (frozen) ────────────────────────────────

class SpatialEncoder(nn.Module):
    """
    Frozen DINOv2 ViT-L/14.

    Input  : (B, 3, 266, 518)  — DINOv2-normalised (ImageNet stats)
    Output : (B, 703, 1024)    — spatial patch features, CLS token removed

    DINOv2 normalization:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    Produces geometrically consistent dense spatial features — substantially
    better than CLIP for tasks requiring spatial precision (depth, layout).
    Patch size 14 on 266x518 gives (19 x 37) = 703 tokens, same as CLIP ViT-L/14.
    """

    def __init__(self):
        super().__init__()
        print(f">> Loading DINOv2: {DINO_MODEL_NAME}")
        self.backbone = Dinov2Model.from_pretrained(DINO_MODEL_NAME)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 3, 266, 518)
        out   = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        feats = out.last_hidden_state[:, 1:, :]   # remove CLS → (B, 703, 1024)
        return feats


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(_HERE, ".."))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = DiT().to(device)

    n_dit  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"DiT trainable params: {n_dit:.1f}M")
    print(f"geo_mask_scale init : {model.geo_mask_scale.item():.1f}")
    print(f"geo_bbox_scale init : {model.geo_bbox_scale.item():.2f} patches")

    B = 2
    x_t         = torch.randn(B, 1, 256, 256, device=device)
    t_val       = torch.rand(B, device=device)
    dino_ctx    = torch.randn(B, 703, 1024, device=device)
    bbox_ctx    = torch.randn(B, 8,   1024, device=device)
    bboxes_raw  = torch.zeros(B, 8,   7,    device=device)
    bboxes_raw[:, :2, :3] = torch.tensor([[20., 5., 0.5], [40., -3., 0.5]])
    bbox_mask   = torch.zeros(B, 8, dtype=torch.bool, device=device)
    bbox_mask[:, :2] = True

    with torch.no_grad():
        v = model(x_t, t_val, dino_ctx, bbox_ctx, bboxes_raw, bbox_mask)

    print(f"Output shape : {v.shape}")
    assert v.shape == x_t.shape

    mask = model._build_attn_mask(B, 703, 8, bboxes_raw, bbox_mask)
    print(f"Attn mask    : {mask.shape}  range [{mask.min():.1f}, {mask.max():.2f}]")
    print(f"fp16 safe    : {mask.min().item() >= -65504}")
    print("Forward pass OK.")
