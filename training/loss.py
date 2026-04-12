"""
Loss functions for RA-SynthGen.

Primary loss  : OT-CFM velocity MSE
Auxiliary loss: TCR (Target-Consistency Regularisation) — ported verbatim
                from Project_RA_Gen_CFM/scripts/loss_functions.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCRLoss(nn.Module):
    """
    Target-Consistency Regularisation.
    Inspired by CFAR detection and Focal Loss to enforce semantic consistency
    between the model's x1 reconstruction and the ground-truth RA map.

    Parameters
    ----------
    kernel_size : int   Local neighbourhood size for computing μ and σ.
    w           : float Threshold = μ + w·σ  (controls sensitivity).
    alpha       : float Sigmoid sharpness.
    gamma       : float Focal exponent.
    """

    def __init__(self, kernel_size=9, w=1.0, alpha=5.0, gamma=2.0):
        super().__init__()
        self.w      = w
        self.alpha  = alpha
        self.gamma  = gamma
        pad         = kernel_size // 2
        self.avg    = nn.AvgPool2d(kernel_size, stride=1, padding=pad, count_include_pad=False)

    def _local_stats(self, x):
        mu    = self.avg(x)
        mu_sq = self.avg(x ** 2)
        sigma = torch.sqrt(torch.clamp(mu_sq - mu ** 2, min=1e-6))
        return mu, sigma

    def _prob_map(self, x):
        mu, sigma = self._local_stats(x)
        tau = mu + self.w * sigma
        return torch.sigmoid(self.alpha * (x - tau))

    def forward(self, x_pred: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
        p_pred = self._prob_map(x_pred)
        with torch.no_grad():
            p_gt = self._prob_map(x_gt)

        eps    = 1e-6
        p_pred = torch.clamp(p_pred, eps, 1.0 - eps)

        t1 = p_gt * torch.pow(1.0 - p_pred, self.gamma) * torch.log(p_pred)
        t2 = (1.0 - p_gt) * torch.pow(p_pred, self.gamma) * torch.log(1.0 - p_pred)
        return -torch.mean(t1 + t2)


class CFMLoss(nn.Module):
    """
    OT-CFM training objective.

    Given:
        x0      : source prior  (B, 1, H, W)
        x1      : clean target  (B, 1, H, W)
        t       : timestep      (B,)
        v_pred  : predicted velocity (B, 1, H, W)

    Computes:
        x_t     = (1-t) * x0 + t * x1            (linear interpolation)
        v_tgt   = x1 - x0                         (constant velocity target)
        L_mse   = || v_pred - v_tgt ||²
        x1_pred = x_t + (1-t) * v_pred            (x1 reconstruction)
        L_tcr   = TCR(x1_pred, x1)
        loss    = L_mse + lambda_tcr * L_tcr
    """

    def __init__(self, lambda_tcr: float = 0.1):
        super().__init__()
        self.lambda_tcr = lambda_tcr
        self.tcr        = TCRLoss(w=1.0, alpha=5.0, gamma=2.0)

    def forward(
        self,
        v_pred: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t:  torch.Tensor,
    ):
        t_view  = t.view(-1, 1, 1, 1)
        x_t     = (1 - t_view) * x0 + t_view * x1
        v_tgt   = x1 - x0

        loss_mse = torch.mean((v_pred - v_tgt) ** 2)

        x1_pred  = x_t + (1 - t_view) * v_pred
        loss_tcr = self.tcr(x1_pred, x1)

        loss     = loss_mse + self.lambda_tcr * loss_tcr
        return loss, loss_mse, loss_tcr
