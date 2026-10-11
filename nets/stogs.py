import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class STOGSGroupOperator(nn.Module):
    """Shared centered 3-D overlapping-group operator for network and loss."""

    def __init__(
        self,
        group_size=(3, 3, 3),
        eps=1e-6,
        weight_bounds=(0.25, 4.0),
    ):
        super().__init__()
        if len(group_size) != 3 or any(size <= 0 or size % 2 == 0 for size in group_size):
            raise ValueError("group_size must contain three positive odd integers")
        self.group_size = tuple(int(size) for size in group_size)
        self.eps = float(eps)
        self.weight_bounds = tuple(float(value) for value in weight_bounds)

    @property
    def _padding(self):
        temporal, height, width = self.group_size
        return temporal // 2, height // 2, width // 2

    def _validate(self, x):
        if x.ndim != 5:
            raise ValueError("ST-OGS expects an N x C x T x H x W tensor")

    def _group_sum(self, x):
        temporal_pad, height_pad, width_pad = self._padding
        volume = math.prod(self.group_size)
        x = F.pad(
            x,
            (
                width_pad, width_pad,
                height_pad, height_pad,
                temporal_pad, temporal_pad,
            ),
        )
        return F.avg_pool3d(x, self.group_size, stride=1) * volume

    def group_energy(self, x):
        """Channel-joint RMS energy in each centered spatio-temporal group."""
        self._validate(x)
        energy = x.float().square().mean(dim=1, keepdim=True)
        group_sum = self._group_sum(energy)
        valid = self._group_sum(torch.ones_like(energy))
        return torch.sqrt(group_sum / valid.clamp_min(1.0) + self.eps)

    def group_weight(self, previous_target):
        """MM inverse-energy weights, fixed with respect to the current step."""
        energy = self.group_energy(previous_target.detach())
        weight = 1.0 / (energy + self.eps)
        weight = weight / weight.mean(dim=(2, 3, 4), keepdim=True).clamp_min(self.eps)
        return weight.clamp(*self.weight_bounds)

    def accumulated_weight(self, group_weight):
        """Map group weights back to voxels with valid-count normalization."""
        self._validate(group_weight)
        weight_sum = self._group_sum(group_weight.float())
        valid = self._group_sum(
            torch.ones_like(group_weight, dtype=torch.float32)
        )
        return weight_sum / valid.clamp_min(1.0)


class CenteredTemporalPropagation(nn.Module):
    """Symmetric fusion around the middle/current frame."""

    def __init__(self, channels, temporal_window=5, alpha_init=0.1):
        super().__init__()
        if temporal_window < 1 or temporal_window % 2 == 0:
            raise ValueError("temporal_window must be a positive odd number")
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must be between zero and one")
        self.temporal_window = int(temporal_window)
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(self.temporal_window, 3, 3),
            padding=0,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.gate = nn.Conv3d(channels * 2, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.bias)
        self.alpha_logit = nn.Parameter(
            torch.tensor(math.log(alpha_init / (1.0 - alpha_init)), dtype=torch.float32)
        )

    def forward(self, residual):
        # Symmetric temporal padding lets the middle/current frame directly
        # fuse two preceding and two following frames.
        temporal_pad = self.temporal_window // 2
        context = F.pad(
            residual,
            (1, 1, 1, 1, temporal_pad, temporal_pad),
        )
        context = self.pointwise(F.silu(self.depthwise(context)))
        gate = torch.sigmoid(self.gate(torch.cat((residual, context), dim=1)))
        alpha = torch.sigmoid(self.alpha_logit).to(dtype=residual.dtype)
        return residual + alpha * gate * context


class AdaptiveSTOGSShrinkage(nn.Module):
    """MM-reweighted soft threshold corresponding to the ST-OGS prior."""

    def __init__(
        self,
        groups,
        use_reweight,
        threshold_init=0.01,
        threshold_max=0.2,
        eps=1e-6,
    ):
        super().__init__()
        if not 0.0 < threshold_init < threshold_max:
            raise ValueError("threshold_init must lie between zero and threshold_max")
        self.groups = groups
        self.use_reweight = bool(use_reweight)
        self.threshold_max = float(threshold_max)
        self.eps = float(eps)
        initial_ratio = threshold_init / threshold_max
        self.threshold_logit = nn.Parameter(
            torch.tensor(math.log(initial_ratio / (1.0 - initial_ratio)), dtype=torch.float32)
        )

    def forward(self, propagated, previous_target):
        if self.use_reweight:
            group_weight = self.groups.group_weight(previous_target)
        else:
            group_weight = propagated.new_ones(
                (propagated.shape[0], 1, *propagated.shape[2:]),
                dtype=torch.float32,
            )

        structural_weight = self.groups.accumulated_weight(group_weight)
        feature_scale = torch.sqrt(
            propagated.float().square().mean(dim=(1, 2, 3, 4), keepdim=True) + self.eps
        ).detach()
        ratio = self.threshold_max * torch.sigmoid(self.threshold_logit)
        threshold = (ratio * feature_scale * structural_weight).to(propagated.dtype)
        return torch.sign(propagated) * F.relu(propagated.abs() - threshold)


class STOGSSparseModule(nn.Module):
    """One unfolded target update: propagation followed by ST-OGS shrinkage."""

    def __init__(
        self,
        channels,
        stage_index,
        group_size=(5, 3, 3),
        eps=1e-6,
    ):
        super().__init__()
        # Use the same temporal support for feature propagation and the
        # overlapping-group sparse prior.
        self.propagation = CenteredTemporalPropagation(
            channels,
            temporal_window=group_size[0],
        )
        groups = STOGSGroupOperator(group_size=group_size, eps=eps)
        self.shrinkage = AdaptiveSTOGSShrinkage(
            groups=groups,
            use_reweight=stage_index > 0,
            eps=eps,
        )

    def forward(self, D, B, previous_target):
        residual = previous_target + D - B
        propagated = self.propagation(residual)
        return self.shrinkage(propagated, previous_target)


class STOGSLoss(nn.Module):
    """Final-stage MM surrogate using T2 weights to regularize T3."""

    def __init__(
        self,
        group_size=(5, 3, 3),
        eps=1e-6,
    ):
        super().__init__()
        self.groups = STOGSGroupOperator(group_size=group_size, eps=eps)
        self.eps = float(eps)

    def _normalize(self, target):
        scale = torch.sqrt(
            target.float().square().mean(dim=(1, 2, 3, 4), keepdim=True) + self.eps
        ).detach()
        return target.float() / scale

    def forward(self, sparse_states):
        if len(sparse_states) < 2:
            raise ValueError("STOGSLoss requires at least the T2 and T3 sparse states")

        previous_target = self._normalize(sparse_states[-2])
        final_target = self._normalize(sparse_states[-1])
        weight = self.groups.group_weight(previous_target)
        energy = self.groups.group_energy(final_target)
        return (weight * energy).mean()
