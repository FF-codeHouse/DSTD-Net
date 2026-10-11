"""Multi-mode tensor background feature extractor.

The module processes video features whose layout is ``B x C x T x H x W``.
The spatio-temporal GLAM block first models explicit local similarity inside
3-D windows and non-local similarity through a compact set of clip-level
background tokens.  It never folds the temporal dimension into the batch.
The M-k Mamba block scans the complete spatio-temporal tensor in three
unfoldings and two directions per unfolding.
"""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as import_error:  # Keep module importable for clear errors.
    Mamba = None
    _MAMBA_IMPORT_ERROR = import_error
else:
    _MAMBA_IMPORT_ERROR = None


def _channel_layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """Apply LayerNorm over C for a B x C x T x H x W tensor."""
    return norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()


class SpatioTemporalGLAM(nn.Module):
    """Global-local attention over a complete video feature tensor.

    The local branch explicitly calculates content-adaptive similarity inside
    compact 3-D windows.  The global branch keeps GLAM's two-way interaction:
    compact tokens aggregate a spatially pooled version of all frames, then
    broadcast the clip-adaptive background representation back to every
    spatio-temporal position.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_global_tokens: int = 4,
        temporal_window: int = 5,
        spatial_window: int = 3,
        global_stride: int = 2,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads "
                f"({num_heads})"
            )
        if min(
            num_global_tokens, temporal_window, spatial_window, global_stride
        ) < 1:
            raise ValueError("GLAM token and window sizes must be positive")
        if temporal_window % 2 == 0 or spatial_window % 2 == 0:
            raise ValueError("GLAM local window sizes must be odd")
        self.channels = channels
        self.temporal_window = temporal_window
        self.spatial_window = spatial_window
        self.global_stride = global_stride
        self.input_norm = nn.LayerNorm(channels)
        self.global_tokens = nn.Parameter(
            torch.empty(1, num_global_tokens, channels)
        )
        nn.init.trunc_normal_(self.global_tokens, std=0.02)

        self.local_attention = nn.MultiheadAttention(
            channels, num_heads, batch_first=True
        )
        self.local_to_global = nn.MultiheadAttention(
            channels, num_heads, batch_first=True
        )
        self.global_to_local = nn.MultiheadAttention(
            channels, num_heads, batch_first=True
        )
        self.global_norm = nn.LayerNorm(channels)
        self.fusion = nn.Sequential(
            nn.Linear(channels * 2, channels, bias=False),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=False),
        )

    def _local_windows(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic self-attention in padded 3-D local windows."""
        b, t, h, w, c = x.shape
        tw = self.temporal_window
        sw = self.spatial_window
        pad_t = (tw - t % tw) % tw
        pad_h = (sw - h % sw) % sw
        pad_w = (sw - w % sw) % sw

        # Pad only the ends of T/H/W; C is not padded.
        padding = (0, 0, 0, pad_w, 0, pad_h, 0, pad_t)
        padded = F.pad(x, padding)
        tp = t + pad_t
        hp = h + pad_h
        wp = w + pad_w
        windows = (
            padded.reshape(b, tp // tw, tw, hp // sw, sw, wp // sw, sw, c)
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .contiguous()
            .reshape(-1, tw * sw * sw, c)
        )

        # Border padding must not become background evidence in attention.
        valid = x.new_ones((b, t, h, w, 1), dtype=torch.bool)
        valid = F.pad(valid, padding, value=False)
        valid_windows = (
            valid.reshape(b, tp // tw, tw, hp // sw, sw, wp // sw, sw, 1)
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .contiguous()
            .reshape(-1, tw * sw * sw)
        )
        attended, _ = self.local_attention(
            windows,
            windows,
            windows,
            key_padding_mask=~valid_windows,
            need_weights=False,
        )
        attended = (
            attended.reshape(
                b, tp // tw, hp // sw, wp // sw, tw, sw, sw, c
            )
            .permute(0, 1, 4, 2, 5, 3, 6, 7)
            .contiguous()
            .reshape(b, tp, hp, wp, c)
        )
        return attended[:, :t, :h, :w, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"SpatioTemporalGLAM expects B x C x T x H x W, "
                f"got {tuple(x.shape)}"
            )
        b, c, t, h, w = x.shape
        features = self.input_norm(
            x.permute(0, 2, 3, 4, 1).contiguous()
        )

        # The local branch mixes time and space jointly inside every 3-D
        # window; frames are not processed independently.
        local_features = self._local_windows(features)

        # Pool only H and W before global aggregation.  The temporal axis is
        # preserved, so every frame contributes to the clip-level tokens.
        flat_features = features.reshape(b, t * h * w, c)
        global_source = features.permute(0, 4, 1, 2, 3).contiguous()
        if self.global_stride > 1:
            global_source = F.avg_pool3d(
                global_source,
                kernel_size=(1, self.global_stride, self.global_stride),
                stride=(1, self.global_stride, self.global_stride),
                ceil_mode=True,
            )
        global_source = (
            global_source.permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(1, 3)
        )
        global_queries = self.global_tokens.expand(b, -1, -1)
        aggregated, _ = self.local_to_global(
            global_queries,
            global_source,
            global_source,
            need_weights=False,
        )
        global_features = self.global_norm(global_queries + aggregated)
        global_feedback, _ = self.global_to_local(
            flat_features,
            global_features,
            global_features,
            need_weights=False,
        )
        global_feedback = global_feedback.reshape(b, t, h, w, c)

        output = self.fusion(
            torch.cat((local_features, global_feedback), dim=-1)
        )
        return output.permute(0, 4, 1, 2, 3).contiguous()


class ModeKMamba(nn.Module):
    """Three tensor unfoldings with forward/reverse Mamba scans per mode."""

    # The first three entries are tensor axes; C remains the Mamba embedding.
    _MODE_PERMUTATIONS: Tuple[Tuple[int, ...], ...] = (
        (0, 2, 3, 4, 1),  # T, H, W
        (0, 3, 2, 4, 1),  # H, T, W
        (0, 4, 2, 3, 1),  # W, T, H
    )
    # Forward and exact reverse traversals of each unfolded tensor.
    _DIRECTION_FLIPS: Tuple[Tuple[int, ...], ...] = (
        (),
        (1, 2, 3),
    )

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "MMIT requires mamba_ssm. Install it in the training environment."
            ) from _MAMBA_IMPORT_ERROR
        self.channels = channels
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(3)])
        self.mamba_blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(3)
            ]
        )
        self.mode_weights = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _inverse_permutation(permutation: Sequence[int]) -> Tuple[int, ...]:
        inverse = [0] * len(permutation)
        for index, axis in enumerate(permutation):
            inverse[axis] = index
        return tuple(inverse)

    def _scan_mode(
        self,
        x: torch.Tensor,
        mode_index: int,
    ) -> torch.Tensor:
        permutation = self._MODE_PERMUTATIONS[mode_index]
        mode_tensor = x.permute(*permutation).contiguous()
        b, axis_1, axis_2, axis_3, channels = mode_tensor.shape

        directional_tensors = [
            mode_tensor.flip(flip_axes) if flip_axes else mode_tensor
            for flip_axes in self._DIRECTION_FLIPS
        ]
        # Run the four traversals together as an enlarged batch.
        sequences = torch.cat(
            [tensor.reshape(b, -1, channels) for tensor in directional_tensors],
            dim=0,
        )
        sequences = self.norms[mode_index](sequences)
        sequences = self.mamba_blocks[mode_index](sequences)

        directional_outputs = sequences.split(b, dim=0)
        restored = []
        for output, flip_axes in zip(
            directional_outputs, self._DIRECTION_FLIPS
        ):
            output = output.reshape(b, axis_1, axis_2, axis_3, channels)
            if flip_axes:
                output = output.flip(flip_axes)
            restored.append(output)

        mode_output = torch.stack(restored, dim=0).mean(dim=0)
        inverse = self._inverse_permutation(permutation)
        return mode_output.permute(*inverse).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"M-k Mamba expects B x C x T x H x W, got {tuple(x.shape)}"
            )
        mode_outputs = [self._scan_mode(x, index) for index in range(3)]
        weights = torch.softmax(self.mode_weights, dim=0)
        return sum(
            weight * output
            for weight, output in zip(weights, mode_outputs)
        )


class SpatioTemporalFeedForward(nn.Module):
    """Gated depthwise 3D feed-forward network."""

    def __init__(self, channels: int, expansion: float = 2.0) -> None:
        super().__init__()
        hidden_channels = max(int(channels * expansion), channels)
        self.norm = nn.LayerNorm(channels)
        self.project_in = nn.Conv3d(
            channels, hidden_channels * 2, kernel_size=1, bias=False
        )
        self.depthwise = nn.Conv3d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
        )
        self.project_out = nn.Conv3d(
            hidden_channels, channels, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _channel_layer_norm(x, self.norm)
        first, second = self.depthwise(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.silu(first) * second)


class MMITBlock(nn.Module):
    """Spatio-temporal GLAM, M-k Mamba and residual 3-D FFN."""

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        mamba_expand: int = 2,
        residual_scale: float = 0.2,
        num_heads: int = 4,
        num_global_tokens: int = 4,
        temporal_window: int = 5,
        spatial_window: int = 3,
        global_stride: int = 2,
    ) -> None:
        super().__init__()
        self.glam = SpatioTemporalGLAM(
            channels,
            num_heads=num_heads,
            num_global_tokens=num_global_tokens,
            temporal_window=temporal_window,
            spatial_window=spatial_window,
            global_stride=global_stride,
        )
        self.mode_k_mamba = ModeKMamba(
            channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
        )
        self.ffn = SpatioTemporalFeedForward(channels)
        self.glam_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.mode_k_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.ffn_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.glam_scale * self.glam(x)
        x = x + self.mode_k_scale * self.mode_k_mamba(x)
        x = x + self.ffn_scale * self.ffn(x)
        return x


class MMIT(nn.Module):
    """Complete multi-frame MMIT background feature extractor."""

    def __init__(
        self,
        channels: int,
        num_blocks: int = 1,
        **block_kwargs,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        self.blocks = nn.Sequential(
            *[
                MMITBlock(channels=channels, **block_kwargs)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"MMIT expects B x C x T x H x W, got {tuple(x.shape)}"
            )
        return self.blocks(x)
