# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused residual addition + LayerNorm for the RADIO vision encoder."""

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# The kernel reproduces the reduction order of PyTorch's vectorized CUDA
# LayerNorm, so its output is bit-identical only where that order was checked.
_QUALIFIED_CAPABILITIES = {(8, 0), (9, 0), (10, 0), (10, 3)}
_QUALIFIED_TORCH = "2.14."
_WIDTH = 1280


def _is_supported(residual: torch.Tensor, update: torch.Tensor, norm: nn.Module):
    return (
        residual.is_cuda
        and not torch.is_grad_enabled()
        and torch.version.hip is None
        and residual.dtype == update.dtype == torch.bfloat16
        and residual.ndim >= 2
        and residual.shape == update.shape
        and residual.shape[-1] == _WIDTH
        and residual.is_contiguous()
        and update.is_contiguous()
        and isinstance(norm, nn.LayerNorm)
        and norm.normalized_shape == (_WIDTH,)
        and norm.weight is not None
        and norm.bias is not None
        and norm.weight.dtype == norm.bias.dtype == torch.bfloat16
        and residual.device == update.device == norm.weight.device == norm.bias.device
        and norm.weight.is_contiguous()
        and norm.bias.is_contiguous()
        and norm.eps > 0
        and torch.__version__.startswith(_QUALIFIED_TORCH)
        and torch.cuda.get_device_capability(residual.device) in _QUALIFIED_CAPABILITIES
        and hasattr(torch.ops._C, "radio_fused_residual_layer_norm")
    )


def fused_residual_layer_norm(
    residual: torch.Tensor, update: torch.Tensor, norm: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(residual + update, norm(residual + update))``.

    The sum is rounded to BF16 before the statistics, as in the unfused
    computation. Unsupported inputs, devices and builds use the unfused path.
    """
    if _is_supported(residual, update, norm):
        summed = torch.empty_like(residual)
        normalized = torch.empty_like(residual)
        torch.ops._C.radio_fused_residual_layer_norm(
            residual, update, norm.weight, norm.bias, summed, normalized, norm.eps
        )
        return summed, normalized
    logger.warning_once(
        "RADIO fused residual LayerNorm is unavailable for this build or input; "
        "using the unfused residual addition and LayerNorm."
    )
    summed = residual + update
    return summed, norm(summed)
