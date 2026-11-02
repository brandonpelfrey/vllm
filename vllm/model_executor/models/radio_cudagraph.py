# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from vllm.distributed import graph_capture

if TYPE_CHECKING:
    from .radio import MaskMetadata


@dataclass
class _EncoderGraph:
    graph: torch.cuda.CUDAGraph
    inputs: torch.Tensor
    # The replays read the captured cu_seqlens, so they must outlive the call.
    mask_meta: "MaskMetadata | None"
    outputs: torch.Tensor


class RadioEncoderCudaGraphRunner:
    """Bounded, exact-shape CUDA graphs for single-sequence RADIO inference.

    The caller only routes a single sequence here: either unmasked, or with
    ``cu_seqlens == [0, num_tokens]``, which the input shape fixes. Packed
    images, videos and shapes outside the capture budget run eagerly. Outputs
    are copied because callers may retain them across later replays.
    """

    def __init__(self, max_tokens: int, max_graphs: int):
        if max_tokens <= 0 or max_graphs <= 0:
            raise ValueError("RADIO graph token and cache limits must be positive")
        self.max_tokens = max_tokens
        self.max_graphs = max_graphs
        self.graphs: dict[tuple, _EncoderGraph] = {}

    def clear(self) -> None:
        self.graphs.clear()

    def __call__(
        self,
        forward: Callable[[torch.Tensor, "MaskMetadata | None"], torch.Tensor],
        inputs: torch.Tensor,
        mask_meta: "MaskMetadata | None" = None,
    ) -> torch.Tensor:
        if (
            not inputs.is_cuda
            or torch.is_grad_enabled()
            or torch.compiler.is_compiling()
            or inputs.numel() // inputs.shape[-1] > self.max_tokens
            or torch.cuda.is_current_stream_capturing()
        ):
            return forward(inputs, mask_meta)

        key = (tuple(inputs.shape), inputs.dtype, inputs.device, mask_meta is None)
        entry = self.graphs.get(key)
        if entry is None:
            if len(self.graphs) >= self.max_graphs:
                return forward(inputs, mask_meta)
            static_inputs = inputs.contiguous().clone()
            static_mask = mask_meta
            if mask_meta is not None:
                # Single sequence: cu_seqlens is [0, num_tokens] for every replay.
                static_mask = replace(
                    mask_meta, cu_seqlens=mask_meta.cu_seqlens.clone()
                )
            graph = torch.cuda.CUDAGraph()
            # vLLM's capture context also registers TP collective buffers.
            with graph_capture(device=inputs.device) as context:
                for _ in range(3):
                    forward(static_inputs, static_mask)
                context.stream.synchronize()
                with torch.cuda.graph(graph, stream=context.stream):
                    outputs = forward(static_inputs, static_mask)
            torch.cuda.current_stream(inputs.device).wait_stream(context.stream)
            entry = _EncoderGraph(graph, static_inputs, static_mask, outputs)
            self.graphs[key] = entry

        entry.inputs.copy_(inputs)
        entry.graph.replay()
        return entry.outputs.clone()
