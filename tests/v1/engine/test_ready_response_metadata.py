# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import msgspec

from vllm.v1.engine import EngineCoreReadyResponse
from vllm.v1.engine.core_client import MPClient


def _ready_response_payload(
    *,
    mm_gpu_video_preprocessing_bytes_per_frame: int = 0,
) -> bytes:
    return msgspec.msgpack.encode(
        EngineCoreReadyResponse(
            max_model_len=8192,
            num_gpu_blocks=100,
            block_size=1056,
            dp_stats_address=None,
            dtype="bfloat16",
            vllm_version="test",
            world_size=1,
            data_parallel_size=1,
            mm_gpu_video_preprocessing_bytes_per_frame=(
                mm_gpu_video_preprocessing_bytes_per_frame
            ),
        )
    )


def test_apply_ready_response_syncs_gpu_video_preprocessing_bytes():
    mm_config = SimpleNamespace(mm_gpu_video_preprocessing_bytes_per_frame=32)
    client = object.__new__(MPClient)
    client.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, num_gpu_blocks=0),
        model_config=SimpleNamespace(max_model_len=8192, multimodal_config=mm_config),
    )
    client.stats_update_address = None

    client._apply_ready_response(
        _ready_response_payload(mm_gpu_video_preprocessing_bytes_per_frame=64)
    )

    assert mm_config.mm_gpu_video_preprocessing_bytes_per_frame == 64


def test_apply_ready_response_preserves_larger_local_gpu_video_profile():
    mm_config = SimpleNamespace(mm_gpu_video_preprocessing_bytes_per_frame=128)
    client = object.__new__(MPClient)
    client.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, num_gpu_blocks=0),
        model_config=SimpleNamespace(max_model_len=8192, multimodal_config=mm_config),
    )
    client.stats_update_address = None

    client._apply_ready_response(
        _ready_response_payload(mm_gpu_video_preprocessing_bytes_per_frame=64)
    )

    assert mm_config.mm_gpu_video_preprocessing_bytes_per_frame == 128
