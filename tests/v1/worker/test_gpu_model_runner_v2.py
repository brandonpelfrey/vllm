# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.multimodal.parse import VideoProcessorItems
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_gpu_model_runner_v2_video_preprocessing_profile_disabled():
    runner = object.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(multimodal_config=None)
    runner.supports_mm_inputs = False

    assert runner.profile_gpu_video_preprocessing_memory() == 0


@pytest.mark.parametrize("requires_metadata", [False, True])
def test_gpu_video_profile_preserves_processor_item_shape(
    monkeypatch: pytest.MonkeyPatch,
    requires_metadata: bool,
):
    runner = object.__new__(GPUModelRunner)
    mm_config = SimpleNamespace(
        media_io_kwargs={"video": {"keep_gpu_frames": True}},
        limit_per_prompt={},
    )
    runner.model_config = SimpleNamespace(
        multimodal_config=mm_config,
        max_model_len=128,
    )
    runner.supports_mm_inputs = True
    runner.device = torch.device("cpu")
    runner._allocated_bytes = lambda device: 0

    metadata = {"fps": 2.0}
    dummy_video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    dummy_item = (dummy_video, metadata) if requires_metadata else dummy_video
    processor_inputs = SimpleNamespace(
        mm_data_items={"video": VideoProcessorItems([dummy_item])},
        mm_uuid_items={},
    )
    captured = {}

    class Processor:
        info = SimpleNamespace(supports_gpu_video_preprocessing=True)
        dummy_inputs = SimpleNamespace(
            get_dummy_processor_inputs=lambda **kwargs: processor_inputs
        )

        def apply(self, inputs, timing_ctx):
            captured["item"] = inputs.mm_data_items["video"].get(0)
            return {}

    runner.mm_registry = SimpleNamespace(
        create_processor=lambda model_config: Processor()
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "reset_peak_memory_stats", lambda device: None
    )
    monkeypatch.setattr(torch.accelerator, "memory_stats", lambda device: {})
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    assert runner.profile_gpu_video_preprocessing_memory() == 0
    profile_item = captured["item"]
    assert isinstance(profile_item, tuple) is requires_metadata
    profile_video = profile_item[0] if requires_metadata else profile_item
    assert isinstance(profile_video, torch.Tensor)
