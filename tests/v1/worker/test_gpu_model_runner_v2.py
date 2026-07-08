# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_gpu_model_runner_v2_video_preprocessing_profile_disabled():
    runner = object.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(multimodal_config=None)
    runner.supports_mm_inputs = False

    assert runner.profile_gpu_video_preprocessing_memory() == 0
