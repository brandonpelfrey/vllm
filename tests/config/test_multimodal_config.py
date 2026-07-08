# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.platforms as platforms
from vllm.config.model import ModelConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_mm_encoder_attn_backend_str_conversion():
    config = MultiModalConfig(mm_encoder_attn_backend="FLASH_ATTN")
    assert config.mm_encoder_attn_backend == AttentionBackendEnum.FLASH_ATTN


def test_mm_encoder_attn_backend_invalid():
    with pytest.raises(ValueError):
        MultiModalConfig(mm_encoder_attn_backend="not_a_backend")


def test_mm_encoder_attn_backend_hash_updates():
    base_hash = MultiModalConfig().compute_hash()
    overridden_hash = MultiModalConfig(
        mm_encoder_attn_backend=AttentionBackendEnum.FLASH_ATTN
    ).compute_hash()
    assert base_hash != overridden_hash


def test_language_model_only_does_not_affect_mm_hash():
    """language_model_only does not affect the ViT computation graph,
    so it should not change the multimodal config hash."""
    base_hash = MultiModalConfig().compute_hash()
    lm_only_hash = MultiModalConfig(language_model_only=True).compute_hash()
    assert base_hash == lm_only_hash


def test_language_model_only_affects_model_hash():
    """language_model_only affects the LM computation graph,
    so it should change the model config hash."""
    model = "llava-hf/llava-1.5-7b-hf"
    base_hash = ModelConfig(model).compute_hash()
    lm_only_hash = ModelConfig(model, language_model_only=True).compute_hash()
    assert base_hash != lm_only_hash


def test_mm_encoder_fp8_scale_path_requires_fp8():
    with pytest.raises(ValueError, match="mm_encoder_attn_dtype"):
        MultiModalConfig(mm_encoder_fp8_scale_path="/tmp/scales.json")


def test_mm_encoder_attn_dtype_hash_updates(tmp_path):
    scale_file = tmp_path / "scales.json"
    scale_file.write_text("{}")
    base_hash = MultiModalConfig().compute_hash()
    fp8_hash = MultiModalConfig(mm_encoder_attn_dtype="fp8").compute_hash()
    fp8_static_hash = MultiModalConfig(
        mm_encoder_attn_dtype="fp8",
        mm_encoder_fp8_scale_path=str(scale_file),
    ).compute_hash()
    assert base_hash != fp8_hash
    assert fp8_hash != fp8_static_hash


def _set_cuda_platform(monkeypatch: pytest.MonkeyPatch, is_cuda: bool) -> None:
    monkeypatch.setattr(
        platforms,
        "current_platform",
        type("Platform", (), {"is_cuda": staticmethod(lambda: is_cuda)})(),
    )


def test_keep_gpu_frames_config_is_valid_with_required_options(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_cuda_platform(monkeypatch, True)

    config = MultiModalConfig(
        media_io_kwargs={
            "video": {
                "backend": "pynvvideocodec",
                "keep_gpu_frames": True,
            }
        },
        mm_tensor_ipc="torch_shm",
        mm_ipc_gpu_memory_gb=1,
    )

    assert config.media_io_kwargs["video"]["keep_gpu_frames"] is True


def test_keep_gpu_frames_config_requires_pynvvideocodec(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_cuda_platform(monkeypatch, True)

    with pytest.raises(ValueError, match="backend']='pynvvideocodec"):
        MultiModalConfig(
            media_io_kwargs={"video": {"backend": "opencv", "keep_gpu_frames": True}},
            mm_tensor_ipc="torch_shm",
            mm_ipc_gpu_memory_gb=1,
        )


def test_keep_gpu_frames_config_requires_cuda(monkeypatch: pytest.MonkeyPatch):
    _set_cuda_platform(monkeypatch, False)

    with pytest.raises(ValueError, match="CUDA-only"):
        MultiModalConfig(
            media_io_kwargs={
                "video": {
                    "backend": "pynvvideocodec",
                    "keep_gpu_frames": True,
                }
            },
            mm_tensor_ipc="torch_shm",
            mm_ipc_gpu_memory_gb=1,
        )


def test_keep_gpu_frames_config_requires_torch_shm(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_cuda_platform(monkeypatch, True)

    with pytest.raises(ValueError, match="--mm-tensor-ipc torch_shm"):
        MultiModalConfig(
            media_io_kwargs={
                "video": {
                    "backend": "pynvvideocodec",
                    "keep_gpu_frames": True,
                }
            },
            mm_tensor_ipc="direct_rpc",
            mm_ipc_gpu_memory_gb=1,
        )


def test_keep_gpu_frames_config_requires_gpu_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_cuda_platform(monkeypatch, True)

    with pytest.raises(ValueError, match="--mm-ipc-gpu-memory-gb"):
        MultiModalConfig(
            media_io_kwargs={
                "video": {
                    "backend": "pynvvideocodec",
                    "keep_gpu_frames": True,
                }
            },
            mm_tensor_ipc="torch_shm",
        )


def test_gpu_video_preprocessing_profile_bytes_do_not_affect_hash():
    base_hash = MultiModalConfig().compute_hash()
    profiled_hash = MultiModalConfig(
        mm_gpu_video_preprocessing_bytes_per_frame=1024
    ).compute_hash()

    assert profiled_hash == base_hash
