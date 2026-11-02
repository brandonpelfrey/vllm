# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, CLIPImageProcessor

from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.models.radio import (
    MaskMetadata,
    RadioLayerScale,
    RadioModel,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.radio import RadioConfig
from vllm.transformers_utils.repo_utils import hf_api
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

from ....conftest import ImageTestAssets

# we use snapshot_download to prevent conflicts between
# dynamic_module and trust_remote_code for hf_runner
DOWNLOAD_PATTERN = ["*.json", "*.py", "*.safetensors", "*.txt", "*.model"]

DEVICE_TYPE = current_platform.device_type


@torch.inference_mode()
def run_radio_test(
    image_assets: ImageTestAssets,
    model_id: str,
    *,
    dtype: str,
):
    model = hf_api().snapshot_download(model_id, allow_patterns=DOWNLOAD_PATTERN)
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

    img_processor = CLIPImageProcessor.from_pretrained(model)
    images = [asset.pil_image for asset in image_assets]
    pixel_values = [
        img_processor(image, return_tensors="pt").pixel_values.to(torch_dtype)
        for image in images
    ]

    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # RADIO model on HF does not properly handle torch_dtype argument
    # And relies on args["dtype"] which we have to patch manually:
    hf_config.args["dtype"] = torch_dtype

    hf_model = AutoModel.from_pretrained(
        model_id,
        config=hf_config,
        dtype=torch_dtype,
        trust_remote_code=True,
    ).to(DEVICE_TYPE)
    hf_model.eval()

    # A HF model has image normalization as a part of model's forward
    # However in vLLM we don't make normalization a part of the model
    # forward step since mean/std stored as model's parameters and
    # subject to precision loss (when using fp16/bf16) which negatively
    # affects evaluation benchmarks.
    hf_model.make_preprocessor_external()

    # Input resolution must be a multiple of the tower's ``min_resolution_step``
    # (patch_size * window_size), which differs across RADIO generations. Crop
    # down to the nearest supported resolution so one test body fits every
    # checkpoint; for C-RADIOv2-H this is a no-op (432x640, as before).
    radio = getattr(hf_model, "radio_model", hf_model)
    step = radio.min_resolution_step
    pixel_values = [
        pv[
            :,
            :,
            : (pv.shape[-2] // step) * step,
            : (min(pv.shape[-1], 640) // step) * step,
        ]
        for pv in pixel_values
    ]

    hf_outputs_per_image = [
        hf_model(pixel_value.to(DEVICE_TYPE)) for pixel_value in pixel_values
    ]

    vllm_config = RadioConfig(
        model_name=hf_config.args["model"],
        **hf_config.args,
    )
    vllm_model = RadioModel(vllm_config)
    loaded = vllm_model.load_weights(hf_model.state_dict())
    # Guard the remote-code -> vLLM key remap: every parameter must be loaded
    # from the checkpoint, except the LayerScale gains (layer_scale{1,2}.lambda1),
    # which are identity-init and may be absent from the remote-code checkpoint.
    expected = {
        name
        for name, _ in vllm_model.named_parameters()
        if not name.endswith((".layer_scale1.lambda1", ".layer_scale2.lambda1"))
    }
    missing = expected - loaded
    assert not missing, f"parameters not loaded from checkpoint: {sorted(missing)}"
    vllm_model = vllm_model.to(DEVICE_TYPE, torch_dtype)

    vllm_outputs_per_image = [
        vllm_model(pixel_values=pixel_value.to(DEVICE_TYPE))
        for pixel_value in pixel_values
    ]
    del vllm_model, hf_model
    cleanup_dist_env_and_memory()

    cos_similar = nn.CosineSimilarity(dim=-1)
    for vllm_output, hf_output in zip(vllm_outputs_per_image, hf_outputs_per_image):
        summary_sim = cos_similar(vllm_output[0], hf_output[0]).mean()
        feature_sim = cos_similar(vllm_output[1], hf_output[1]).mean()
        print(
            f"[radio] {model_id} {dtype} summary={summary_sim:.6f} "
            f"features={feature_sim:.6f}"
        )
        assert summary_sim > 0.99
        assert feature_sim > 0.99


@pytest.mark.parametrize(
    "model_id",
    [
        "nvidia/C-RADIOv2-H",
        "nvidia/C-RADIOv3-H",
        "nvidia/C-RADIOv4-H",
    ],
)
@pytest.mark.parametrize("dtype", ["half", "bfloat16"])
def test_radio(
    default_vllm_config, dist_init, image_assets, model_id, dtype: str
) -> None:
    run_radio_test(
        image_assets,
        model_id,
        dtype=dtype,
    )


def _radio_summary(*, teachers, cls_token_per_teacher):
    """Summary tensor from a tiny RadioModel with no encoder layers, so it
    builds no parallel layers and needs neither a GPU nor a checkpoint."""
    config = RadioConfig(
        model_name="vit_small_patch16_224",
        teachers=teachers,
        cls_token_per_teacher=cls_token_per_teacher,
    )
    model = RadioModel(config, num_hidden_layers_override=0)
    # Synthetic encoder output: [batch, num_skip + num_patches, hidden].
    y = torch.randn(2, model.embeddings.num_skip + 4, config.hidden_size)
    summary, _ = model._extract_final(y)
    return summary


def test_summary_idxs_no_teachers_keeps_class_tokens():
    # No teachers -> summary_idxs is None -> keep the class-token summary.
    summary = _radio_summary(teachers=[], cls_token_per_teacher=False)
    assert summary.shape[1] > 0


def test_summary_idxs_all_use_summary_false_is_empty():
    # Teachers present but none flagged use_summary -> empty selection.
    summary = _radio_summary(
        teachers=[{"name": "a", "use_summary": False}],
        cls_token_per_teacher=True,
    )
    assert summary.shape[1] == 0


def _native_state_dict_from_model(model: RadioModel) -> dict[str, torch.Tensor]:
    """Reverse-map a model's parameters onto native Transformers checkpoint keys
    (``encoder.layer.N.*`` with split ``attention.attention.{query,key,value}``
    and ``attention.output.dense``), so loading it must round-trip back to the
    original parameters."""
    state_dict: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        data = param.detach().clone()
        # The module tree already mirrors the native ``encoder.layer.N.*`` names
        # (including ``layer_scale{1,2}.lambda1``); only the fused attention
        # projections differ from the native split ones.
        if ".attention.qkv." in name:
            base, suffix = name.split(".attention.qkv.")
            query, key, value = data.chunk(3, dim=0)
            for proj, shard in (("query", query), ("key", key), ("value", value)):
                state_dict[f"{base}.attention.attention.{proj}.{suffix}"] = (
                    shard.clone()
                )
        elif ".attention.proj." in name:
            base, suffix = name.split(".attention.proj.")
            state_dict[f"{base}.attention.output.dense.{suffix}"] = data
        else:
            state_dict[name] = data
    return state_dict


def test_native_format_weight_loading(default_vllm_config, dist_init):
    # Native Transformers checkpoints nest the encoder under ``encoder.layer.N``
    # with split attention projections; loading must fuse q/k/v into the packed
    # ``qkv`` in the right order and land every other tensor in place, matching
    # the legacy layout the integration test covers.
    config = RadioConfig(model_name="vit_small_patch16_224")
    model = RadioModel(config, num_hidden_layers_override=2)
    # Uninitialized parameters may hold NaN, which never compares equal.
    with torch.no_grad():
        for param in model.parameters():
            param.normal_()

    # Every parameter (including layer_scale) must load from the native keys.
    expected = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    native_weights = _native_state_dict_from_model(model)
    # Keys both layouts intentionally drop must be skipped, not error.
    native_weights["input_conditioner.norm_mean"] = torch.zeros(3)
    native_weights["input_conditioner.norm_std"] = torch.ones(3)
    native_weights["summary_idxs"] = torch.zeros(1, dtype=torch.long)

    # Zero every parameter so a weight that fails to load is visible as zeros.
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    loaded = model.load_weights(native_weights)

    missing = set(expected) - loaded
    assert not missing, f"native weights not loaded: {sorted(missing)}"
    params = dict(model.named_parameters())
    for name, original in expected.items():
        assert torch.equal(params[name], original), (
            f"parameter mismatch after native load: {name}"
        )


def test_native_format_from_transformers(default_vllm_config, dist_init):
    # Real-checkpoint native-format round-trip: load a published (legacy)
    # checkpoint through Transformers' native ``RadioModel`` (which converts it to
    # the native layout), re-save it with ``save_original_format=False`` so the
    # on-disk checkpoint stays native -- the default re-applies the reverse
    # native->legacy mapping for backward compatibility -- then confirm vLLM's
    # ``RadioModel`` loads that native checkpoint with values matching the
    # reference. ``nvidia/C-RADIOv4-H`` loads on Transformers' config defaults and
    # is already downloaded by ``test_radio``.
    import transformers

    hf_radio_cls: Any = getattr(transformers, "RadioModel", None)
    if hf_radio_cls is None:
        pytest.skip("this transformers version has no native RadioModel")

    model_id = "nvidia/C-RADIOv4-H"
    hf_native = hf_radio_cls.from_pretrained(model_id, dtype=torch.float32)
    if (
        "save_original_format"
        not in inspect.signature(hf_native.save_pretrained).parameters
    ):
        pytest.skip("transformers save_pretrained has no save_original_format flag")

    with tempfile.TemporaryDirectory() as tmp:
        hf_native.save_pretrained(tmp, save_original_format=False)
        native_weights: dict[str, torch.Tensor] = {}
        for shard in sorted(Path(tmp).glob("*.safetensors")):
            native_weights.update(load_file(str(shard)))

    # Guard that the saved checkpoint is genuinely native-layout (split attention
    # under ``encoder.layer.N``), not the legacy ``radio_model.model.blocks.*``.
    assert any(".attention.q_proj." in name for name in native_weights), (
        "expected native-layout keys in the save_original_format=False checkpoint"
    )
    assert not any(name.startswith("radio_model.") for name in native_weights)

    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    vllm_config = RadioConfig(model_name=hf_config.args["model"], **hf_config.args)
    vllm_model = RadioModel(vllm_config)
    loaded = vllm_model.load_weights(native_weights)

    expected = {
        name
        for name, _ in vllm_model.named_parameters()
        if not name.endswith((".layer_scale1.lambda1", ".layer_scale2.lambda1"))
    }
    missing = expected - loaded
    assert not missing, f"native weights not loaded: {sorted(missing)}"

    # Values match the reference: the fused qkv is the concatenation of the
    # native split q/k/v_proj, and proj is the native o_proj.
    for name, param in vllm_model.named_parameters():
        if ".attention.qkv." in name:
            base, suffix = name.split(".attention.qkv.")
            ref = torch.cat(
                [
                    native_weights[f"{base}.attention.{proj}_proj.{suffix}"]
                    for proj in ("q", "k", "v")
                ],
                dim=0,
            )
        elif ".attention.proj." in name:
            base, suffix = name.split(".attention.proj.")
            ref = native_weights[f"{base}.attention.o_proj.{suffix}"]
        elif name in native_weights:
            ref = native_weights[name]
        else:
            continue
        assert torch.equal(param, ref), f"value mismatch after native load: {name}"


@pytest.mark.parametrize("value", [1.0, 0.75, 0.0])
@pytest.mark.parametrize("skip", [False, True])
def test_layer_scale_skip_preserves_nonunit_scales(value, skip):
    """Skipping exactly-one scales must not bypass trained LayerScale values."""
    config = RadioConfig(model_name="vit_huge_patch16_224")
    config.radio_encoder_skip_identity_scales = skip
    layer_scale = RadioLayerScale(config)
    with torch.no_grad():
        layer_scale.lambda1.fill_(value)
    layer_scale.refresh_identity()
    assert layer_scale.is_identity == (skip and value == 1.0)
    x = torch.randn(2, 4, config.hidden_size)
    torch.testing.assert_close(layer_scale(x), x * layer_scale.lambda1, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph test")
@pytest.mark.parametrize("masked", [False, True])
@torch.inference_mode()
def test_radio_graph_replay_uses_new_inputs_and_owns_outputs(dist_init, masked):
    """A replay cannot reuse previous image features or overwrite retained output."""
    from vllm.model_executor.models.radio_cudagraph import RadioEncoderCudaGraphRunner

    runner = RadioEncoderCudaGraphRunner(max_tokens=8, max_graphs=1)
    weights = torch.randn(8, 8, device="cuda")

    def single_sequence(x):
        if not masked:
            return None
        cu_seqlens = torch.tensor([0, x.shape[1]], dtype=torch.int32, device="cuda")
        return MaskMetadata(cu_seqlens=cu_seqlens, max_seqlen=torch.tensor(x.shape[1]))

    def expected(x):
        # The masked forward reads cu_seqlens on the device, like the attention does.
        return x @ weights + (x.shape[1] if masked else 0)

    def forward(x, mask_meta):
        assert (mask_meta is not None) == masked
        if mask_meta is None:
            return x @ weights
        assert mask_meta.cu_seqlens.numel() == 2
        return x @ weights + mask_meta.cu_seqlens[1].to(x.dtype)

    x = torch.randn(1, 4, 8, device="cuda")
    y = torch.randn_like(x)
    first = runner(forward, x, single_sequence(x))
    retained = first.clone()
    # Reuse any memory the capture freed; the replay must not read from it.
    clobber = [
        torch.full((2,), 1000, dtype=torch.int32, device="cuda") for _ in range(64)
    ]
    second = runner(forward, y, single_sequence(y))
    del clobber
    torch.testing.assert_close(first, retained, rtol=0, atol=0)
    torch.testing.assert_close(first, expected(x), rtol=0, atol=0)
    torch.testing.assert_close(second, expected(y), rtol=0, atol=0)
    assert len(runner.graphs) == 1
    for tokens in (5, 9):
        other_shape = torch.randn(1, tokens, 8, device="cuda")
        torch.testing.assert_close(
            runner(forward, other_shape, single_sequence(other_shape)),
            expected(other_shape),
        )
    assert len(runner.graphs) == 1
    runner.clear()
    assert not runner.graphs
