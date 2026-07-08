# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
from PIL import Image

from vllm.multimodal.gpu_ipc_memory import GPUVideoFrames
from vllm.multimodal.parse import (
    ImageProcessorItems,
    MultiModalDataParser,
    VideoProcessorItems,
)

H, W = 480, 640


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", (W, H)),
        # HWC, e.g. from np.array(PIL.Image)
        np.zeros((H, W, 3), dtype=np.uint8),
        torch.zeros((H, W, 3), dtype=torch.uint8),
        # CHW, standard PyTorch / numpy convention
        np.zeros((3, H, W), dtype=np.uint8),
        torch.zeros((3, H, W), dtype=torch.uint8),
    ],
)
def test_image_size_hwc_chw(image):
    """Image sizes must be channel-layout agnostic.

    `get_image_size` determines the multimodal placeholder count; reading an
    HWC array (the layout `np.array(PIL.Image)` produces) as CHW yields a
    bogus size and a placeholder/embedding count mismatch at inference time.
    """
    items = ImageProcessorItems([image])

    assert items.get_image_size(0) == (W, H)


@pytest.mark.parametrize(
    "frame",
    [
        Image.new("RGB", (W, H)),
        np.zeros((H, W, 3), dtype=np.uint8),
        torch.zeros((H, W, 3), dtype=torch.uint8),
        np.zeros((3, H, W), dtype=np.uint8),
        torch.zeros((3, H, W), dtype=torch.uint8),
    ],
)
def test_frame_size_hwc_chw(frame):
    """`get_frame_size` must stay consistent with `get_image_size`."""
    items = VideoProcessorItems([[frame]])

    assert items.get_frame_size(0) == (W, H)


def test_parser_preserves_gpu_video_frame_wrapper():
    parser = MultiModalDataParser()
    frames = torch.zeros((2, H, W, 3), dtype=torch.uint8)
    metadata = {"frames_indices": [0, 1]}

    parsed_frames, parsed_metadata = parser._get_video_with_metadata(
        (GPUVideoFrames(frames), metadata)
    )

    assert parsed_frames is frames
    assert parsed_metadata is metadata


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_parser_preserves_cuda_video_tensor():
    parser = MultiModalDataParser()
    frames = torch.zeros((2, H, W, 3), dtype=torch.uint8, device="cuda")

    parsed_frames, parsed_metadata = parser._get_video_with_metadata(frames)

    assert parsed_frames is frames
    assert parsed_metadata is None
