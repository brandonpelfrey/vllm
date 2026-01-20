# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for video memory calculation utilities."""

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.multimodal import VideoProfilingConfig
from vllm.multimodal.video_memory import (
    DEFAULT_MAX_CONCURRENT_VIDEOS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_NUM_FRAMES,
    DEFAULT_VIDEO_WIDTH,
    allocate_video_memory_stress,
    calculate_decoder_memory_bytes,
    calculate_total_video_memory_bytes,
    calculate_video_frame_memory_bytes,
    get_video_memory_config_from_vllm_config,
)


def test_video_profiling_config_defaults():
    """Test VideoProfilingConfig defaults proc_width/height to width/height."""
    config = VideoProfilingConfig(width=1920, height=1080, frames=32)
    
    assert config.width == 1920
    assert config.height == 1080
    assert config.frames == 32
    assert config.proc_width == 1920  # Should default to width
    assert config.proc_height == 1080  # Should default to height


def test_video_profiling_config_explicit():
    """Test VideoProfilingConfig with explicit proc dimensions."""
    config = VideoProfilingConfig(
        width=1920,
        height=1080,
        frames=32,
        proc_width=512,
        proc_height=512,
    )
    
    assert config.width == 1920
    assert config.height == 1080
    assert config.frames == 32
    assert config.proc_width == 512
    assert config.proc_height == 512


def test_calculate_decoder_memory_bytes():
    """Test decoder memory calculation."""
    # Test with default num_decoders
    memory = calculate_decoder_memory_bytes(max_concurrent_videos=4)
    assert memory == 4 * 100 * 1024 * 1024  # 4 decoders * 100 MB each
    
    # Test with explicit num_decoders
    memory = calculate_decoder_memory_bytes(
        max_concurrent_videos=4,
        num_decoders=2,
    )
    assert memory == 2 * 100 * 1024 * 1024  # 2 decoders * 100 MB each


def test_calculate_video_frame_memory_bytes():
    """Test video frame memory calculation."""
    width = 1920
    height = 1080
    num_frames = 32
    max_concurrent_videos = 4
    bytes_per_pixel = 3  # RGB
    
    memory = calculate_video_frame_memory_bytes(
        width=width,
        height=height,
        num_frames=num_frames,
        max_concurrent_videos=max_concurrent_videos,
        bytes_per_pixel=bytes_per_pixel,
    )
    
    expected = width * height * bytes_per_pixel * num_frames * max_concurrent_videos
    assert memory == expected


def test_calculate_total_video_memory_bytes():
    """Test total video memory calculation."""
    width = 1920
    height = 1080
    num_frames = 32
    max_concurrent_videos = 4
    
    total_memory = calculate_total_video_memory_bytes(
        width=width,
        height=height,
        num_frames=num_frames,
        max_concurrent_videos=max_concurrent_videos,
    )
    
    # Should be sum of decoder and frame memory
    decoder_memory = calculate_decoder_memory_bytes(max_concurrent_videos)
    frame_memory = calculate_video_frame_memory_bytes(
        width, height, num_frames, max_concurrent_videos
    )
    
    assert total_memory == decoder_memory + frame_memory


def test_get_video_memory_config_no_multimodal():
    """Test config extraction when multimodal is not configured."""
    from unittest.mock import MagicMock
    
    # Create a mock VllmConfig without multimodal config
    vllm_config = MagicMock()
    vllm_config.model_config.multimodal_config = None
    
    config = get_video_memory_config_from_vllm_config(vllm_config)
    
    assert config["needs_profiling"] is False
    assert config["width"] == 0
    assert config["height"] == 0
    assert config["num_frames"] == 0
    assert config["max_concurrent_videos"] == 0


def test_get_video_memory_config_with_profiling():
    """Test config extraction with video profiling config."""
    from unittest.mock import MagicMock
    
    # Create VideoProfilingConfig
    video_prof = VideoProfilingConfig(
        width=1920,
        height=1080,
        frames=32,
    )
    
    # Create a mock VllmConfig with video profiling
    vllm_config = MagicMock()
    mm_config = MagicMock()
    mm_config.video_profiling = video_prof
    mm_config.max_concurrent_videos = None
    vllm_config.model_config.multimodal_config = mm_config
    
    config = get_video_memory_config_from_vllm_config(vllm_config)
    
    assert config["needs_profiling"] is True
    assert config["width"] == 1920
    assert config["height"] == 1080
    assert config["num_frames"] == 32
    # proc_width/height should default to width/height
    assert config["proc_width"] == 1920
    assert config["proc_height"] == 1080
    assert config["max_concurrent_videos"] == DEFAULT_MAX_CONCURRENT_VIDEOS


def test_get_video_memory_config_with_preprocessing():
    """Test config extraction with preprocessing dimensions."""
    from unittest.mock import MagicMock
    
    # Create VideoProfilingConfig with preprocessing dimensions
    video_prof = VideoProfilingConfig(
        width=1920,
        height=1080,
        frames=32,
        proc_width=512,
        proc_height=512,
    )
    
    # Create a mock VllmConfig
    vllm_config = MagicMock()
    mm_config = MagicMock()
    mm_config.video_profiling = video_prof
    mm_config.max_concurrent_videos = 8
    vllm_config.model_config.multimodal_config = mm_config
    
    config = get_video_memory_config_from_vllm_config(vllm_config)
    
    assert config["needs_profiling"] is True
    assert config["width"] == 1920
    assert config["height"] == 1080
    assert config["num_frames"] == 32
    assert config["proc_width"] == 512
    assert config["proc_height"] == 512
    assert config["max_concurrent_videos"] == 8


def test_get_video_memory_config_no_profiling_config():
    """Test that without video_profiling config, profiling is skipped."""
    from unittest.mock import MagicMock
    
    # Create a mock VllmConfig without video_profiling
    vllm_config = MagicMock()
    mm_config = MagicMock()
    mm_config.video_profiling = None
    mm_config.max_concurrent_videos = 4
    vllm_config.model_config.multimodal_config = mm_config
    
    config = get_video_memory_config_from_vllm_config(vllm_config)
    
    assert config["needs_profiling"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_allocate_video_memory_stress():
    """Test video memory stress allocation."""
    device = torch.device("cuda:0")
    width = 640
    height = 480
    num_frames = 16
    max_concurrent_videos = 2
    
    tensors = allocate_video_memory_stress(
        width=width,
        height=height,
        num_frames=num_frames,
        max_concurrent_videos=max_concurrent_videos,
        device=device,
    )
    
    # Should allocate one tensor per concurrent video
    assert len(tensors) == max_concurrent_videos
    
    # Each tensor should have correct shape
    for tensor in tensors:
        assert tensor.shape == (num_frames, height, width, 3)
        assert tensor.dtype == torch.uint8
        assert tensor.device == device
    
    # Clean up
    del tensors
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_allocate_video_memory_stress_oom_handling():
    """Test that OOM during stress allocation is handled gracefully."""
    device = torch.device("cuda:0")
    
    # Try to allocate an unreasonably large amount
    # This should fail gracefully and return empty list
    tensors = allocate_video_memory_stress(
        width=100000,  # Very large
        height=100000,  # Very large
        num_frames=1000,
        max_concurrent_videos=100,
        device=device,
    )
    
    # Should return empty list on OOM
    assert len(tensors) == 0


def test_video_memory_breakdown_calculations():
    """Test that video memory breakdown calculations are correct."""
    # Test case 1: Basic configuration without preprocessing
    width = 1920
    height = 1080
    num_frames = 32
    max_concurrent_videos = 4
    
    raw_frame_memory = calculate_video_frame_memory_bytes(
        width=width,
        height=height,
        num_frames=num_frames,
        max_concurrent_videos=max_concurrent_videos,
        bytes_per_pixel=3,
    )
    
    decoder_memory = calculate_decoder_memory_bytes(
        max_concurrent_videos=max_concurrent_videos,
    )
    
    # When proc dimensions match raw dimensions, proc_frame_memory should be 0
    proc_frame_memory = 0
    total = raw_frame_memory + proc_frame_memory + decoder_memory
    
    # Verify raw frame memory calculation
    expected_raw = width * height * 3 * num_frames * max_concurrent_videos
    assert raw_frame_memory == expected_raw
    
    # Verify decoder memory calculation
    expected_decoder = 100 * 1024 * 1024 * max_concurrent_videos
    assert decoder_memory == expected_decoder
    
    # Verify total
    assert total == expected_raw + expected_decoder
    
    # Test case 2: With preprocessing
    proc_width = 512
    proc_height = 512
    
    proc_frame_memory = calculate_video_frame_memory_bytes(
        width=proc_width,
        height=proc_height,
        num_frames=num_frames,
        max_concurrent_videos=max_concurrent_videos,
        bytes_per_pixel=3,
    )
    
    total_with_proc = raw_frame_memory + proc_frame_memory + decoder_memory
    
    # Verify processed frame memory calculation
    expected_proc = proc_width * proc_height * 3 * num_frames * max_concurrent_videos
    assert proc_frame_memory == expected_proc
    
    # Verify total includes both raw and processed
    assert total_with_proc == expected_raw + expected_proc + expected_decoder
    assert total_with_proc > total  # Should be larger with preprocessing
