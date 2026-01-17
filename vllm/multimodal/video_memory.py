# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for calculating and managing video decoder VRAM requirements."""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib

logger = init_logger(__name__)

# Default values for video memory profiling when not specified
DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080
DEFAULT_VIDEO_NUM_FRAMES = 32
DEFAULT_MAX_CONCURRENT_VIDEOS = 4


def calculate_decoder_memory_bytes(
    max_concurrent_videos: int,
    num_decoders: int | None = None,
) -> int:
    """
    Calculate VRAM used by PyNvVideoCodec decoder instances.
    
    Args:
        max_concurrent_videos: Maximum number of concurrent video decoding operations
        num_decoders: Number of decoder instances (defaults to max_concurrent_videos)
    
    Returns:
        Estimated VRAM usage in bytes for decoder instances
    
    Note:
        Each PyNvVideoCodec decoder instance uses approximately 50-100 MB of VRAM
        depending on the video codec and resolution. This is a conservative estimate.
    """
    if num_decoders is None:
        num_decoders = max_concurrent_videos
    
    # Conservative estimate: 100 MB per decoder instance
    bytes_per_decoder = 100 * 1024 * 1024  # 100 MB
    total_decoder_memory = num_decoders * bytes_per_decoder
    
    logger.debug(
        "Estimated decoder memory: %s GiB for %d decoders",
        format_gib(total_decoder_memory),
        num_decoders,
    )
    
    return total_decoder_memory


def calculate_video_frame_memory_bytes(
    width: int,
    height: int,
    num_frames: int,
    max_concurrent_videos: int,
    bytes_per_pixel: int = 3,
) -> int:
    """
    Calculate VRAM required for decoded video frames.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        num_frames: Number of frames per video
        max_concurrent_videos: Maximum number of videos processed concurrently
        bytes_per_pixel: Bytes per pixel (3 for RGB, 4 for RGBA)
    
    Returns:
        Estimated VRAM usage in bytes for video frames
    """
    # Calculate memory for one frame
    bytes_per_frame = width * height * bytes_per_pixel
    
    # Total memory for all frames across all concurrent videos
    total_frame_memory = bytes_per_frame * num_frames * max_concurrent_videos
    
    logger.debug(
        "Estimated frame memory: %s GiB for %dx%d resolution, "
        "%d frames, %d concurrent videos",
        format_gib(total_frame_memory),
        width,
        height,
        num_frames,
        max_concurrent_videos,
    )
    
    return total_frame_memory


def calculate_total_video_memory_bytes(
    width: int,
    height: int,
    num_frames: int,
    max_concurrent_videos: int,
    bytes_per_pixel: int = 3,
) -> int:
    """
    Calculate total VRAM required for video decoding including both
    decoder instances and decoded frames.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        num_frames: Number of frames per video
        max_concurrent_videos: Maximum number of videos processed concurrently
        bytes_per_pixel: Bytes per pixel (3 for RGB, 4 for RGBA)
    
    Returns:
        Total estimated VRAM usage in bytes
    """
    decoder_memory = calculate_decoder_memory_bytes(max_concurrent_videos)
    frame_memory = calculate_video_frame_memory_bytes(
        width, height, num_frames, max_concurrent_videos, bytes_per_pixel
    )
    
    total_memory = decoder_memory + frame_memory
    
    logger.info(
        "Total estimated video decoder VRAM: %s GiB "
        "(decoders: %s GiB, frames: %s GiB)",
        format_gib(total_memory),
        format_gib(decoder_memory),
        format_gib(frame_memory),
    )
    
    return total_memory


def get_video_memory_config_from_vllm_config(
    vllm_config: VllmConfig,
) -> dict[str, Any]:
    """
    Extract video memory configuration from VllmConfig.
    
    Uses the --video-profiling configuration if provided, otherwise returns
    disabled profiling. Logs assumptions made.
    
    Args:
        vllm_config: The vLLM configuration object
    
    Returns:
        Dictionary with keys:
            - width: Raw decoded video frame width
            - height: Raw decoded video frame height
            - num_frames: Number of frames per video
            - proc_width: Frame width after model preprocessing
            - proc_height: Frame height after model preprocessing
            - max_concurrent_videos: Max concurrent video operations
            - needs_profiling: Whether video VRAM profiling is needed
    """
    mm_config = vllm_config.model_config.multimodal_config
    
    if mm_config is None:
        return {
            "width": 0,
            "height": 0,
            "num_frames": 0,
            "proc_width": 0,
            "proc_height": 0,
            "max_concurrent_videos": 0,
            "needs_profiling": False,
        }
    
    # Check if video_profiling config is provided
    video_profiling = mm_config.video_profiling
    
    if video_profiling is None:
        # No video profiling config, skip profiling
        logger.debug(
            "No --video-profiling configuration provided, skipping video VRAM profiling"
        )
        return {
            "width": 0,
            "height": 0,
            "num_frames": 0,
            "proc_width": 0,
            "proc_height": 0,
            "max_concurrent_videos": 0,
            "needs_profiling": False,
        }
    
    # Extract parameters from VideoProfilingConfig
    width = video_profiling.width
    height = video_profiling.height
    num_frames = video_profiling.frames
    proc_width = video_profiling.proc_width
    proc_height = video_profiling.proc_height
    
    # Get max_concurrent_videos from config with default
    max_concurrent_videos = mm_config.max_concurrent_videos
    if max_concurrent_videos is None:
        max_concurrent_videos = DEFAULT_MAX_CONCURRENT_VIDEOS
        logger.info(
            "Video VRAM profiling: max_concurrent_videos not specified, "
            "using default: %d",
            max_concurrent_videos,
        )
    
    logger.info(
        "Video VRAM profiling configuration: %dx%d raw resolution, "
        "%dx%d processed resolution, %d frames, %d max concurrent videos",
        width,
        height,
        proc_width,
        proc_height,
        num_frames,
        max_concurrent_videos,
    )
    
    return {
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "proc_width": proc_width,
        "proc_height": proc_height,
        "max_concurrent_videos": max_concurrent_videos,
        "needs_profiling": True,
    }


def allocate_video_memory_stress(
    width: int,
    height: int,
    num_frames: int,
    max_concurrent_videos: int,
    device: torch.device,
    proc_width: int | None = None,
    proc_height: int | None = None,
) -> list[torch.Tensor]:
    """
    Allocate tensors on GPU to stress VRAM during memory profiling.
    
    This simulates the VRAM usage of video decoding operations including
    both raw decoded frames and preprocessed frames.
    
    Args:
        width: Raw decoded frame width in pixels
        height: Raw decoded frame height in pixels
        num_frames: Number of frames per video
        max_concurrent_videos: Maximum number of videos processed concurrently
        device: CUDA device to allocate tensors on
        proc_width: Frame width after preprocessing (None if no preprocessing)
        proc_height: Frame height after preprocessing (None if no preprocessing)
    
    Returns:
        List of allocated tensors (keep references to prevent deallocation)
    """
    allocated_tensors = []
    
    try:
        # Allocate raw decoded frame buffers for concurrent videos
        # Each video has num_frames of shape (height, width, 3) in uint8
        for i in range(max_concurrent_videos):
            frame_tensor = torch.empty(
                (num_frames, height, width, 3),
                dtype=torch.uint8,
                device=device,
            )
            allocated_tensors.append(frame_tensor)
            
            logger.debug(
                "Allocated raw video frame buffer %d/%d: %s",
                i + 1,
                max_concurrent_videos,
                tuple(frame_tensor.shape),
            )
        
        # If preprocessing changes dimensions, allocate processed frame buffers
        if proc_width is not None and proc_height is not None:
            if proc_width != width or proc_height != height:
                for i in range(max_concurrent_videos):
                    proc_tensor = torch.empty(
                        (num_frames, proc_height, proc_width, 3),
                        dtype=torch.uint8,
                        device=device,
                    )
                    allocated_tensors.append(proc_tensor)
                    
                    logger.debug(
                        "Allocated preprocessed video frame buffer %d/%d: %s",
                        i + 1,
                        max_concurrent_videos,
                        tuple(proc_tensor.shape),
                    )
        
        # Calculate actual allocated memory
        total_allocated = sum(
            tensor.element_size() * tensor.numel()
            for tensor in allocated_tensors
        )
        
        logger.info(
            "Allocated %s GiB for video frame stress testing (%d tensors)",
            format_gib(total_allocated),
            len(allocated_tensors),
        )
        
    except RuntimeError as e:
        logger.warning(
            "Failed to allocate video stress tensors: %s. "
            "Video VRAM estimation may be inaccurate.",
            str(e),
        )
        # Clean up any partially allocated tensors
        allocated_tensors.clear()
    
    return allocated_tensors
