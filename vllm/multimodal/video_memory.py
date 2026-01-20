# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for calculating and managing video decoder VRAM requirements."""

from dataclasses import dataclass
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

@dataclass
class GPUAttributes:
    device_name: str
    sm_count: int
    max_threads_per_sm: int


def get_gpu_attributes(device_id: int = 0) -> GPUAttributes:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return GPUAttributes(
        device_name=props.name,
        sm_count=props.multi_processor_count,
        max_threads_per_sm=props.max_threads_per_multi_processor
    )


def align_to(value: int, alignment: int) -> int:
    """Align a value up to the next multiple of alignment."""
    return ((value + alignment - 1) // alignment) * alignment


def calculate_frame_buffer_size(
    width: int,
    height: int,
    bit_depth: int = 8,
    width_alignment: int = 128,
) -> dict:
    """
    Calculate frame buffer size based on resolution and format.

    Args:
        width: Video width in pixels
        height: Video height in pixels
        bit_depth: Bit depth (8 for NV12, 10/16 for P016)
        width_alignment: Width alignment in pixels (default: 128)

    Returns:
        Dictionary with size calculations
    """
    padded_width = align_to(width, width_alignment)
    bytes_per_component = 1 if bit_depth == 8 else 2

    # Assumed RGB output surface format
    # RGB: H × W × 3 bytes (or 6 for 16-bit)
    total_bytes = height * padded_width * 3 * bytes_per_component

    return total_bytes


def calculate_pynvvideocodec_per_thread_usage(
    width: int = 1920,
    height: int = 1080,
    batch_size: int = 8,
) -> int:
    """
    Estimate peak VRAM usage for internal PyNvVideoCodec decoder instance per thread.

    Args:
        width: Video width in pixels (default: 1920)
        height: Video height in pixels (default: 1080)
        batch_size: Batch size for get_batch_frames()

    Returns:
        Total VRAM usage in bytes for PyNvVideoCodec decoder instance per thread
    """

    # Calculate frame buffer size based on resolution
    frame_buffer_bytes = calculate_frame_buffer_size(
        width=width,
        height=height,
        bit_depth=8,
        width_alignment=128,
    )

    # Calculate thread memory component (in MiB). The below constants are particular to PyNvVideoCodec implementation.
    num_decode_surfaces = 12
    num_output_surfaces = 2
    per_thread_bytes = 2048

    gpu_attributes = get_gpu_attributes()
    thread_memory_bytes = gpu_attributes.sm_count * gpu_attributes.max_threads_per_sm * per_thread_bytes

    # Get number of decoders in use. Assume equal to hardware decoder count.
    import PyNvVideoCodec as nvc
    decoder_instances = nvc.GetDecoderCaps()['num_decoder_engines']

    # Calculate surface memory component, assume all decoders in use
    total_surfaces = num_decode_surfaces + num_output_surfaces + batch_size
    surface_memory_bytes = total_surfaces * frame_buffer_bytes * decoder_instances
    logger.info("Surface memory: %s GiB for %d decoder instances", format_gib(surface_memory_bytes), decoder_instances)

    # Total VRAM usage
    # PyNvVideoCodec has a fixed cost per context and additional cost per decoder for a given video format
    total_vram_bytes = thread_memory_bytes + surface_memory_bytes

    return total_vram_bytes


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
