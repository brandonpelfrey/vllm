#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example demonstrating video VRAM profiling configuration.

This example shows how to configure vLLM for accurate VRAM profiling when
using hardware-accelerated video decoding.

NOTE: This script demonstrates configuration patterns. It requires a GPU
to actually run. The examples below show the configuration structure.
"""

def example_1_basic():
    """Example 1: Basic video profiling without preprocessing"""
    print("=" * 80)
    print("Example 1: Basic video profiling (1920x1080, 32 frames)")
    print("=" * 80)
    
    config = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        # VRAM profiling configuration (engine initialization only)
        "video_profiling": {
            "width": 1920,  # Raw decoded frame width
            "height": 1080,  # Raw decoded frame height
            "frames": 32,  # Number of frames per video
            # proc_width/proc_height will default to width/height
        },
        "maximum_concurrent_videos": 4,  # Max concurrent video decoding
        # Actual video processing at inference time
        "media_io_kwargs": {"video": {"num_frames": 32}},
        "limit_mm_per_prompt": {"video": 1},  # Only 1 video per prompt
        "gpu_memory_utilization": 0.9,
        "tensor_parallel_size": 1,
    }
    
    print("Configuration:")
    import json
    print(json.dumps(config, indent=2))
    print("\nEstimated VRAM for video: ~1.2 GB")
    print("  - Raw frames (1920x1080x32x4): 796 MB")
    print("  - Decoders (4x100MB): 400 MB")
    return config


def example_2_with_preprocessing():
    """Example 2: Video profiling with model preprocessing"""
    print("\n" + "=" * 80)
    print("Example 2: With preprocessing (1920x1080 → 512x512)")
    print("=" * 80)
    
    config = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        # VRAM profiling: account for both raw and preprocessed frames
        "video_profiling": {
            "width": 1920,  # Raw decoded frame dimensions
            "height": 1080,
            "frames": 32,
            "proc_width": 512,  # After model preprocessing
            "proc_height": 512,
            # This reserves VRAM for BOTH 1920x1080 AND 512x512 buffers
        },
        "maximum_concurrent_videos": 4,
        # Model-specific preprocessing parameters
        "mm_processor_kwargs": {"max_pixels": 512 * 512},
        "media_io_kwargs": {"video": {"num_frames": 32}},
        "limit_mm_per_prompt": {"video": 1},
        "gpu_memory_utilization": 0.9,
    }
    
    print("Configuration:")
    import json
    print(json.dumps(config, indent=2))
    print("\nEstimated VRAM for video: ~1.3 GB")
    print("  - Raw frames (1920x1080x32x4): 796 MB")
    print("  - Processed frames (512x512x32x4): 100 MB")
    print("  - Decoders (4x100MB): 400 MB")
    return config


def example_3_low_resolution():
    """Example 3: Low-resolution scenario"""
    print("\n" + "=" * 80)
    print("Example 3: Low-resolution (640x480, 16 frames, 8 concurrent)")
    print("=" * 80)
    
    config = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "video_profiling": {
            "width": 640,
            "height": 480,
            "frames": 16,
        },
        "maximum_concurrent_videos": 8,  # Can handle more concurrent videos
        "media_io_kwargs": {"video": {"num_frames": 16}},
        "limit_mm_per_prompt": {"video": 2},  # Allow 2 videos per prompt
    }
    
    print("Configuration:")
    import json
    print(json.dumps(config, indent=2))
    print("\nEstimated VRAM for video: ~0.6 GB")
    print("  - Raw frames (640x480x16x8): 118 MB")
    print("  - Decoders (8x100MB): 800 MB")
    return config


def example_4_high_resolution():
    """Example 4: High-resolution 4K scenario"""
    print("\n" + "=" * 80)
    print("Example 4: High-resolution 4K (3840x2160 → 1024x1024)")
    print("=" * 80)
    
    config = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "video_profiling": {
            "width": 3840,  # 4K resolution
            "height": 2160,
            "frames": 64,  # More frames
            "proc_width": 1024,
            "proc_height": 1024,
        },
        "maximum_concurrent_videos": 2,  # Fewer concurrent due to high res
        "gpu_memory_utilization": 0.95,  # Use more GPU memory
        "media_io_kwargs": {"video": {"num_frames": 64}},
    }
    
    print("Configuration:")
    import json
    print(json.dumps(config, indent=2))
    print("\nEstimated VRAM for video: ~2.5 GB")
    print("  - Raw frames (3840x2160x64x2): 1.58 GB")
    print("  - Processed frames (1024x1024x64x2): 402 MB")
    print("  - Decoders (2x100MB): 200 MB")
    return config


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("VIDEO VRAM PROFILING CONFIGURATION EXAMPLES")
    print("=" * 80)
    print("\nThese examples show how to configure video profiling for different")
    print("scenarios. To actually use them, instantiate an LLM with these configs:")
    print("\n  from vllm import LLM")
    print("  llm = LLM(**config)")
    print("\nNote: Requires a system with CUDA GPU.\n")
    
    configs = []
    configs.append(example_1_basic())
    configs.append(example_2_with_preprocessing())
    configs.append(example_3_low_resolution())
    configs.append(example_4_high_resolution())
    
    print("\n" + "=" * 80)
    print("CLI USAGE EXAMPLES")
    print("=" * 80)
    
    print("\nExample 1 (Basic):")
    print("vllm serve Qwen/Qwen2.5-VL-3B-Instruct \\")
    print('  --video-profiling \'{"width": 1920, "height": 1080, "frames": 32}\' \\')
    print("  --maximum-concurrent-videos 4")
    
    print("\nExample 2 (With preprocessing):")
    print("vllm serve Qwen/Qwen2.5-VL-3B-Instruct \\")
    print('  --video-profiling \'{"width": 1920, "height": 1080, "frames": 32, "proc_width": 512, "proc_height": 512}\' \\')
    print("  --maximum-concurrent-videos 4 \\")
    print('  --mm-processor-kwargs \'{"max_pixels": 262144}\'')
    
    print("\n" + "=" * 80)
    print("For more information, see:")
    print("  docs/features/video_memory_profiling.md")
    print("=" * 80)
    
    return configs


if __name__ == "__main__":
    main()
    
    # Uncomment below to actually instantiate an LLM (requires GPU):
    """
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        video_profiling={
            "width": 1920,
            "height": 1080,
            "frames": 32,
            "proc_width": 512,
            "proc_height": 512,
        },
        maximum_concurrent_videos=4,
        media_io_kwargs={"video": {"num_frames": 32}},
        limit_mm_per_prompt={"video": 1},
    )
    
    # Generate with video
    prompt = "Describe what happens in this video."
    video_url = "https://example.com/video.mp4"
    
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"video": [video_url]},
    })
    
    print(outputs[0].outputs[0].text)
    """
