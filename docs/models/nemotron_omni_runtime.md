# Nemotron Nano Omni RADIO encoder runtime options

The Nemotron Nano Omni vision encoder (C-RADIO) supports three independent,
opt-in runtime optimizations. They keep the checkpoint, image preprocessing,
dense attention and visual-token counts unchanged.

```bash
vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
    --trust-remote-code \
    --hf-overrides '{
      "radio_encoder_cudagraph": true,
      "radio_encoder_cudagraph_max_tokens": 16384,
      "radio_encoder_skip_identity_scales": true,
      "radio_encoder_fused_residual_norm": true
    }'
```

Each boolean defaults to `false`; leave it out or set it to `false` to disable
that feature.

| HF override | Default | Behavior |
| --- | --- | --- |
| `radio_encoder_cudagraph` | `false` | Capture encoder inference for a single image (or an unmasked batch) at its exact input shape, then replay it. Each replay copies in the new input and returns an output the caller owns. |
| `radio_encoder_cudagraph_max_tokens` | `2048` | Maximum encoder tokens per captured input, including register and class tokens. The example above raises it to 16384. |
| `radio_encoder_cudagraph_cache_size` | `4` | Maximum number of captured shapes. New shapes beyond this limit run eagerly. Moving or reloading the encoder clears its graphs. |
| `radio_encoder_skip_identity_scales` | `false` | Skip a LayerScale multiplication, but only after checking that every loaded element is exactly one. Trained, non-unit scales stay active. |
| `radio_encoder_fused_residual_norm` | `false` | Fuse each residual addition with the LayerNorm that follows it. The sum is rounded to BF16 first, and the reduction order matches PyTorch's LayerNorm, so the output is unchanged. |

Graph capture falls back to ordinary execution for multi-image (packed)
requests, video, CPU inputs, autograd, outer graph capture, compilation and
shapes outside the limits. The first request with a new eligible shape pays a
warmup and capture cost, and each captured graph keeps its device memory.
These graphs are independent of the decoder's CUDA graphs.

The fused residual/LayerNorm kernel handles contiguous BF16 tensors of width
1280 with BF16 affine parameters and a positive epsilon. It runs on PyTorch
2.14 with CUDA SM80, SM90, SM100 or SM103, and
`tests/kernels/core/test_layernorm.py` checks it bit for bit against
PyTorch's LayerNorm. Other inputs, devices and builds, training, and
layers compiled with `compile_mm_encoder` use the unfused addition and
LayerNorm.

## Validation

Compare against the same build with the options off, using the same checkpoint
revision, prompts, media, generation parameters and GPU. Check that processed
geometry and prompt-token counts are identical. To catch stale graph inputs,
include different inputs with the same shape; to catch aliasing, keep outputs
across replays.

For latency, separate the first request of each shape (capture) from warmed
requests, and report raw samples, mean, median and tail latency.
