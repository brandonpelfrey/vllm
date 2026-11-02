// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "torch_utils.h"
#include <torch/csrc/stable/library.h>
#include <cuda_bf16.h>

namespace {

struct Moments {
  float mean, m2, count;
};

// Match the vector-of-four, four-warp Welford order of PyTorch 2.14's
// vectorized CUDA LayerNorm. The residual is rounded to BF16 before statistics.
__device__ Moments update(Moments m, float x) {
  const float delta = x - m.mean;
  const float count = m.count + 1.0f;
  const float mean = m.mean + delta * (1.0f / count);
  return {mean, m.m2 + delta * (x - mean), count};
}

__device__ Moments combine(Moments b, Moments a) {
  const float count = a.count + b.count;
  if (count == 0.0f) return {0.0f, 0.0f, 0.0f};
  const float delta = b.mean - a.mean;
  const float reciprocal = 1.0f / count;
  const float na = a.count * reciprocal;
  const float nb = b.count * reciprocal;
  // PyTorch's merge contracts to fma(na, a.mean, nb * b.mean); spell it out so
  // the compiler cannot fuse the other product instead.
  return {__fmaf_rn(na, a.mean, __fmul_rn(nb, b.mean)),
          a.m2 + b.m2 + delta * delta * a.count * nb, count};
}

__global__ void radio_add_layer_norm(const __nv_bfloat16* residual,
                                     const __nv_bfloat16* update_values,
                                     const __nv_bfloat16* weight,
                                     const __nv_bfloat16* bias,
                                     __nv_bfloat16* sum, __nv_bfloat16* output,
                                     float epsilon) {
  constexpr int width = 1280;
  constexpr int threads = 128;
  const int thread = threadIdx.x;
  const int lane = thread % 32;
  const int warp = thread / 32;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * width;
  Moments moments{0.0f, 0.0f, 0.0f};
  for (int vector = thread; vector < width / 4; vector += threads) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int64_t index = row + vector * 4 + item;
      const auto rounded =
          __float2bfloat16_rn(__bfloat162float(residual[index]) +
                              __bfloat162float(update_values[index]));
      sum[index] = rounded;
      moments = update(moments, __bfloat162float(rounded));
    }
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    Moments other{__shfl_down_sync(0xffffffff, moments.mean, offset),
                  __shfl_down_sync(0xffffffff, moments.m2, offset),
                  __shfl_down_sync(0xffffffff, moments.count, offset)};
    moments = combine(moments, other);
  }
  __shared__ float mean_m2[4];
  __shared__ float counts[2];
  for (int offset = 2; offset > 0; offset /= 2) {
    if (lane == 0 && warp >= offset && warp < 2 * offset) {
      const int target = warp - offset;
      mean_m2[2 * target] = moments.mean;
      mean_m2[2 * target + 1] = moments.m2;
      counts[target] = moments.count;
    }
    __syncthreads();
    if (lane == 0 && warp < offset) {
      moments = combine(
          moments, {mean_m2[2 * warp], mean_m2[2 * warp + 1], counts[warp]});
    }
    __syncthreads();
  }
  if (thread == 0) {
    mean_m2[0] = moments.mean;
    mean_m2[1] = moments.m2 / static_cast<float>(width);
  }
  __syncthreads();
  const float mean = mean_m2[0];
  const float inverse_std = rsqrtf(mean_m2[1] + epsilon);
  for (int column = thread; column < width; column += threads) {
    const float normalized =
        __bfloat162float(weight[column]) *
            (inverse_std * (__bfloat162float(sum[row + column]) - mean)) +
        __bfloat162float(bias[column]);
    output[row + column] = __float2bfloat16_rn(normalized);
  }
}

void radio_fused_residual_layer_norm(const torch::stable::Tensor& residual,
                                     const torch::stable::Tensor& update_values,
                                     const torch::stable::Tensor& weight,
                                     const torch::stable::Tensor& bias,
                                     torch::stable::Tensor& sum,
                                     torch::stable::Tensor& output,
                                     double epsilon) {
  STD_TORCH_CHECK(residual.dim() >= 2 && residual.size(-1) == 1280);
  STD_TORCH_CHECK(epsilon > 0);
  const torch::stable::Tensor* tensors[] = {&residual, &update_values, &weight,
                                            &bias,     &sum,           &output};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(tensor->is_cuda() && tensor->is_contiguous());
    STD_TORCH_CHECK(tensor->scalar_type() ==
                    torch::headeronly::ScalarType::BFloat16);
    STD_TORCH_CHECK(tensor->get_device_index() == residual.get_device_index());
  }
  STD_TORCH_CHECK(weight.numel() == 1280 && bias.numel() == 1280);
  STD_TORCH_CHECK(update_values.sizes() == residual.sizes());
  STD_TORCH_CHECK(sum.sizes() == residual.sizes());
  STD_TORCH_CHECK(output.sizes() == residual.sizes());
  if (residual.numel() == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(
      residual.get_device_index());
  radio_add_layer_norm<<<residual.numel() / 1280, 128, 0,
                         get_current_cuda_stream()>>>(
      static_cast<const __nv_bfloat16*>(residual.data_ptr()),
      static_cast<const __nv_bfloat16*>(update_values.data_ptr()),
      static_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<const __nv_bfloat16*>(bias.data_ptr()),
      static_cast<__nv_bfloat16*>(sum.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<float>(epsilon));
  const auto error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "radio_fused_residual_layer_norm(Tensor residual, Tensor update, "
      "Tensor weight, Tensor bias, Tensor! sum, Tensor! output, "
      "float epsilon) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("radio_fused_residual_layer_norm",
           TORCH_BOX(&radio_fused_residual_layer_norm));
}
