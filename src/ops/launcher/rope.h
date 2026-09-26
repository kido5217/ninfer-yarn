#pragma once

// ninfer::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/tensor.h"
#include "ninfer/ops/rope.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 cudaStream_t stream);

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        cudaStream_t stream);

// YaRN extension: keys on table presence (a null table is a wrapper error, not reached here).
// theta is retained for the call signature; the YaRN path reads the device table, not theta.
void rope_yarn_launch(const Tensor& positions, int rotary_dim, float theta, const YarnScale& yarn,
                      Tensor& q, Tensor& k, cudaStream_t stream);

void rope_yarn_single_launch(const Tensor& positions, int rotary_dim, float theta,
                             const YarnScale& yarn, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops::detail
