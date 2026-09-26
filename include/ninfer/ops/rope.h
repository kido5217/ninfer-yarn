#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>
#include <vector>

namespace ninfer::ops {

// Device-resident YaRN extension state for rope. `inv_freq` points to `rotary_dim / 2` fp32
// frequencies in device memory (one per pair); `mscale` is the fp32 attention magnitude applied
// to cos/sin after the trig. A null `inv_freq` denotes the unextended (pure power-law) path.
struct YarnScale {
    const float* inv_freq; // device-resident, null = unextended
    float mscale;
};

// Result of the host-side YaRN table construction (see compute_rope_yarn_table).
struct YarnTable {
    std::vector<float> inv_freq; // rotary_dim / 2 fp32 values
    float mscale;
};

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. The per-pair inverse frequency is the pure
 * power law theta^(-2*i/rotary_dim) unless a YaRN table is supplied (see below). Supported modes
 * are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   DFlash full-head domain head_dim=rotary_dim=128; phi=positions[t]*inv_freq[i].
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * YaRN extension: when a device-resident frequency table is supplied (the YarnScale overloads),
 * the per-pair inverse frequency is read from the table (computed host-side by
 * compute_rope_yarn_table) and the attention magnitude `mscale` is applied to cos/sin after the
 * trig, i.e. (sin,cos) -> (mscale*sin, mscale*cos). The table is a represented input; the
 * rotation is otherwise identical to the power-law path. At extension factor s == 1 the table
 * degenerates to the power law and mscale == 1, so the path is structurally inert. The YaRN path
 * dispatches on table PRESENCE (a null table never selects it), not on theta.
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. The registered optimized domains are D256/R64 Text Q/K head geometries
 * 24/4 and 16/2, D128/R128 1-D Text geometry 32/8, plus Vision geometry 16/16. q and k must not
 * overlap one another or positions. The Op mutates only dimensions [0,rotary_dim) of the supplied
 * Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs (for the YaRN path, positions + the fp32 inv_freq table + mscale), modeling
 * the fp32 angle quantization (phi = float(p * inv_freq[i]) before the FP64 trig). The updated
 * BF16 values are promoted and compared directly with that result; output storage rounding belongs
 * to the Op's numerical criterion, not the oracle. Unrotated dimensions remain bit-exact. Private
 * kernel arithmetic is implementation-defined. The Op uses no workspace or persistent state.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);

// YaRN extension pair form: the rotation reads the device-resident per-pair frequency table and
// applies `yarn.mscale` to cos/sin. `yarn.inv_freq` must be non-null and point to
// `rotary_dim / 2` device-resident fp32 values. theta is retained for the call signature but the
// YaRN path keys on table presence, not theta.
void rope(const Tensor& positions, int rotary_dim, float theta, const YarnScale& yarn, Tensor& q,
          Tensor& k, cudaStream_t stream);

// YaRN extension single-tensor form.
void rope(const Tensor& positions, int rotary_dim, float theta, const YarnScale& yarn, Tensor& x,
          cudaStream_t stream);

// Host-side YaRN table construction (fp64 math, cast to fp32). Computes the per-pair inverse
// frequencies (rotary_dim / 2 fp32 values) and the fp32 attention magnitude for the extension
// factor `scale` over the original context length `original_max`. `theta` is the RoPE base;
// `beta_fast` / `beta_slow` are the correction-range rotation bounds (spec defaults 32 / 1) and
// `ext_factor` the extrapolation mix (spec default 1.0).
//
//   ramp(i)  = clamp((i - low) / (high - low), 0, 1)      i = pair index 0..R/2-1
//   mask(i)  = (1 - ramp(i)) * ext_factor
//   extrap[i] = theta ** (-2i / rotary_dim)
//   interp[i] = extrap[i] / scale
//   inv_freq[i] = interp[i] * (1 - mask[i]) + extrap[i] * mask[i]
//   low = floor(corr_dim(beta_fast)), high = ceil(corr_dim(beta_slow))  (clamped to [0, R-1])
//   corr_dim(n) = rotary_dim * ln(original_max / (n * 2*pi)) / (2 * ln theta)
//   mscale = 1.0 if scale <= 1 else 0.1 * ln(scale) + 1
//
// The base is the pure power law theta ** (-2i / rotary_dim) (matching the `__constant__`
// tables), so the table degenerates to the unextended power law and mscale == 1 at scale == 1.
YarnTable compute_rope_yarn_table(float theta, int rotary_dim, std::int64_t original_max,
                                  double scale, int beta_fast = 32, int beta_slow = 1,
                                  double ext_factor = 1.0);

} // namespace ninfer::ops
