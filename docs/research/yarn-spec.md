# YaRN spec research — reference, llama.cpp, transformers, vLLM, SGLang

Research for wayfinder map issue #6 (ticket #7). Findings feed the YaRN design ticket (#9).
All claims are grounded in the pinned sources listed in [Sources](#11-sources); file/line
pointers refer to those exact commits.

**Target geometry (NInfer Qwen3.6/3.8):** `rope_theta = 1e7`, `head_dim = 256`,
`rotary_dim = 64` (32 pairs, `partial_rotary_factor = 0.25`), interleaved M-RoPE with
`mrope_section = [11, 11, 10]`, `max_position_embeddings = 262144`
([qwen3_5-model.md:72-76](../maintainer/qwen3_5-model.md)). YaRN activation is the existing
`--max-context` switch; `factor = max_context / max_position_embeddings`, spec defaults for
everything else (standing decision, 2026-09-26).

---

## 1. Core formulas (common to every surveyed implementation)

Let `R = rotary_dim` (64 here), pairs `i = 0..R/2-1` (32 pairs), `s = factor > 1`,
`L = original_max_position_embeddings` (262144 here), `b = base` (1e7),
`β_fast`, `β_slow` (default 32, 1).

**Inverse frequencies per pair** (original = "extrapolation", NTK = "interpolation"):

```text
pos_freqs[i]             = b ** (2i / R)
inv_freq_extrap[i]       = b ** (-2i / R)                    # original RoPE frequency
inv_freq_interp[i]       = b ** (-2i / R) / s                # NTK-by-parts scaled frequency
```

**Correction range** (the per-pair crossover between the two, indexed by pair index):

```text
corr_dim(n_rot) = R · ln( L / (n_rot · 2π) ) / (2 · ln b)
low  = floor( corr_dim(β_fast) )     # clamped to [0, R-1]
high = ceil ( corr_dim(β_slow) )     # clamped to [0, R-1]
```

**Ramp + blend** (`i` = pair index 0..R/2-1; guard `high += 0.001` if `low == high`):

```text
ramp(i)  = clamp( (i - low) / (high - low), 0, 1 )
mask(i)  = (1 - ramp(i)) · ext_factor          # ext_factor default 1
inv_freq[i] = inv_freq_interp[i] · (1 - mask(i)) + inv_freq_extrap[i] · mask(i)
```

Interpretation: high-frequency pairs (`i < low`, many rotations in `L`) keep the **original**
frequency; low-frequency pairs (`i > high`, long wavelengths) get the **NTK-scaled** frequency
(`÷ s`); in between they blend linearly. This matches the paper's `γ(r)` with
`r = L · inv_freq[i]` (rotations within the original window) and the default
`α = β_slow = 1`, `β = β_fast = 32` for the Llama family (paper §3.2).

**Attention (magnitude) scale** — folded into cos/sin, never applied to logits:

```text
mscale = 1.0                          if s <= 1
mscale = 0.1 · ln(s) + 1.0            if s > 1          # paper Eq. 22: 1/√t
```

**Per-position table** (what every serving engine materializes, in some form):

```text
θ[p, i]      = p · inv_freq[i]         # p = position (0-based), p ≤ max_context
cos_cache[p,i] = cos(θ[p, i]) · mscale
sin_cache[p,i] = sin(θ[p, i]) · mscale
```

**Meaning of `factor`.** `s` is the ratio of extended to original context:
effective context = `L × s`. Confirmed by:

- paper §3.1, Eq. 11: `s = L'/L` with `L'` the extended length;
- vLLM cache length = `max_position_embeddings × scaling_factor` where
  `max_position_embeddings` is the **original** (`yarn_scaling_rope.py:79-84`,
  `__init__.py:240-279`);
- transformers validation: `implicit_factor = max_position_embeddings / original_max_position_embeddings`
  must equal the explicit `factor` (`modeling_rope_utils.py:968-979`);
- llama.cpp: `n_ctx_train = n_ctx_orig_yarn / rope_freq_scale` where
  `rope_freq_scale = 1/s` (`llama-context.cpp:3844-3846`, `common/arg.cpp:2341-2346`).

### Concrete numbers for the NInfer geometry

With `R = 64`, `b = 1e7`, `L = 262144`, `β_fast = 32`, `β_slow = 1`:

```text
corr_dim(32) = 64 · ln(262144/(32·2π)) / (2·ln 1e7) ≈ 14.24  →  low  = 14
corr_dim(1)  = 64 · ln(262144/(1·2π))  / (2·ln 1e7) ≈ 21.12  →  high = 22
```

- pairs `0..13`:  `mask = 1` → original frequency `b^(-2i/64)` (no scaling)
- pairs `14..21`: linear blend
- pairs `22..31`: `mask = 0` → NTK frequency `b^(-2i/64) / s`

`mscale` examples: `s=2 → 1.0693`, `s=4 → 1.1386`, `s=8 → 1.2079` (paper reports ≈1.208 for s=8).

---

## 2. Original — paper + reference repo

**Paper:** Peng, Quesnelle, Fan, Shippole, *"YaRN: Efficient Context Window Extension of
Large Language Models"*, ICLR 2024, openreview `wHBfxhZu1u`; preprint arXiv:2309.00071 (v2,
2023-11-01). The repo's `paper/yarn.pdf` is the v2 preprint.

Key content:

- §3.1 NTK-aware interpolation: change base to `b' = b · s^((|D|-2)/|D|)` (Eq. 16).
- §3.2 NTK-by-parts: per-component wavelength `λ_d = 2π b^((|D|-2d)/|D|)`,
  rotations `r(d) = L / λ_d` (Eq. 17), piecewise weight `γ(r) = 0` for `r < α`,
  `(r-α)/(β-α)` in between, `1` for `r > β` (Eqs. 18-20). "For the Llama family of models,
  good values for α and β are α = 1 and β = 32."
- §3.3 Dynamic scaling: (1) fixed `s = L'/L` for the whole inference cycle, or
  (2) **dynamic** `s = max(1, l'/L)` per forward pass ("Dynamic NTK"). The paper argues
  fixed `s` "may experience a performance discount at a length less than L and an abrupt
  degradation when the sequence length is longer than L'". **All serving engines surveyed
  implement variant (1)**; transformers exposes variant (2) separately as
  `rope_type: "dynamic"` (dynamic NTK), not as yarn.
- §3.4 Attention scaling: `softmax(q^T k / (√|D| · t))`; implement by scaling `q` and `k`
  each by `1/√t`, "achieved simply by scaling the complex RoPE embedding by the same
  amount". Recommended `1/√t = 0.1·ln(s) + 1` (Eq. 22). Also notes that with dynamic `s`
  and a KV cache, KV must be cached **before** RoPE application (not needed for fixed `s`).
- §4: results (see [Caveats](#9-caveats--best-practices)).

**Reference repo:** `jquesnelle/yarn` @ `995db5b` (master, 2024-04-17).
`scaled_rope/LlamaYaRNScaledRotaryEmbedding.py`:

```python
# lines 5-6
def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

# lines 9-14: low = floor(...low_rot...), high = ceil(...high_rot...), clamped [0, dim-1]

# lines 16-22
def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

# lines 24-27
def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0
```

- Constructor (line 30): `__init__(self, dim, max_position_embeddings=2048, base=10000,
  scale=1, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1,
  beta_fast=32, beta_slow=1, finetuned=False, device=None)`. Here `scale` is the extension
  factor `s`; `max_position_embeddings` is the *extended* (new) context length.
- `yarn()` (lines 74-83): `pos_freqs = base ** (arange(0, dim, 2).float() / dim)`;
  `inv_freq_extrapolation = 1/pos_freqs`; `inv_freq_interpolation = 1/(scale·pos_freqs)`;
  `low, high = find_correction_range(beta_fast, beta_slow, dim, base,
  original_max_position_embeddings)`; `inv_freq_mask = (1 - linear_ramp_mask(low, high,
  dim//2)) * extrapolation_factor`;
  `inv_freq = inv_freq_interp·(1-mask) + inv_freq_extrap·mask`;
  `self.mscale = get_mscale(scale) * attn_factor`.
- Table (lines 49-54): `t = arange(max_position_embeddings)`, `freqs = t[:,None] *
  inv_freq[None,:]`, then `register_buffer("cos_cached", emb.cos() * self.mscale)`,
  `sin_cached` likewise — **attention scale applied to the cos/sin table**.
- Defaults: `scale=1`, `extrapolation_factor=1`, `attn_factor=1`, `beta_fast=32`,
  `beta_slow=1`. The README only links the paper and publishes models (Llama 2 / Mistral /
  SOLAR fine-tuned at 32k/64k/128k); no extra formulas beyond the code.

Note the `dim` semantics: the repo takes the **full** rotary dimension (`dim`), which for
its targets equals head_dim (no partial rotary). Every modern engine generalizes this to
`rotary_dim` (see [Divergences](#8-divergence-table)).

---

## 3. llama.cpp

Pinned: `ggml-org/llama.cpp` @ `2145525a` (master, 2026-09-26).

### 3.1 Parameters and defaults

- `--rope-scale N` — "RoPE context scaling factor, expands context by a factor of N";
  stored inverted: `rope_freq_scale = 1/N` (`common/arg.cpp:2341-2346`).
- `--yarn-orig-ctx N` — original context size; default = model training context, from GGUF
  key `rope_scaling_orig_ctx_len`, else `n_ctx_train` (`src/llama-model.cpp:1406-1407`,
  `src/llama-context.cpp:138`).
- `--yarn-ext-factor` — "extrapolation mix factor (default: 1.00, 0.0 = full interpolation)"
  (`common/arg.cpp:2369-2374`); negative = unset → `1.0` when scaling type is YARN
  (`src/llama-context.cpp:173-174`).
- `--yarn-attn-factor` — "scale sqrt(t) or attention magnitude" (`common/arg.cpp:2376-2381`);
  `--yarn-beta-fast` / `--yarn-beta-slow` (defaults per the paper, 32/1, set upstream of the
  context init).
- The context init derives `factor = 1.0f / cparams.rope_freq_scale`
  (`src/llama-context.cpp:182`) — i.e. `s`, the extension factor.
- The model keeps **two** RoPE variants and picks per sequence length:
  `rope_long` (scaled) when `n_ctx_seq > n_ctx_orig_yarn`, else `rope_short`
  (`src/llama-model.cpp:2334, 2339`) — so scaling only engages beyond the original context,
  matching the "keep today's behavior unchanged below `max_position_embeddings`" decision.

### 3.2 Where the attention scale is applied — the cancel trick

llama.cpp splits the mscale between host and kernel:

1. Host (`src/llama-context.cpp:178-215`):

   ```cpp
   static auto get_mscale = [](float scale, float mscale) {
       return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
   };
   const float factor = 1.0f / cparams.rope_freq_scale;
   ...
   cparams.yarn_attn_factor = get_mscale(factor, 1.0f);          // = 0.1 ln(s) + 1
   // "when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor"
   cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));  // → 1.0 by default
   ...
   cparams.yarn_attn_factor *= hparams.rope_attn_factor;          // GGUF rope_attn_factor, default 1.0
   ```

   (`hparams.rope_attn_factor` default 1.0: `src/llama-hparams.h:149`.)

2. Kernel — the ggml `ggml_rope_ext` op (`ggml/src/ggml-cpu/ops.cpp:5958-5973`, identical
   device version `ggml/src/ggml-cuda/rope.cu:22-44`):

   ```cpp
   float theta_interp = freq_scale * theta_extrap;      // NTK angle (freq_scale = 1/s)
   float theta = theta_interp;
   if (ext_factor != 0.0f) {
       float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
       theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
       mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale); // ← the "cancelled" factor re-added here
   }
   *cos_theta = cosf(theta) * mscale;
   *sin_theta = sinf(theta) * mscale;
   ```

**Net effect:** cos/sin (hence q and k individually) are scaled by
`(0.1·ln(s) + 1) · hparams.rope_attn_factor` — attention logits effectively scaled by its
square, exactly the paper's `q, k ← (1/√t)·q, k` with `1/√t = 0.1·ln(s)+1`. The attention
scale is **not** applied to the attention logits anywhere; it lives in the RoPE rotation.

### 3.3 Inverse-frequency / angle construction

- `ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, n_rot, base) =
  n_dims · logf(n_ctx_orig / (n_rot·2π)) / (2·logf(base))` (`ggml/src/ggml.c:4469-4471`);
  `ggml_rope_yarn_corr_dims` floors/ceils and clamps to `[0, n_dims-1]`
  (`ggml/src/ggml.c:4473-4481`). `n_ctx_orig` is `n_ctx_orig_yarn`; `n_dims` is the rotated
  dimension (our `rotary_dim`, 64).
- **No precomputed inv_freq table and no precomputed cos/sin table**: the kernel computes,
  per (position, pair), the angle and trig **at runtime**:
  - CPU (`ops.cpp:5975-5989`): `theta` starts at the position `p` and is stepped per pair by
    multiplying `theta_scale = powf(freq_base, -2.0f/n_dims)` (line 6143) — cumulative
    floating-point multiplication.
  - CUDA (`rope.cu:107-114`): `theta_base = pos[i2] * powf(theta_scale, iw/2.0f)` — one
    `powf` per pair, all fp32, `cosf`/`sinf` fp32.
- Ramp (`ops.cpp:5951-5954`): `y = (i0/2 - low) / MAX(0.001f, high - low)`,
  `return 1 - MIN(1, MAX(0, y))` — same clamp semantics as the reference, `i0/2` = pair index.
- Non-integer factors: fully supported (all `float`). Effective context bound: no table —
  unbounded, limited only by memory (`--n_ctx`).

### 3.4 M-RoPE interaction (ggml)

`ggml_mrope_cache_init` (`ops.cpp:5991-6059`): the same `rope_yarn` per-pair correction is
applied with the **flat pair index** `i0` over the full `n_dims`, while the M-RoPE section
only selects which position id (t/h/w/e) the pair rotates by. Interleaved mode
(`GGML_ROPE_TYPE_IMROPE`, "qwen3vl apply interleaved mrope", line 6027-6036):

```cpp
if (sector % 3 == 1 && sector < 3 * sections[1])      theta = theta_h;
else if (sector % 3 == 2 && sector < 3 * sections[2]) theta = theta_w;
else if (sector % 3 == 0 && sector < 3 * sections[0]) theta = theta_t;
else                                                  theta = theta_e;
```

where `sector = (i0/2) % sum(sections)`. For `[11,11,10]` (sum 32) this assigns
`pair k → axis k%3` (t/h/w) with all 32 pairs covered — identical to transformers/vLLM
(§4, §5) and to NInfer's existing axis table
([config.cpp:90-91](../../src/models/qwen3_5/config.cpp)).

---

## 4. HuggingFace transformers

Pinned: `huggingface/transformers` @ `27166ea` (master, 2026-09-25).

**Naming note:** the standalone `apply_yarn_rotary_pos_emb` from transformers 4.x is **not
present in current master**. The v5 layout: YaRN *parameter computation* lives in
`src/transformers/modeling_rope_utils.py::_compute_yarn_parameters` (lines 345-484); the
attention factor is then applied **to the computed cos/sin inside each model's own rotary
module**. For Qwen3.5 (`src/transformers/models/qwen3_5/modeling_qwen3_5.py`):
`Qwen3_5TextRotaryEmbedding.__init__` dispatches on
`rope_parameters["rope_type"]` — `"yarn"` selects `_compute_yarn_parameters`
(lines 153-156) — and `forward` applies
`cos = freqs.cos() * self.attention_scaling; sin = freqs.sin() * self.attention_scaling`
(lines 197-198). The model's own `apply_rotary_pos_emb` (line 674, adapted from GLM) then
multiplies q/k by cos/sin. Semantics are the same as 4.x: **attention factor on cos/sin**.

### 4.1 `_compute_yarn_parameters` (modeling_rope_utils.py:345-484)

```python
dim = int(head_dim * partial_rotary_factor)          # lines 414-415  → our 64
factor = rope_parameters_dict["factor"]              # line 417
if factor is None:
    factor = config.max_position_embeddings / original_max_position_embeddings  # 423-424

def get_mscale(scale, mscale=1):                     # lines 426-429
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

if attention_factor is None:                         # lines 435-439
    if mscale and mscale_all_dim:
        attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
    else:
        attention_factor = get_mscale(factor)        # ← default: 0.1·ln(factor)+1

beta_fast = rope_parameters_dict.get("beta_fast") or 32    # line 442
beta_slow = rope_parameters_dict.get("beta_slow") or 1     # line 443

pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)  # 473
inv_freq_extrapolation = 1.0 / pos_freqs                            # 474
inv_freq_interpolation = 1.0 / (factor * pos_freqs)                 # 475
truncate = config.rope_parameters.get("truncate", True)             # 477
low, high = find_correction_range(beta_fast, beta_slow, dim, base,
                                  original_max_position_embeddings, truncate)   # 478
inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2)     # 481
inv_freq = (inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor)           # 482-484
return inv_freq, attention_factor
```

- `find_correction_dim`/`find_correction_range`/`linear_ramp_factor` (lines 447-469) are the
  reference formulas verbatim (incl. the `max += 0.001` singularity guard).
- The correction range is computed with `dim` (= `rotary_dim`, **not** head_dim) and
  `original_max_position_embeddings`.
- `extrapolation_factor` is **not a config key** — hard-coded 1.
- `original_max_position_embeddings` is a **required** config key (line 421, `KeyError` if
  missing); `factor` is required for validation (§4.2) but may be `None` at compute time
  (DeepSeek-style, derived from `max_position_embeddings / original_max_position_embeddings`).

### 4.2 Validation (`_validate_yarn_rope_parameters`, lines 931-985)

- Required keys: `rope_type`, `factor`, `original_max_position_embeddings`.
- Optional: `rope_theta`, `attention_factor`, `beta_fast`, `beta_slow`, `mscale`,
  `mscale_all_dim`, `truncate`.
- `factor` must be numeric and `>= 1.0` (warning otherwise, lines 941-943).
- `attention_factor`, if given, must be `> 0`; `beta_fast`/`beta_slow` numeric;
  `beta_fast or 32 >= beta_slow or 1` (lines 945-966).
- Cross-check (lines 968-979): `implicit_factor = max_position_embeddings /
  original_max_position_embeddings`; if it differs from the explicit `factor` (and is not 1),
  warns and **uses the explicit factor**. Confirms `factor = post-yarn / pre-yarn context`.

### 4.3 Runtime behavior for Qwen3.5 text

- `dynamic_rope_update` (lines 34-132) recomputes `inv_freq` **only** for rope types
  containing `"dynamic"` or `"longrope"` (wrapper, lines 118-130). **YaRN never recomputes at
  runtime** — the table is fixed at init, exactly the fixed-`s` variant (paper §3.3 option 1).
- M-RoPE forward (modeling_qwen3_5.py, `Qwen3_5TextRotaryEmbedding.forward`):
  `position_ids` has 3 rows (t, h, w) of shape `(3, bs, T)`;
  `inv_freq` is expanded to `(3, T, dim/2, 1)` and multiplied against each row;
  `cos = freqs.cos() * attention_scaling`, same for sin; then
  `recomposition_frequencies` (interleaved layout):

  ```python
  freqs_thw = freq[0]                      # start from the T row
  for dim, offset in enumerate((1, 2)):    # H, W
      length = self.mrope_section[dim] * 3
      idx = slice(offset, length, 3)
      freqs_thw[..., idx] = freq[dim, ..., idx]
  return torch.cat((freqs_thw, freqs_thw), dim=-1)
  ```

  For `mrope_section [11, 11, 10]`: pair `k` uses the **t** position for `k%3 == 0`
  (`k < 33`), **h** for `k%3 == 1` (`k < 33`), **w** for `k%3 == 2` (`k < 30`). This is the
  same mapping as llama.cpp's `imrope` (§3.4) and vLLM's `apply_interleaved_rope` (§5.2).
  For pure text all three rows are equal, so the layout degenerates to single-position RoPE.
  The `torch.cat((freqs, freqs))` duplicates the 32-pair cos/sin to a 64-entry (2·R) table
  matching Qwen's GPT-J pair head layout (pair `i` at head dims `2i, 2i+32` — NInfer uses the
  NeoX split-half equivalent over the same 32 pairs, [rope.h:10-14](../../include/ninfer/ops/rope.h)).
- `Qwen3_5TextConfig` defaults (`configuration_qwen3_5.py`): `head_dim = 256`,
  `max_position_embeddings = 32768` (artifact config carries 262144),
  `partial_rotary_factor = 0.25` (BC default in `__post_init__`, line 111);
  `mrope_section`/`mrope_interleaved` are model-level keys, exempt from rope validation
  (`ignore_keys_at_rope_validation`, line 108); `mrope_section` default `[11, 11, 10]`
  (modeling_qwen3_5.py line 160).
- No cos/sin cache (inv_freq only): `inv_freq` (32 floats) is kept and angles computed per forward pass in
  fp32 under disabled autocast.

---

## 5. vLLM

Pinned: `vllm-project/vllm` @ `7d8c5fe9` (main HEAD, 2026-09-26).
`vllm/model_executor/layers/rotary_embedding/`.

### 5.1 Text YaRN (`yarn_scaling_rope.py`, `common.py`)

- Helpers (`common.py`): `yarn_find_correction_dim` (34-42), `yarn_find_correction_range`
  (46-59, `truncate: bool = True` param), `yarn_linear_ramp_mask` (62-70),
  `yarn_get_mscale(scale=1, mscale=1) = 0.1·mscale·ln(scale) + 1` (73-76) — same as the
  reference with the DeepSeek `mscale` scalar.
- Class (lines 11-84):
  - `self.mscale` resolution (44-53): explicit `attention_factor` wins; else
    `yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim)`
    when both given; else `yarn_get_mscale(scaling_factor)` — i.e. default `0.1·ln(s)+1`.
    (There is **no** `extrapolation_factor` parameter — hard-coded 1.)
  - `_compute_inv_freq` (55-76): `pos_freqs = base ** (arange(0, rotary_dim, 2).float() /
    rotary_dim)`; mask `= 1 - yarn_linear_ramp_mask(low, high, rotary_dim // 2)` —
    **without** an `extrapolation_factor` multiplier; correction range over
    `self.rotary_dim` and `self.max_position_embeddings`.
  - `_compute_cos_sin_cache` (78-84): `t = arange(self.max_position_embeddings *
    self.scaling_factor)` (fp32); `freqs = einsum(t, inv_freq)`;
    `cos = freqs.cos() * self.mscale`, `sin = freqs.sin() * self.mscale` —
    **attention scale on the cos/sin table**.

### 5.2 Config dispatch (`__init__.py::get_rope`, lines 240-279)

For `rope_type == "yarn"`:

- `scaling_factor = rope_parameters["factor"]`, `original_max_position =
  rope_parameters["original_max_position_embeddings"]` (required keys).
- The rope module is constructed with `max_position_embeddings = original_max_position` —
  i.e. the **original** context, and the table length is
  `original_max_position_embeddings × factor` = effective context.
- `extra_kwargs` passes `beta_fast`, `beta_slow`, `mscale`, `mscale_all_dim`,
  `attention_factor`, `truncate` (HF-derived keys).
- **If `mrope_section` is present**, dispatch goes to `MRotaryEmbedding(...,
  scaling_factor=...)` instead — and there `cache_max_position_num =
  max_position_embeddings * 4` (mrope.py:347-348, a Qwen2.5-VL video-cache workaround).
  Consequence: the M-RoPE + YaRN path computes the correction range with
  `max_position_embeddings = 4 × original` (mrope.py:364-367 reuses
  `YaRNScalingRotaryEmbedding._compute_inv_freq(self, ...)`). **Text-only YaRN must not copy
  this 4×**; it only affects the crossover pair bounds.

### 5.3 M-RoPE interleaved layout (`mrope.py`)

`apply_interleaved_rope` (lines 236-250):

```python
channels = torch.arange(x.shape[-1])
is_height = (channels % 3 == 1) & (channels < mrope_section[1] * 3)
is_width  = (channels % 3 == 2) & (channels < mrope_section[2] * 3)
result = torch.where(is_height, x[1], x[0])     # x[0]=T, x[1]=H, x[2]=W
return torch.where(is_width, x[2], result)
```

Identical pair→axis assignment to transformers' `recomposition_frequencies` and ggml
imrope. Non-interleaved (chunked) fallback: per-section frequency selection
`cat([m[i] for i, m in enumerate(cos.split(mrope_section))])` (forward, lines ~440-462).

**Effective context bound:** table length `original_max × factor` (text) — hard bound.
Non-integer factors: supported (float factor; `arange` with a float end).

---

## 6. SGLang

Pinned: `sgl-project/sglang` @ `cbdea5dc` (main HEAD, 2026-09-26).
`python/sglang/srt/layers/rotary_embedding/`.

- `yarn.py` (180 lines): helpers at lines 14-63 — `yarn_find_correction_dim`,
  `yarn_find_correction_range` (with `truncate: bool = True`), `yarn_linear_ramp_mask`,
  `yarn_get_mscale_simple(scale) = 0.1·ln(s)+1` (56-57),
  `yarn_get_mscale(scale, mscale=1) = 0.1·mscale·ln(s)+1` (60-63).
- `YaRNScalingRotaryEmbedding` (lines 91-180):
  - defaults (107-113): `extrapolation_factor=1`, `attn_factor=1`, `beta_fast=32`,
    `beta_slow=1`, `truncate=True`, `mscale=None`, `mscale_all_dim=None`.
  - mscale (122-131): both `mscale` and `mscale_all_dim` given →
    `yarn_get_mscale(sf, mscale) / yarn_get_mscale(sf, mscale_all_dim)` — comment:
    "Match Hugging Face's YaRN RoPE scaling"; else `yarn_get_mscale_simple(sf)`;
    then `self.mscale *= attn_factor`.
  - `_compute_inv_freq` (137-161): same core; the mask **does** multiply by
    `self.extrapolation_factor` (lines 153-156); correction range over `rotary_dim` and
    `max_position_embeddings` (the original, per factory below).
  - `_compute_cos_sin_cache` (171-180): `t = arange(max_position_embeddings *
    scaling_factor)` (fp32); `cos/sin = trig * self.mscale`.
  - **Dynamic cache extension** (`_extend_yarn_cache`, lines 66-88;
    `_ensure_cos_sin_cache_length`, 163-169): if a request needs positions beyond the
    cached table, new rows are appended, recomputed with the **same** YaRN `inv_freq`
    (same scaling factor and correction-range bound) — SGLang can grow the table past
    `original × factor` at runtime; vLLM/transformers cannot.
- `factory.py` (lines 311-369): for `scaling_type == "yarn"` —
  `scaling_factor = rope_scaling.get("factor", 1.0)`;
  `original_max_position = rope_scaling.get("original_max_position_embeddings", max_position)`
  (falls back to the model's `max_position_embeddings` when the key is missing); M-RoPE
  models go to `YaRNScalingMRotaryEmbedding` constructed with `original_max_position`
  (no 4× cache blow-up — unlike vLLM).
- Extra SGLang keys: `extrapolation_factor`, `attn_factor` (lines 321-331) — HF config
  convention is `attention_factor`; SGLang accepts its own spelling plus the HF one via the
  mscale path.

**Effective context bound:** `original × factor` initially, extendable at runtime.

---

## 7. Defaults table

| Parameter | Paper default | jquesnelle repo | llama.cpp | transformers | vLLM | SGLang |
|---|---|---|---|---|---|---|
| `factor` / `s` | per model (`s = L'/L`) | `scale=1` | `--rope-scale N` (stored `1/N`) | `rope_parameters.factor` (or `max/original`) | `rope_scaling.factor` | `rope_scaling.factor` (1.0) |
| `original_max_position_embeddings` | `L` | ctor arg | GGUF `rope_scaling_orig_ctx_len` or `n_ctx_train` | **required** | **required** | model `max_position_embeddings` fallback |
| `beta_fast` | 32 (Llama) | 32 | 32 (CLI) | 32 (`or 32`) | 32 | 32 |
| `beta_slow` | 1 (Llama) | 1 | 1 (CLI) | 1 (`or 1`) | 1 | 1 |
| `extrapolation_factor` | 1 | 1 | 1.0 (`0.0` = pure interpolation) | hard-coded 1 | hard-coded 1 (no param) | 1 |
| `attention_factor` / mscale | `0.1·ln s + 1` | `get_mscale(s)·attn_factor` | `0.1·ln s + 1` (kernel) × GGUF `rope_attn_factor` | `attention_factor` → `get_mscale(s)` | `attention_factor` → `yarn_get_mscale(s)` | `attn_factor` → `0.1·ln s + 1` |
| `mscale` / `mscale_all_dim` | — (n/a) | — | GGUF `rope_yarn_log_mul` (DeepSeek) | ratio `get_mscale(s,ms)/get_mscale(s,msad)` | ratio | ratio ("match Hugging Face") |
| `truncate` | — (always) | always | always | `True` (config key) | `True` | `True` |
| `base` | 10000 / model | 10000 | GGUF `rope_freq_base_train` (10000) | `rope_theta` (1e7 for Qwen3.5) | `rope_theta` | `rope_theta` |
| Attention scale applied to | RoPE embedding (cos/sin) | cos/sin table | cos/sin (kernel) | cos/sin (`attention_scaling`) | cos/sin table | cos/sin table |
| Table / bound | `max_position_embeddings` | `max_position_embeddings` | unbounded (runtime angles) | unbounded (`inv_freq` only) | `original × factor` (text), `4×original` (mrope) | `original × factor`, runtime-extendable |

## 8. Divergence table

Material differences across implementations (all implement the §1 core):

| Aspect | jquesnelle | llama.cpp | transformers (27166ea) | vLLM (7d8c5fe) | SGLang (cbdea5dc) |
|---|---|---|---|---|---|
| Correction-range `dim` arg | full `dim` (head) | `n_dims` (rotated) | `head_dim·partial_rotary_factor` | `rotary_dim` | `rotary_dim` |
| Correction-range `max_pos` arg | `original_max` | `n_ctx_orig_yarn` | `original_max` | **original** (text) / **4×original** (mrope+yarn) | original (text and mrope) |
| `extrapolation_factor < 1` | yes (ctor) | yes (`--yarn-ext-factor`) | no (hard 1) | no (no param) | yes (ctor/factory) |
| Attention factor override | `attn_factor` (multiplied) | `--yarn-attn-factor` + GGUF `rope_attn_factor` | `attention_factor` (wins over mscale) | `attention_factor` (wins over mscale) | `attn_factor` (multiplied) |
| Non-integer factor | yes | yes | yes (`>= 1.0`) | yes | yes |
| Table strategy | static cos/sin table | per-token runtime angles (CPU cumulative `powf` chain; CUDA per-pair `powf`) | `inv_freq` only, per-forward fp32 angles | static cos/sin table (fp32 → cast) | static table + dynamic extension |
| Runtime recompute | no | no | no (yarn not dynamic) | no | yes (cache growth only, same `inv_freq`) |
| Scaling engages only beyond original ctx | n/a | yes (`rope_long` iff `n_ctx_seq > n_ctx_orig_yarn`) | n/a (config-declared) | n/a (config-declared) | n/a (config-declared) |
| M-RoPE + YaRN ramp index | n/a | flat pair index over full `n_dims` | flat pair index (recomposition) | flat pair index | flat pair index |

The **only** semantic divergence that can change results is the vLLM M-RoPE+YaRN 4×
correction-range `max_pos` (shifts the low/high crossover pairs; irrelevant to text-only
Qwen3.5). Everything else is parameter plumbing.

---

## 9. Caveats / best practices

1. **Partial RoPE (rotary_dim 64 of 256).** Every engine computes the YaRN table over the
   `rotary_dim` pairs only: `pos_freqs = base^(2i/R)` with `R = 64`, ramp over 32 pairs,
   correction range clamped to `[0, R-1]`. The non-rotated `head_dim - R = 192` dims are
   untouched (paper's `|D|` is the rotated dim; NInfer's existing op already isolates
   `[0, rotary_dim)`, [rope.h:32-33](../../include/ninfer/ops/rope.h)).
2. **M-RoPE interleaving is orthogonal to YaRN.** YaRN only replaces the 32-entry `inv_freq`
   vector; the pair→axis (t/h/w) interleaved assignment
   (`pair k → axis k%3`, bounded by `3·mrope_section[axis]`) is unchanged, and is verified
   identical across transformers (`recomposition_frequencies`), vLLM
   (`apply_interleaved_rope`), ggml imrope, and NInfer's axis table (config.cpp:90-91). For
   pure text all three position rows are equal → one table; the YaRN table must be the same
   for every axis. **Do not copy vLLM's 4× mrope `max_pos` quirk.**
3. **Fixed factor vs the paper's dynamic NTK.** Paper §3.3: fixed `s` costs some
   short-context performance and degrades abruptly past the target; dynamic
   `s = max(1, l'/L)` avoids both but requires caching KV **before** RoPE. The ecosystem
   standard (all four engines) is fixed `s` from the target context — matching NInfer's
   startup-fixed `--max-context` decision.
4. **Factor range.** Paper validated `s = 2, 16, 32` (fine-tuned; §4.1-4.2); Code Llama
   reference point `s ≈ 88.6` (355k context) shows large-`s` extrapolation can work but is
   unvalidated territory. `llama.cpp` documents `--yarn-ext-factor` with `0.0 = full
   interpolation` as the conservative end. For NInfer, `s = max_context/262144` lands at
   1.5–4× for realistic 400k–1M contexts — near the well-behaved `s=2` regime, but see the
   zero-shot caveat below.
5. **Zero-shot (inference-only) extension is the paper's least-supported regime.** All
   paper results used ~400M-token fine-tuning (≈0.1% of pretraining). Inference-only YaRN at
   `s≈2` on a model trained at 262144 is extrapolation from the `s=2` fine-tuned evidence,
   not a measured result — the bounded e2e run on the 5090 (map decision) is the check.
6. **Numerical pitfalls.**
   - *Angle magnitude / trig precision.* `θ = p·inv_freq[0] = p` for the fastest pair; at
     `p ≈ 1e6`, fp32 argument quantization is ≈0.03 rad. Every surveyed engine uses fp32
     angles (host torch tables in vLLM/SGLang, in-kernel `cosf`/`sinf` in llama.cpp CUDA,
     fp32 matmul in transformers) — none uses fp64 in the hot path — but fast-math CUDA
     `sinf/cosf` (`--use_fast_math`) is **not** adequate for large arguments; NInfer's op
     contract already requires a naive FP64 oracle ([rope.h:30-32](../../include/ninfer/ops/rope.h)),
     which is the right qualification path for the YaRN variant.
   - *Table construction variants.* Reference/transformers/vLLM/SGLang: `base^(2i/R)` power
     + reciprocal (fp32); llama.cpp CPU: cumulative multiplication by
     `base^(-2/R)` per pair (rounding accumulates); llama.cpp CUDA: one `powf` per pair.
     Differences are last-ulp-level; keep the fp64 oracle as the reference, not any of
     these.
   - *Ramp guard.* `high += 0.001` (or `max(0.001, high-low)` in ggml) when the range
     collapses; clamp both bounds to `[0, R-1]`.
   - *Positions.* 0-based `I32`; `p ≤ max_context ≤ ~1e6 « 2^24` so fp32 position casts are
     exact.
   - *`mscale`.* Single fp32 multiply of cos/sin; no issue at any realistic `s`.
7. **Expected quality degradation (paper evidence, fine-tuned models).**
   - In-range (≤ target): negligible — Table 1 (Llama-2 7B 4k→8k, 128k Proof-pile sliding
     PPL): at 8192, PI 3.34 / NTK 3.59 / **YaRN 3.35**; Table 3 benchmarks at `s=16/32`
     within ~1-2 pts of the 4k baseline (MMLU 42.5/41.7 vs 43.8; HellaSwag 78.8/78.4 vs
     77.8 — YaRN actually *beats* baseline on HellaSwag).
   - Out-of-range: degrades but most gracefully — Table 1 at 10240 (2.5× target): PI 8.07 /
     NTK 6.24 / **YaRN 6.04**; Table 2 at 131072: Together-PI >104 / Code Llama-NTK 2.54 /
     **YaRN s=32 2.37**.
   - Training cost: 400 steps (400M tokens) for `s=16`, +200 steps for `s=32` — 10× less
     than PI (Rozière et al.), 2.5× less than NTK (Chen et al.).
8. **Effective context bound.** Define it as `max_context` (the requested `--max-context`);
   the table/positions must cover it. vLLM's hard table bound is `original × factor`;
   transformers/llama.cpp have no bound; SGLang extends on demand. Non-integer `factor`
   (e.g. `500000/262144 ≈ 1.9074`) is fine everywhere — in NInfer `factor` is an exact
   rational of two integers, and `original × factor == max_context` by construction.
9. **Degenerate path.** `s ≤ 1` must reproduce today's behavior exactly (standing decision:
   `--max-context <= max_position_embeddings` keeps current behavior). All formulas
   degenerate: `mscale = 1`, NTK branch `inv_freq/s = inv_freq` at `s=1`; the cleanest
   implementation dispatches to the YaRN path only when `s > 1`.
10. **Scope note (NInfer).** MTP/draft components inherit the Text RoPE geometry
    ([qwen3_5-model.md:46-47](../maintainer/qwen3_5-model.md)); the `ops::rope` op also
    serves the DFlash full-head domain (`head_dim=rotary_dim=128`, rope.h:19) — the YaRN
    design should state which rope call sites consume the YaRN table (text full-attention
    layers are the in-scope set per the map).

---

## 10. Open questions for the design ticket (#9)

1. **Op shape.** Today `ops::rope` takes a scalar `theta` and is stateless, computing angles
   in-kernel ([rope.h:36-41](../../include/ninfer/ops/rope.h)). YaRN needs either
   (a) an `inv_freq` table input (32 fp32 values, built once at startup) + `mscale`, angles
   still computed in-kernel (llama.cpp-CUDA style; keeps the op stateless and CUDA-Graph
   friendly; cost: 32 muls + sin/cos per token per head-row), or (b) a precomputed
   cos/sin table `[max_context × 32]` (vLLM/transformers style; `1e6 × 32 × 2 × 4B ≈ 256MB`
   persistent device memory at 1M context, conflicting with the op's "no persistent state"
   contract). Recommendation to decide: (a) — matches the existing contract and the graph
   capture model.
2. **Table construction.** Compute `inv_freq` on the host at startup (fp64 for the
   `base^(2i/64)` powers, then fp32 device buffer) and qualify against the naive FP64
   oracle per op-development rules. Confirm the `low=14 / high=22` crossover for the pinned
   geometry (base 1e7, R 64, L 262144, β 32/1) as a test vector.
3. **Attention factor path.** Apply `mscale = 0.1·ln(s)+1` to the per-pair cos/sin in the
   kernel (all four engines agree; logits get `mscale²` implicitly). Verify NInfer's
   attention op adds no separate scaling, and that `--max-context <= 262144` (s ≤ 1) routes
   to today's op unchanged.
4. **Dispatch semantics.** YaRN engages iff `max_context > max_position_embeddings` with
   `factor = max_context / max_position_embeddings` (standing decision). Should `factor`
   read back as the exact rational (e.g. 1.9074... for 500000), or is a rounded factor
   acceptable? (All references take a float; table length is the integer `max_context`.)
5. **M-RoPE axis handling.** Confirm the implementation order: YaRN `inv_freq` per pair
   first, then per-axis position selection (axis only chooses *which position* multiplies
   the pair; same table for t/h/w). Pure-text path can collapse to single-position
   angles, but the op should keep the 3-row path to stay identical for future multimodal.
6. **Precision target.** In-kernel fp32 sin/cos at `p` up to 1e6 — which CUDA math mode?
   Default (non-fast-math) `sinf/cosf` vs an explicit argument reduction; the FP64 oracle
   comparison will decide, but the design should pick the kernel mode up front.
7. **Draft/MTP coverage.** Do draft/MTP rope call sites (DFlash geometry, MTP inherited
   geometry) get the YaRN table in v1, or only the main-text full-attention layers? (Map
   scope is the text path; state it explicitly in the design.)

---

## 11. Sources

| Source | Pinned revision | Date |
|---|---|---|
| jquesnelle/yarn | `995db5b575e75230b3384d658f8b944c9662f775` (master) | 2024-04-17 |
| ggml-org/llama.cpp | `2145525a4081d66ff1a87cf43ef809f95a85ac0c` (master) | 2026-09-26 |
| huggingface/transformers | `27166ea03f12c940f23176a904ab1d2ff1a3dcbb` (master) | 2026-09-25 |
| vllm-project/vllm | `7d8c5fe9a95dd7a9734f05fdbad57c1ad61f77fe` (main HEAD) | 2026-09-26 |
| sgl-project/sglang | `cbdea5dccf0a8085200abad617910c9688e932fe` (main HEAD) | 2026-09-26 |
| YaRN paper | arXiv:2309.00071v2 (repo `paper/yarn.pdf`); ICLR 2024 openreview `wHBfxhZu1u` | 2023-11-01 |
| NInfer repo | master @ `404469d4` (worktree) | 2026-09-26 |

File pointers (per implementation):

- **jquesnelle/yarn** — `scaled_rope/LlamaYaRNScaledRotaryEmbedding.py`: helpers 5-27,
  ctor 29-46, table 46-54 (incl. `emb = torch.cat((freqs, freqs))` GPT-J layout, line 50),
  `forward` 56-72, `yarn()` 74-83; `README.md` (paper link, published models).
- **llama.cpp** — `common/arg.cpp:2341-2381` (CLI flags); `src/llama-model.cpp:1406-1407`
  (orig ctx), `1419-1425` (freq scale train), `2322-2344` (rope_short/rope_long);
  `src/llama-context.cpp:135-138, 173-215, 3844-3846` (init, mscale, cancel, n_ctx);
  `src/llama-hparams.h:149` (`rope_attn_factor`); `ggml/src/ggml.c:4469-4481` (corr
  dims); `ggml/src/ggml-cpu/ops.cpp:5949-6059` (ramp, rope_yarn, cache init, mrope);
  `ggml/src/ggml-cuda/rope.cu:15-44, 107-114` (CUDA).
- **transformers** — `src/transformers/modeling_rope_utils.py`: `dynamic_rope_update`
  34-132, `_compute_yarn_parameters` 345-484, `ROPE_INIT_FUNCTIONS` 666-673,
  `RopeParameters` 678-733, `_validate_yarn_rope_parameters` 931-985;
  `src/transformers/models/qwen3_5/configuration_qwen3_5.py:89, 108, 111`;
  `src/transformers/models/qwen3_5/modeling_qwen3_5.py` (`Qwen3_5TextRotaryEmbedding`
  143-213 incl. mrope forward 189-198, `recomposition_frequencies` 204-212,
  `apply_rotary_pos_emb` 674);
  `src/transformers/models/qwen3_vl/modular_qwen3_vl.py:285-301` (Qwen3-VL interleaved
  recomposition, default sections [24,20,20]).
- **vLLM** — `vllm/model_executor/layers/rotary_embedding/common.py:34-76` (helpers);
  `yarn_scaling_rope.py:11-84`; `__init__.py:240-279` (dispatch); `mrope.py:236-250,
  305-367` (interleaved apply, scaling, 4× cache).
- **SGLang** — `python/sglang/srt/layers/rotary_embedding/yarn.py:14-180` (helpers, class,
  dynamic extension); `factory.py:311-369` (dispatch, original-max default).
- **NInfer** — `include/ninfer/ops/rope.h:1-43`; `src/models/qwen3_5/config.cpp:66-96`;
  `docs/maintainer/qwen3_5-model.md:55-76`.

*Method note: llama.cpp and jquesnelle/yarn were shallow-cloned; transformers fetched via
blobless clone at the pinned SHA; vLLM/SGLang files fetched as raw content at the pinned
HEAD SHAs (verified with `git ls-remote`). Line numbers are from those exact revisions.*
