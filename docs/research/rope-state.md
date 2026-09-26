# RoPE state in the engine today — call-site and capture audit

Research ticket for the YaRN wayfinder map (issue #6; audit ticket #8). All pointers are
against `master` @ `404469d4` (2026-09-26). Read-only audit: no engine code modified.

## 1. TL;DR

- RoPE rotation is one Op (`ops::rope`, `include/ninfer/ops/rope.h`) plus two DFlash-only fused
  surfaces (`ops::rmsnorm_rope`, `context_kv_materialize`). All three hard-code θ=1e7 (and
  vision θ=1e4) **compile-time `__constant__` frequency tables**, and the fast kernels are
  dispatch-keyed on the literal `theta == 1.0e7F`.
- YaRN's per-pair frequency table is **not** a pure power law `θ^(-2i/R)`, so neither the fast
  kernels (constant tables) nor the generic kernel (`powf(theta, exponent)`) can express it.
  The dispatch predicate `theta == 1e7` is a silent-correctness hazard: a YaRN model keeps
  `rope_theta = 1e7` in its config, so dispatch would still select the *unextended* fixed
  kernels. The predicate must key on the table (or a YaRN flag), not on θ alone.
- Graph capture runs once in the `ProgramImpl` constructor (`prepare_graphs`,
  `program_impl.cpp:303–307` → `graphs.cpp:106`), before any user launch. Captured graphs bake
  kernel parameters (pointers and scalars). A runtime frequency table must therefore live at a
  **stable device address, uploaded before capture** — the `io.rope_pos` pattern
  (planned persistent tensor + `set_device_i32`) is the existing precedent.
- The `1/sqrt(head_dim)` attention scale is a by-value `float` kernel argument at eight
  attention call sites (host-computed, baked per captured graph). YaRN's `attention_factor`
  multiplies into those same expressions; no device state is required for it to be
  graph-safe.
- `max_context > text.max_position_embeddings` is rejected at startup
  (`startup.cpp:744–747`); `max_context > draft->max_position_embeddings` is rejected at
  `startup.cpp:740–743`. KV page geometry derives from `max_context` alone
  (`page_count`, `startup.cpp:67`), so relaxing the validation is the only capacity gate.
- The perplexity / offline causal-scoring route **shares the engine route and the RoPE code**:
  `Program::causal_score` (`program_impl.cpp:318`) drives `prefill_text_chunk`
  (`program/prefill.cpp:61`) into the same `TextContext::attn_mix` → `text_rope` → `ops::rope`
  path as generation.
- The converter rejects scaled RoPE (`tools/convert/qwen3_5.py:55–71`): `type`/`rope_type`
  must be `"default"`, and `factor != 1.0` raises "scaled RoPE is not implemented". The C++
  loader rejects unknown rope-config members (`config.cpp:65–100`).

## 2. Call-site inventory

RoPE consumers by path. "Table" = which compile-time constant the fixed kernel actually uses.

| # | Path | File / function | Line | Tensors rotated | θ source | Table used |
|---|------|-----------------|------|-----------------|----------|------------|
| 1 | Text full-attention layer (prefill + verify) | `src/models/qwen3_5/execution/text.cpp` `TextContext::attn_mix` | 879 (`text_rope`) | Q + K (D256/R64, 24/4 or 16/2 heads; axes 1 or 3) | `config_.rope_parameters->rope_theta` (1e7) | `kTextRopeInvFrequency[32]` (float) |
| 2 | MTP verify target Q/K | `text.cpp` `TextContext::mtp_forward_tail` | 357 (`text_rope` pair) | MTP Q + K (D256/R64) | `config_.rope_parameters->rope_theta` | `kTextRopeInvFrequency[32]` |
| 3 | MTP prefill chunk: draft K append | `text.cpp` `TextContext::mtp_prefill_chunk` | 490 (`text_rope` single-K) | MTP K (D256/R64) | `config_.rope_parameters->rope_theta` | `kTextRopeInvFrequency[32]` |
| 4 | MTP final chunk: draft Q | `text.cpp` `TextContext::mtp_prefill_chunk` | 535 (`text_rope` single-Q) | MTP Q (D256/R64) | `config_.rope_parameters->rope_theta` | `kTextRopeInvFrequency[32]` |
| 5 | DFlash context KV append (K only) | `src/models/qwen3_5/execution/draft.cpp` `dflash_context_append` | 211–213 (`ops::rope` single-K) | Draft K (D128/R128, 8 KV heads) | `config.rope_theta` (draft, 1e7) | `kDflashRopeInvFrequency[64]` (double) |
| 6 | DFlash proposal attention (Q + K) | `draft.cpp` `dflash_proposal` | 457–458 (`ops::rope` pair) | Draft Q + K (D128/R128, 32/8) | `config.rope_theta` (draft, 1e7) | `kDflashRopeInvFrequency[64]` (double) |
| 7 | DFlash2 proposal (fused rmsnorm+RoPE) | `draft.cpp` (DFlash2 branch loop) | 316 (`ops::rmsnorm_rope`) | Draft Q + K (D128/R128, 32/8) | **none — θ=1e7 hard-coded in the Op contract** | `kDflashRopeInvFrequency[64]` via `dflash_rope_sincos` |
| 8 | DFlash cyclic-cache KV materialization | `src/ops/context_kv_materialize/materialize.cu` `store_key_head` | 49–50 (`dflash_rope_sincos` direct) | Materialized K rows (D128) | **none — hard-coded in the helper** | `kDflashRopeInvFrequency[64]` |
| 9 | Vision self-attention | `src/models/qwen3_5/execution/vision.cpp` (per-layer attention) | 371–372 (`ops::rope`, literal `10'000.0F`) | Vision Q + K (D72/R72, 16/16) | literal θ=1e4 in the call | `kVisionRopeInvFrequency[18]` (float) |

Notes on the table:

- The MTP path reuses the *text* model's `rope_parameters` (`TextConfig::rope_parameters`,
  `config.h:86`), so MTP Q/K consume the same text table as the target layers — there is no
  separate MTP table. (MTP's own `rope_delta`/`target_rope_positions` are position plumbing
  only; see §5.)
- Rows 7–8 do not go through the `ops::rope` Op API at all: `rmsnorm_rope` has no θ parameter
  (`include/ninfer/ops/rmsnorm_rope.h:31–32`), and `materialize.cu` calls the
  `__device__` helper directly. Both silently assume θ=1e7.
- The `mtp_round` Op (`include/ninfer/ops/mtp_round.h:21,26`) only computes
  `ar_rope_positions = ar_positions + rope_deltas` (I32 arithmetic, no rotation) — not a RoPE
  consumer, but it is where MTP autoregressive positions are formed.

### Constant tables

| Constant | File | Type | Size | θ | Expressed as |
|----------|------|------|------|---|--------------|
| `kTextRopeInvFrequency` | `src/ops/kernel/rope.cuh:26` | `__device__ __constant__ float` | 32 | 1e7 | `θ^(-2i/64)`, i=0..31 (R=64) |
| `kVisionRopeInvFrequency` | `src/ops/kernel/rope.cuh:36` | `__device__ __constant__ float` | 18 | 1e4 | `θ^(-2(i%18)/36)`, i=0..17 (R=72) |
| `kDflashRopeInvFrequency` | `src/ops/common/dflash_rope.cuh:11` | `__device__ __constant__ double` | 64 | 1e7 | `θ^(-2i/128)`, i=0..63 (R=128), double precision + explicit range reduction in `dflash_rope_sincos` (`dflash_rope.cuh:36–44`) |

Consumers of each table:

- `kTextRopeInvFrequency`: `fixed_axis_frequency<Text1D|TextMrope>` (`rope.cuh:44–52`) →
  `rope_fixed_kernel` / `rope_fixed_split_kernel` in Text1D/TextMrope modes; generic fallback
  does **not** use it (it recomputes `powf(theta, -2i/R)`).
- `kVisionRopeInvFrequency`: `fixed_axis_frequency<Vision2D>` (`rope.cuh:45–47`).
- `kDflashRopeInvFrequency`: `dflash_rope_sincos` — used by `fixed_sincos<DflashText1D>`
  (`rope.cuh:57–58`), by the generic kernel's DFlash special case (`rope.cuh:193–195`), by the
  `rmsnorm_rope` kernel (`src/ops/rmsnorm_rope/kernel.cuh:2` includes the header), and by
  `context_kv_materialize/materialize.cu:49–50`.

## 3. Kernel dispatch: what breaks

### 3.1 Op API surface

`include/ninfer/ops/rope.h` — two overloads:

```
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k, cudaStream_t stream);
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);
```

The API already carries `theta`; the wrapper (`src/ops/wrapper/rope.cpp`) validates
`theta > 0, finite` (L67), `rotary_dim` positive/even (L70), and the mode/shape envelope
(L86–95: vision D72/R72, text D256/R64 or D128/R128, MRoPE R64). Nothing in the *API*
restricts θ to 1e7 — the restriction lives entirely in dispatch and the fixed kernels.

### 3.2 Fast path (fixed kernels)

`src/ops/launcher/rope.cu`:

- `launch_fixed_pair` (L76–119) keys on:
  - DFlash 32/8: `axes==1 && q.ne[0]==128 && rotary_dim==128 && theta==1.0e7F && q.ne[1]==32 && k.ne[1]==8` (L80–81) → token-count-tuned split/one-CTA-per-token launches (L82–89).
  - Text 24/4 and 16/2: `rotary_dim==64 && theta==1.0e7F`, axes 1 → `Text1D`, axes 3 → `TextMrope` (L92–112).
  - Vision 16/16: `axes==2 && rotary_dim==72 && theta==10'000.0F` (L114–117).
- `launch_fixed_single_dispatch` (L140–167) mirrors the same predicates for the single-tensor
  form (DFlash 32 or 8 heads; text 24/4/16/2; vision 16).
- On predicate miss, `launch_generic` (L169–180) → `rope_generic_kernel` (`rope.cuh:180–229`).

What breaks if frequencies become a runtime device table:

1. **Dispatch predicate is θ-only.** YaRN keeps `rope_theta = 1e7` in the model config while
   applying per-pair extrapolation on top of it. Every text/dflash call site would still match
   `theta == 1.0e7F` and dispatch to the fixed kernels, which then rotate with the *unextended*
   `__constant__` table — wrong results with no error. The predicate must gain a
   table-identity / YaRN-active dimension (e.g. a frequency-table pointer or a
   `yarn_factor != 1` flag in the Op call).
2. **The fixed kernels read the table from constant memory.** `fixed_sincos`
   (`rope.cuh:54–68`) resolves the pair frequency from `kTextRopeInvFrequency` /
   `kVisionRopeInvFrequency` / `kDflashRopeInvFrequency` — compile-time globals, one value
   per program, no per-model/factor upload. To keep the tuned kernels, either
   (a) replace the table lookup with a kernel-parameter pointer to a program-owned device
   buffer (contents identical to today's values when YaRN is off), or
   (b) keep `__constant__` and `cudaMemcpyToSymbol` the YaRN table before graph capture
   (constant memory is graph-safe: the graph bakes the symbol, not its values; one program =
   one model + one factor, so the table never changes after startup).
3. **Double-precision DFlash table.** `kDflashRopeInvFrequency` is `double[64]` with manual
   two-pi range reduction (`dflash_rope.cuh:36–44`) because positions reach ~2^18 and
   single-precision `sincosf` loses accuracy there. Any YaRN replacement table must preserve
   this precision strategy (table in double, or the same reduction) for the DFlash geometry.

### 3.3 Generic path

`rope_generic_kernel` (`rope.cuh:180–229`) computes per pair:
`frequency = powf(theta, -2*pair/rotary_dim)` (or `-2*(pair%18)/36` for vision,
`rope.cuh:168–178`) — i.e. it already supports an *arbitrary scalar θ* from a kernel
parameter, fully device-resident and graph-safe. Its one table-based branch is the DFlash
special case at L193–195 (`theta == 1e7` again → `kDflashRopeInvFrequency`).

So the generic path breaks in exactly two ways:

- It cannot express YaRN's non-power-law table — it would need the same table-pointer input.
- The L193 `theta == 1e7` DFlash special case inherits the same silent-mismatch hazard as the
  fast path (a YaRN DFlash call with θ=1e7 would take the unextended double table).

Practical weight: every shipped geometry (rows 1–9 of §2) resolves to a fixed kernel today;
the generic path is the fallback for unregistered shapes. It still needs the table input for
consistency, and the op test's oracle (below) runs the same shapes through both.

### 3.4 Graph-capture constraints

- **Capture point.** `ProgramImpl::prepare_graphs()` (`src/models/qwen3_5/program/graphs.cpp:106`)
  runs inside the `ProgramImpl` constructor, gated by `use_cuda_graph`
  (`program_impl.cpp:303–307`, NVTX `CudaGraphPrepare`), **before the first user launch**.
  It captures one graph per (frontier bucket × batch size) profile per family: ordinary
  (L304–336), MTP (L338–372), DFlash (L373–421), then instantiates/upload/replay-warms each
  topology family (`instantiate_graph_family`, L41–102).
- **What the captured graphs contain.** The full decode round — including the `ops::rope`
  launches of rows 1, 2, 5, 6, 7 above (`ordinary_decode_batch`, `mtp_decode_batch`,
  `dflash_decode_batch` execute through `attn_mix`/`mtp_forward_*`/draft attention).
  Pre- and post-capture, `prepare_representative` (L188–291) seeds device-resident controls:
  `set_device_i32(io.pos/io.rope_pos)` (L212–213), DFlash `target_rope_positions`
  (L234–237), MTP `target_rope_positions` + `rope_deltas` (L264–266, L272), ordinary
  `rope_positions` (L283–284). `clear_stable_controls` (L160–177) zero-fills
  `io.rope_pos`/`io.rope_delta` etc. before capture.
- **Constraint for a runtime frequency table.** A CUDA graph bakes every kernel parameter at
  capture time — pointers and scalars alike. Therefore:
  1. The table (and anything the table pointer aliases) must be **device-resident before
     `prepare_graphs()` runs** — i.e. allocated and uploaded during planning/program
     construction. The established pattern is a planned persistent tensor: `io.rope_pos` is a
     `TensorRegion` in `RoundStateLayout` (`program/round_buffers.h:160`), allocated at
     planning time via `add_tensor(builder, DType::I32, {1}, "step rope position")`
     (`program/round_buffers.cpp:75`), written with `set_device_i32`
     (`program/program_impl.h:1136`, `program/storage/context.cpp:1543`) before capture.
     A frequency table can ride the same `RoundStateLayout`/persistent-layout mechanism.
  2. The **pointer must be stable across capture and replays** — a workspace-arena
     allocation is not acceptable (addresses move with `work.reset()`); the program-owned
     persistent region is.
  3. **Contents** may be updated after capture only via host/device memcpy into the fixed
     address — but for YaRN the table is static per (model, factor), so contents are written
     once at construction; only the *positions* keep changing per launch (and they already
     flow through device tensors, which is exactly what the graphs are built to read).
  4. **Scalar kernel parameters (θ, attention scale) are baked per graph.** The attention
     scale (`1/sqrt(head_dim)`, §4) is host-computed and passed by value; a YaRN
     `attention_factor` changes the *value* captured, not the mechanism — graph-safe with no
     device state. If the scale ever became per-launch-variable it would need a device
     scalar in the same stable region as the table.
  5. **Op-level precedent.** `tests/ops/test_rope.cpp:276–296` already captures `ops::rope` in
     a `cudaStreamBeginCapture` graph and replays it twice, so the Op contract implicitly
     admits capture; the constraint list above is about *where new state lives*, not about
     capture legality.
  6. **Position capacity.** Positions are I32 end to end (`checked_i32` guards at the ingress
     sites, e.g. `program/decode.cpp:323–324, 478–479, 672–673`; `checked_i32` at
     `graphs.cpp:212–213, 226–228, 235–236, 254, 265, 282–284`). YaRN factors up to ~8× on
     the 262144 native base stay far inside I32; no arithmetic change needed, but the
     guards' labels say "position capacity" and should be read as `max_context`-bounded.

## 4. Attention-scale call sites (the `1/sqrt(head_dim)` inline)

All host-computed `float` kernel arguments, baked per captured graph:

| File | Line | Op | Route |
|------|------|----|-------|
| `execution/text.cpp` | 386 | `ops::causal_softmax_attention` | MTP verify (batched) |
| `execution/text.cpp` | 394 | `ops::causal_softmax_attention` | MTP verify (single) |
| `execution/text.cpp` | 544 | `ops::causal_softmax_attention_cached` | MTP final-chunk attention |
| `execution/text.cpp` | 909 | `ops::causal_softmax_attention` | Text verify (batched) |
| `execution/text.cpp` | 918 | `ops::causal_softmax_attention` | Text verify (single) |
| `execution/draft.cpp` | 326 | `ops::sliding_window_attention` | DFlash2 local attention |
| `execution/draft.cpp` | 479 | `ops::sliding_window_attention` | DFlash local attention |
| `execution/draft.cpp` | 489 | `ops::context_softmax_attention` | DFlash full attention |
| `execution/vision.cpp` | 380 | `ops::packed_softmax_attention` | Vision (out of YaRN scope) |

(The two `1.0 / std::sqrt(...)` at `text.cpp:1031, 1038` are GDN linear-attention scales —
no RoPE, out of scope.)

YaRN's `attention_factor` multiplies into these nine expressions (eight text/draft; vision
excluded). Each is a literal `static_cast<float>(1.0 / std::sqrt(...head_dim))` inlined at the
call site — the touch is mechanical, but it is the *only* place the scale appears; there is
no shared helper, so the design must pick a home for `attention_factor` (e.g. a member of
`RopeConfig`/the program plan) and thread it into each site.

## 5. Contracts and tests

### 5.1 Op contract — `include/ninfer/ops/rope.h`

- Contract (L9–35): split-half NeoX rotation; mode table (Text 1-D, DFlash full-head
  D128/R128, Text MRoPE, Vision 2-D); formula stated as
  `phi = positions[t] * theta^(-2*i/rotary_dim)` (L19) — a **pure power law**, which is what
  YaRN invalidates.
- Oracle clause (L30–34): "The oracle evaluates the rotated dimensions naively in FP64 from
  the represented inputs. The updated BF16 values are promoted and compared directly…
  Unrotated dimensions remain bit-exact. … The Op uses no workspace or persistent state."
  For YaRN, the *represented inputs* must come to include the frequency table (or the YaRN
  parameters from which the oracle derives it) — the contract text, the API, and the oracle
  all change together.
- `rmsnorm_rope` contract (`include/ninfer/ops/rmsnorm_rope.h:1–42`) hard-codes
  `angle(i) = position * (1e7)^(-2*i/128)` (L18) with fixed D128/R128 32/8 geometry and W/B
  bounds — no θ parameter at all. If YaRN applies to the draft path, this Op's formula,
  signature, and oracle clause need the same table input.

### 5.2 Op test — `tests/ops/test_rope.cpp` (481 lines, standalone `main`)

- Oracle: `rope_oracle` (L80–114) — naive FP64 split-half rotation; frequency derived from
  `Geometry.theta` by `pow(theta, exponent)` (L96) — generic in θ but **pure power law**.
- Criteria: scale-invariant pair-profile `kRopePointwisePairRtol = 6.9e-3` (L26, applied in
  `verify_rope_profile` L141–196); bit-exact passthrough for unrotated dims (L198–216);
  bit-exact padding (L218–230); `GuardedDeviceBuffer` canaries; `verify_exact` on positions.
- Case matrix (`main`, L444–478): text pair (axes 1/3 × widths 2/7/16 × batch 1/8),
  27b decode/mrope-prefill, 35b tail position 262'137, MTP single-K, vision packed QKV,
  DFlash proposal 32/8 and DFlash context single-K at 131'072 tokens.
- Graph-capture case: `run_pair_case(..., graph = true)` L233, L276–296 — captures
  `ops::rope` and replays twice (used for the 16×8 dflash2-lane case, L458).
- θ constants fixed at the top: `kTextTheta = 1.0e7F`, `kVisionTheta = 10'000.0F` (L20–21).
- Companion tests: `tests/ops/test_rmsnorm_rope.cpp` (fused DFlash op; calls at L210, L262)
  and `tests/ops/test_context_kv_materialize.cpp`.

### 5.3 Converter + loader contracts

- `tools/convert/qwen3_5.py` `_rope_source` (L55–71): merges `rope_scaling`/`rope_parameters`
  aliases; `_fixed` enforces `type`/`rope_type == "default"` (L63–64);
  **`factor != 1.0` → `ValueError("…: scaled RoPE is not implemented")` (L65–66)**. Text
  `rope_parameters` emitted at L153–157 (`rope_theta`, `partial_rotary_factor`,
  `mrope_section`); draft `rope_parameters = {"rope_theta": …}` at L252–258.
- `src/models/qwen3_5/config.cpp` `rope()` (L65–100): `require_members(value,
  {"rope_theta", "partial_rotary_factor", "mrope_section"}, {}, "text RoPE")` — a fourth
  member (e.g. a YaRN scaling block) is rejected today; `rotary_dim` derived from
  `partial_rotary_factor` (L72); MRoPE section/axis consistency checks (L76–98). Draft rope
  check at L228–229 (`rope_theta` only, label "draft RoPE").
- `tests/convert/test_qwen3_5.py` L343–383
  (`test_rope_aliases_preserve_supported_parameters_and_reject_changed_mathematics`):
  asserts `type: "default"` is accepted for both alias fields, `type`/`rope_type` ≠ default
  is rejected (`match=alias`), and the draft conversion yields
  `rope_parameters == {"rope_theta": 500_000}` (L373).
  **Gap:** the `factor: 2` case (L355, L381) is rejected by the earlier *type* check; the
  "scaled RoPE is not implemented" rejection (L65–66) has **no direct assertion** anywhere in
  `tests/convert/` (verified by search). The YaRN work must add that assertion (or fold it
  into the replacement) so the acceptance/rejection boundary is pinned.

## 6. Capacity plumbing (every `max_context` / `max_position_embeddings` touch point)

| File | Line | What it does |
|------|------|--------------|
| `apps/cli/options.cpp` | 137–138 | `--max-context N` → `EngineOptions.max_context` (default 2048, `apps/cli/options.h:23`); `--kv-capacity` derived from it (L211, L223–224) |
| `apps/cli/main.cpp` | 273 | `engine_options.max_context = cli.max_context` |
| `apps/perplexity/main.cpp` | 214–216 | `purpose = CausalScoring`, `max_context = options.context` |
| `src/models/qwen3_5/program/planning/startup.cpp` | 730–747 | `validate_target_options`: **draft gate** `max_context > draft->max_position_embeddings` → error (L740–743); **text gate** `max_context == 0 || max_context > text.max_position_embeddings` → "max_context exceeds the configured position capacity" (L744–747) |
| same | 67 | `page_count(capacity)` — KV page arithmetic |
| same | 754–760 | `logical_pages = page_count(options.max_context)`; max page-count overflow guard |
| same | 761–777 | `kv_capacity` explicit-mode bounds vs `max_context` |
| same | 888–902 | `SequencePlanningInputs`: `.capacity = options.max_context` (L890) |
| same | 903–910 | page bounds from `inputs.capacity` |
| same | 215 | DFlash full-KV layout `.max_context = plan.capacity` |
| same | 113 | KV execution-table `logical_page_capacity` from `plan.capacity` |
| `src/models/qwen3_5/program/program_impl.h` | 563 | `ProgramImpl::capacity` member (carries `max_context`) |
| `src/models/qwen3_5/program/program_impl.cpp` | 508 | memory report `max_context = capacity` |
| `src/models/qwen3_5/config.h` | 78, 116 | `TextConfig.max_position_embeddings`, `DraftConfig.max_position_embeddings` |
| `src/models/qwen3_5/config.cpp` | 118, 226 | parsed from the artifact config |

Observations for the YaRN change:

- The text gate (L744–747) is the **single validation** that must become
  "`max_context > max_position_embeddings` → activate YaRN with
  `factor = max_context / max_position_embeddings`" (per the map's standing decision).
- The draft gate (L740–743) is a separate constraint: DFlash has no YaRN path (θ=1e7 tables,
  §2 rows 5–8), so with a draft backend `max_context` must presumably stay ≤
  `draft->max_position_embeddings` — the design ticket must decide whether YaRN + draft is
  rejected or the draft gate is relaxed.
- KV sizing, page pools, execution tables, and graph-profile frontiers all derive from
  `max_context`/`capacity` *alone* — nothing derives from `max_position_embeddings` after the
  gate, so relaxing the gate does not change memory math; it only changes which positions the
  RoPE math sees.
- Positions are written as I32 at every ingress (`decode.cpp:323–324, 478–479, 672–673`,
  `prefill.cpp:642, 1159`, `graphs.cpp:212–213, 226–228, 235–236, 254, 265, 282–284`,
  `text.cpp:1275`) with `checked_i32` overflow guards — capacity-bounded but not
  θ-bounded.

## 7. Causal-scoring / perplexity route (confirmed)

`Program::causal_score` (`program.h:851`, `program.cpp:254`) → `ProgramImpl::causal_score`
(`program_impl.cpp:318–459`):

- Requires `causal_scoring` (set from `EnginePurpose::CausalScoring`, `startup.cpp:899`) and
  rejects generation-only features (no speculative, no vision, no CUDA graph, no context
  cache — L324–327).
- Loops `execution::prefill_text_chunk` (`program/prefill.cpp:61`) in
  `prefill_chunk`-sized tiles — the **same prefill entry** the generation path uses
  (also from `program/transactions/commit.cpp:372`).
- `prefill_text_chunk` drives `TextContext` layer execution; full-attention blocks go through
  `TextContext::run_layers` → `attn_mix` (`execution/text.cpp:1076–1108`, call at L1095) →
  `text_rope` (`text.cpp:879`) → `ops::rope` — **identical RoPE code** to generation.
- The perplexity binary (`apps/perplexity/main.cpp:214–216`) builds the same public
  `.ninfer` Engine with `purpose = CausalScoring`.

Conclusion: one route, one RoPE implementation; YaRN changes to the Op/kernels/validation
cover the perplexity route automatically. (Note: causal scoring programs run **without**
CUDA graphs per L324–327, so the graph-capture constraint binds the generation route only —
but the frequency table still must exist for the non-graph prefill path, which is
strictly a weaker requirement.)

## 8. Touch-point list for a runtime frequency table + attention scale

Minimal set, in dependency order:

1. **Op API** — `include/ninfer/ops/rope.h`: carry the frequency table (device pointer or
   a `std::span` of precomputed frequencies) instead of/in addition to scalar `theta`;
   update the mode table and the power-law formula in the contract (L9–35); extend the
   oracle clause (L30–34) so the oracle evaluates from the *represented* table.
2. **Wrapper** — `src/ops/wrapper/rope.cpp`: validate the table (length = `rotary_dim/2`,
   finite, positive), keep the θ validation as the non-YaRN default;
   `src/ops/launcher/rope.h` + `rope.cu`: thread the table into `launch_fixed_pair`
   (L76–119), `launch_fixed_single_dispatch` (L140–167), `launch_generic` (L169–180);
   **change the dispatch predicates** from `theta == 1.0e7F` / `theta == 10'000.0F` to
   table-identity predicates (default table → today's fixed kernels; extended table →
   table-reading kernels).
3. **Kernels** — `src/ops/kernel/rope.cuh`: `rope_fixed_kernel` / `rope_fixed_split_kernel`
   read the table from a parameter (global or constant memory) instead of
   `fixed_axis_frequency` (L44–52); `rope_generic_kernel` replaces the
   `powf(theta, exponent)` branch (L197–205) and the DFlash `theta == 1e7` special case
   (L193–195) with the table. Preserve the double-precision path for D128
   (`dflash_rope.cuh:36–44`) — either the table is double or the same range reduction is
   kept.
4. **DFlash fused surfaces** — `include/ninfer/ops/rmsnorm_rope.h` (contract L18 hard-codes
   θ=1e7) + `src/ops/rmsnorm_rope/kernel.cuh`; and
   `src/ops/context_kv_materialize/materialize.cu:49–50` (direct `dflash_rope_sincos`).
   Either both gain the table input, or the design declares DFlash non-extendable (then the
   draft gate at `startup.cpp:740–743` is the enforcement point).
5. **Program state** — allocate the table in the planned persistent layout
   (precedent: `RoundStateLayout.rope_pos`, `program/round_buffers.h:160`,
   `round_buffers.cpp:75`); upload at construction **before `prepare_graphs()`**
   (`program_impl.cpp:303–307`); stable pointer into the captured graphs.
   Compute the table on the host (FP64) from (θ, base max_position, factor, beta_fast/slow)
   at planning time.
6. **Model plumbing** — `src/models/qwen3_5/config.cpp` `rope()` (L65–100) + `config.h`
   `RopeConfig` (L36–42): accept the YaRN scaling block from the artifact config **or**
   derive it from `--max-context` per the map's standing decision (v1: factor from the
   option, spec defaults, no artifact `yarn` block); carry `attention_factor` alongside.
   `src/models/qwen3_5/execution/attention.cpp` `text_rope` (L47–57) passes the table into
   `ops::rope`; `execution/text.cpp` (L357, 490, 535, 879) and `execution/draft.cpp`
   (L211, 316, 457) are the model call sites that must receive it.
7. **Attention scale** — the nine inlined `1/sqrt(head_dim)` sites (§4 table, eight
   in-scope) multiply by `attention_factor`.
8. **Startup validation** — `startup.cpp:744–747` (activate YaRN above
   `max_position_embeddings`; validate `factor`, the YaRN parameter ranges, and
   `max_context` vs i32/position guards), plus the draft decision at L740–743.
9. **Converter** — `tools/convert/qwen3_5.py` `_rope_source` (L55–71): accept `type: "yarn"`
   (and the parameters: factor, original_max_position_embeddings, attention_factor,
   beta_fast/slow) or keep rejecting scaled RoPE and derive everything from
   `--max-context`; emit the chosen representation into `rope_parameters`.
10. **Tests** — `tests/ops/test_rope.cpp`: oracle gains the table (L80–114); new YaRN cases
    at shipped geometries including a graph-capture replay; pin the factor boundaries;
    `tests/convert/test_qwen3_5.py` L343–383: replace/extend the
    reject-changed-mathematics test and add the missing direct assertion for
    "scaled RoPE is not implemented" (or its replacement); `test_rmsnorm_rope.cpp` /
    `test_context_kv_materialize.cpp` if the DFlash surfaces change.
11. **Docs** — `docs/maintainer/op-development.md`-style contract references,
    `README.md`/`docs/cli.md` for the `--max-context`-activated YaRN behavior, and
    `docs/perplexity.md` (shared route, so no separate note needed beyond the capacity
    bound).

## 9. Open questions for the design ticket (#9)

1. **Table representation.** Precomputed inverse-frequency array (float or double,
   `rotary_dim/2` entries) vs. passing (θ, factor, beta) and computing per-pair in the
   kernel. A precomputed array is simpler, matches how the oracle would consume it, and
   keeps kernels table-agnostic; but DFlash's double-precision/range-reduction behavior
   argues for either double storage or keeping the reduction with float storage.
2. **Dispatch keying.** How does the Op know "this is the default table, use today's exact
   kernels" vs. "extended table"? Options: identity by value comparison is out (hot path);
   a caller flag (e.g. `bool extended` / `std::span` that is empty for the default), or
   constant-memory update per program. Note the hazard: θ alone is *not* a distinguishing
   key under YaRN (θ stays 1e7).
3. **DFlash / MTP scope.** The map's decision activates YaRN on the text path. DFlash rows
   5–8 (§2) are non-extendable without touching three kernels; MTP rows 2–4 ride the text
   table for free. Decide: reject `--max-context` extension when a draft backend is
   selected (keep the L740–743 gate), or extend the draft tables too.
4. **`rmsnorm_rope` θ-less contract.** Today the fused DFlash Op has no θ parameter at all,
   while `DraftConfig.rope_theta` is parsed and used by the non-fused `ops::rope` calls —
   a latent inconsistency for any draft with θ≠1e7 (converter defaults to 1e7). The YaRN
   change is the natural moment to either thread θ/the table through or delete the
   inconsistency explicitly.
5. **`attention_factor` home.** It belongs to the RoPE scaling (not the attention Op);
   suggest `RopeConfig`/the program plan, threaded into the eight call sites. Confirm the
   factor semantics (Qwen YaRN applies it to the attention score pre-softmax — verify
   against the reference implementation at design time).
6. **Table upload ordering.** The table must be resident before `prepare_graphs()`; with
   `use_cuda_graph = false` (the causal-scoring route) there is no capture, but the
   non-graph prefill path still reads the table — so the upload belongs at program
   construction unconditionally, not inside the graph phase.
7. **MRoPE × YaRN interaction.** The 27b target uses MRoPE (3-axis positions, `mrope_section`
   — same table across axes, §2 row 1). Qwen's YaRN spec is defined for 1-D positions;
   confirm per-axis behavior (same table for all axes) against the reference for the e2e
   artifact (qwen3_8_27b, MRoPE).
8. **e2e oracle.** The map's standing decision requires independent-oracle math
   qualification. The existing `rope_oracle` is the right shape; the question is which
   reference defines the YaRN table values (transformers `YaRNScalingRotaryEmbedding`
   numbers vs. the Qwen spec's `attention_factor`/beta defaults) and at which positions
   (native-range vs. extended-range) the e2e perplexity comparison is taken.
