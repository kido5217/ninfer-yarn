#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/rope.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t
attention_projection_workspace_bytes(const AttentionParameters& parameters, std::int32_t first,
                                     std::int32_t last);
void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream);

// `yarn` is the device-resident YaRN state (table + magnitude); a null pointer keeps the
// unextended power-law route structurally unchanged.
void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               const ops::YarnScale* yarn, cudaStream_t stream);
void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               const ops::YarnScale* yarn, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
