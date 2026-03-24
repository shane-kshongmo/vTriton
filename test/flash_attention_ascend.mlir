// RUN: ascend-perf-model-opt %s -assign-op-ids -estimate-cycles -analyze-pipeline -perf-report | FileCheck %s

// Simplified Flash Attention in AscendModel dialect
// Models the fused attention pattern from triton-ascend:
//   S = Q @ K^T (Cube)  →  P = softmax(S * scale) (Vector)  →  O = P @ V (Cube)
//
// This demonstrates Cube↔Vector interleaving and pipeline overlap analysis.
//
// Reference: triton-ascend-ops/tutorial/best_practice/002-decode_grouped_attention.py

// CHECK: ascend.matmul
// CHECK: ascend.reduce_max
// CHECK: ascend.exp
// CHECK: ascend.div

module {
  // Single-head attention: seq_len=64, head_dim=64, FP16
  // Models one block of flash attention (no outer loop over blocks)
  func.func @attention_single_block(
      %Q: tensor<64x64xf16>,    // Query [seq_len x head_dim]
      %K: tensor<64x64xf16>,    // Key [seq_len x head_dim]
      %V: tensor<64x64xf16>     // Value [seq_len x head_dim]
  ) -> tensor<64x64xf16> {

    // ===== Stage 1: Cube Path — Compute S = Q @ K^T =====

    // Load Q into Cube (L0A via MTE2→L1→MTE1)
    %q_loaded = ascend.cube_load %Q {bytes = 8192 : i64}
      : tensor<64x64xf16> -> tensor<64x64xf16>

    // Load K into Cube (L0B)
    %k_loaded = ascend.cube_load %K {bytes = 8192 : i64}
      : tensor<64x64xf16> -> tensor<64x64xf16>

    // S = Q @ K^T on Cube Core [64x64] @ [64x64] → [64x64]
    %S = ascend.matmul %q_loaded, %k_loaded {M = 64 : i64, N = 64 : i64, K = 64 : i64}
      : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>

    // ===== Stage 2: Vector Path — Softmax(S * scale) =====
    // Data moves from L0C → UB (Unified Buffer) for vector processing

    // Cast FP16 → FP32 for numerical stability in softmax
    %S_f32 = ascend.cast %S : tensor<64x64xf16> -> tensor<64x64xf32>

    // Scale: S_scaled = S * (1/sqrt(d_k)) where d_k = 64
    // scale = 1/sqrt(64) = 0.125, modelled as element-wise mul
    %S_scaled = ascend.mul %S_f32, %S_f32
      : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    // Note: In actual implementation, the second operand would be a
    // broadcasted scalar. Here we model the compute cost only.

    // Softmax row-wise: P = softmax(S_scaled, axis=1)
    // Step a: row_max = max(S_scaled, axis=1)
    %row_max = ascend.reduce_max %S_scaled axis = 1
      : tensor<64x64xf32> -> tensor<64x1xf32>

    // Step b: broadcast max back
    %max_bc = ascend.broadcast %row_max to [64, 64]
      : tensor<64x1xf32> -> tensor<64x64xf32>

    // Step c: S_shifted = S_scaled - row_max (numerical stability)
    %S_shifted = ascend.sub %S_scaled, %max_bc
      : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>

    // Step d: exp_val = exp(S_shifted)
    %exp_val = ascend.exp %S_shifted
      : tensor<64x64xf32> -> tensor<64x64xf32>

    // Step e: row_sum = sum(exp_val, axis=1)
    %row_sum = ascend.reduce_sum %exp_val axis = 1
      : tensor<64x64xf32> -> tensor<64x1xf32>

    // Step f: broadcast sum
    %sum_bc = ascend.broadcast %row_sum to [64, 64]
      : tensor<64x1xf32> -> tensor<64x64xf32>

    // Step g: P = exp_val / row_sum (attention weights)
    %P_f32 = ascend.div %exp_val, %sum_bc
      : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>

    // Cast back FP32 → FP16 for Cube
    %P = ascend.cast %P_f32 : tensor<64x64xf32> -> tensor<64x64xf16>

    // ===== Stage 3: Cube Path — Compute O = P @ V =====

    // Load V into Cube (L0B)
    %v_loaded = ascend.cube_load %V {bytes = 8192 : i64}
      : tensor<64x64xf16> -> tensor<64x64xf16>

    // O = P @ V on Cube Core [64x64] @ [64x64] → [64x64]
    %O = ascend.matmul %P, %v_loaded {M = 64 : i64, N = 64 : i64, K = 64 : i64}
      : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>

    // Store output via FixPipe (L0C → HBM)
    ascend.cube_store %O {bytes = 8192 : i64} : tensor<64x64xf16>

    return %O : tensor<64x64xf16>
  }

  // Multi-block flash attention with KV loop
  // Models the outer loop over key/value blocks
  func.func @attention_flash_loop(
      %Q: tensor<64x64xf16>,    // Query block [block_q x head_dim]
      %K_full: tensor<256x64xf16>,  // Full Key [seq_len x head_dim]
      %V_full: tensor<256x64xf16>   // Full Value [seq_len x head_dim]
  ) -> tensor<64x64xf16> {

    // Load Q once (reused across all K/V blocks)
    %q_loaded = ascend.cube_load %Q {bytes = 8192 : i64}
      : tensor<64x64xf16> -> tensor<64x64xf16>

    // Initialize accumulator O = zeros (modelled as a load of zero block)
    %O_init = ascend.vector_load %Q {bytes = 8192 : i64}
      : tensor<64x64xf16> -> tensor<64x64xf16>

    // Loop over 4 K/V blocks (seq_len=256, block_size=64, trip_count=4)
    // Each iteration processes one 64x64 block of K and V
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    %O_final = scf.for %i = %c0 to %c4 step %c1
        iter_args(%O_acc = %O_init) -> (tensor<64x64xf16>) {

      // Load K block into Cube
      %k_block = ascend.cube_load %K_full {bytes = 8192 : i64}
        : tensor<256x64xf16> -> tensor<64x64xf16>

      // S = Q @ K_block^T
      %S = ascend.matmul %q_loaded, %k_block {M = 64 : i64, N = 64 : i64, K = 64 : i64}
        : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>

      // Softmax on Vector (simplified — in flash attention this is "online softmax")
      %S_f32 = ascend.cast %S : tensor<64x64xf16> -> tensor<64x64xf32>

      %row_max = ascend.reduce_max %S_f32 axis = 1
        : tensor<64x64xf32> -> tensor<64x1xf32>
      %max_bc = ascend.broadcast %row_max to [64, 64]
        : tensor<64x1xf32> -> tensor<64x64xf32>
      %S_shifted = ascend.sub %S_f32, %max_bc
        : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
      %exp_val = ascend.exp %S_shifted
        : tensor<64x64xf32> -> tensor<64x64xf32>
      %row_sum = ascend.reduce_sum %exp_val axis = 1
        : tensor<64x64xf32> -> tensor<64x1xf32>
      %sum_bc = ascend.broadcast %row_sum to [64, 64]
        : tensor<64x1xf32> -> tensor<64x64xf32>
      %P_f32 = ascend.div %exp_val, %sum_bc
        : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
      %P = ascend.cast %P_f32 : tensor<64x64xf32> -> tensor<64x64xf16>

      // O_block = P @ V_block
      %v_block = ascend.cube_load %V_full {bytes = 8192 : i64}
        : tensor<256x64xf16> -> tensor<64x64xf16>
      %O_block = ascend.matmul %P, %v_block {M = 64 : i64, N = 64 : i64, K = 64 : i64}
        : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>

      // Accumulate (simplified — real flash attention rescales by lse correction)
      %O_new = ascend.add %O_acc, %O_block
        : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>

      scf.yield %O_new : tensor<64x64xf16>
    }

    // Store final output
    ascend.cube_store %O_final {bytes = 8192 : i64} : tensor<64x64xf16>

    return %O_final : tensor<64x64xf16>
  }
}
