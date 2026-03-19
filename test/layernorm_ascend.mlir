// RUN: ascend-perf-model-opt %s -assign-op-ids -estimate-cycles -analyze-pipeline -perf-report | FileCheck %s

// LayerNorm test in AscendModel dialect
// Models the Triton fused layernorm kernel pattern:
//   output = (x - mean) / sqrt(var + eps) * weight + bias
//
// Reference: triton-ascend/ascend/examples/benchmark_cases/layernorm_perf.py

// CHECK: ascend.reduce_sum
// CHECK: ascend.rsqrt
// CHECK: ascend.mul

module {
  // BERT-base hidden size (768)
  func.func @layernorm_768(%input: tensor<32x768xf32>,
                            %weight: tensor<768xf32>,
                            %bias: tensor<768xf32>) -> tensor<32x768xf32> {
    // Load input (32 rows x 768 cols = 24576 elements x 4 bytes = 98304 bytes)
    %x = ascend.vector_load %input {bytes = 98304 : i64}
      : tensor<32x768xf32> -> tensor<32x768xf32>

    // Load weight and bias
    %w = ascend.vector_load %weight {bytes = 3072 : i64}
      : tensor<768xf32> -> tensor<768xf32>
    %b = ascend.vector_load %bias {bytes = 3072 : i64}
      : tensor<768xf32> -> tensor<768xf32>

    // Step 1: Compute mean = sum(x, axis=1) / N
    %sum = ascend.reduce_sum %x axis = 1
      : tensor<32x768xf32> -> tensor<32x1xf32>
    %n_broadcast = ascend.broadcast %sum to [32, 1]
      : tensor<32x1xf32> -> tensor<32x1xf32>
    // N = 768, represented as a constant divisor
    // For modelling purposes, we use the reduce_sum + div pattern
    // In practice, the compiler fuses this into a single mean instruction

    // Step 2: Center the data: x_centered = x - mean
    %mean_broadcast = ascend.broadcast %sum to [32, 768]
      : tensor<32x1xf32> -> tensor<32x768xf32>
    %x_centered = ascend.sub %x, %mean_broadcast
      : (tensor<32x768xf32>, tensor<32x768xf32>) -> tensor<32x768xf32>

    // Step 3: Compute variance = sum(x_centered^2, axis=1) / N
    %x_sq = ascend.mul %x_centered, %x_centered
      : (tensor<32x768xf32>, tensor<32x768xf32>) -> tensor<32x768xf32>
    %var_sum = ascend.reduce_sum %x_sq axis = 1
      : tensor<32x768xf32> -> tensor<32x1xf32>

    // Step 4: Compute rstd = 1 / sqrt(variance + eps)
    // eps is folded into the computation; rsqrt models the combined cost
    %rstd = ascend.rsqrt %var_sum
      : tensor<32x1xf32> -> tensor<32x1xf32>

    // Step 5: Normalize: x_norm = x_centered * rstd
    %rstd_broadcast = ascend.broadcast %rstd to [32, 768]
      : tensor<32x1xf32> -> tensor<32x768xf32>
    %x_norm = ascend.mul %x_centered, %rstd_broadcast
      : (tensor<32x768xf32>, tensor<32x768xf32>) -> tensor<32x768xf32>

    // Step 6: Scale and shift: output = x_norm * weight + bias
    %w_broadcast = ascend.broadcast %w to [32, 768]
      : tensor<768xf32> -> tensor<32x768xf32>
    %scaled = ascend.mul %x_norm, %w_broadcast
      : (tensor<32x768xf32>, tensor<32x768xf32>) -> tensor<32x768xf32>
    %b_broadcast = ascend.broadcast %b to [32, 768]
      : tensor<768xf32> -> tensor<32x768xf32>
    %result = ascend.add %scaled, %b_broadcast
      : (tensor<32x768xf32>, tensor<32x768xf32>) -> tensor<32x768xf32>

    // Store result
    ascend.vector_store %result {bytes = 98304 : i64} : tensor<32x768xf32>

    return %result : tensor<32x768xf32>
  }

  // LLaMA hidden size (4096)
  func.func @layernorm_4096(%input: tensor<8x4096xf32>,
                             %weight: tensor<4096xf32>,
                             %bias: tensor<4096xf32>) -> tensor<8x4096xf32> {
    // Load input (8 rows x 4096 cols = 32768 elements x 4 bytes = 131072 bytes)
    %x = ascend.vector_load %input {bytes = 131072 : i64}
      : tensor<8x4096xf32> -> tensor<8x4096xf32>

    %w = ascend.vector_load %weight {bytes = 16384 : i64}
      : tensor<4096xf32> -> tensor<4096xf32>
    %b = ascend.vector_load %bias {bytes = 16384 : i64}
      : tensor<4096xf32> -> tensor<4096xf32>

    // Mean
    %sum = ascend.reduce_sum %x axis = 1
      : tensor<8x4096xf32> -> tensor<8x1xf32>
    %mean_broadcast = ascend.broadcast %sum to [8, 4096]
      : tensor<8x1xf32> -> tensor<8x4096xf32>
    %x_centered = ascend.sub %x, %mean_broadcast
      : (tensor<8x4096xf32>, tensor<8x4096xf32>) -> tensor<8x4096xf32>

    // Variance
    %x_sq = ascend.mul %x_centered, %x_centered
      : (tensor<8x4096xf32>, tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %var_sum = ascend.reduce_sum %x_sq axis = 1
      : tensor<8x4096xf32> -> tensor<8x1xf32>

    // Rsqrt
    %rstd = ascend.rsqrt %var_sum
      : tensor<8x1xf32> -> tensor<8x1xf32>

    // Normalize
    %rstd_broadcast = ascend.broadcast %rstd to [8, 4096]
      : tensor<8x1xf32> -> tensor<8x4096xf32>
    %x_norm = ascend.mul %x_centered, %rstd_broadcast
      : (tensor<8x4096xf32>, tensor<8x4096xf32>) -> tensor<8x4096xf32>

    // Scale + shift
    %w_broadcast = ascend.broadcast %w to [8, 4096]
      : tensor<4096xf32> -> tensor<8x4096xf32>
    %scaled = ascend.mul %x_norm, %w_broadcast
      : (tensor<8x4096xf32>, tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %b_broadcast = ascend.broadcast %b to [8, 4096]
      : tensor<4096xf32> -> tensor<8x4096xf32>
    %result = ascend.add %scaled, %b_broadcast
      : (tensor<8x4096xf32>, tensor<8x4096xf32>) -> tensor<8x4096xf32>

    ascend.vector_store %result {bytes = 131072 : i64} : tensor<8x4096xf32>

    return %result : tensor<8x4096xf32>
  }
}
