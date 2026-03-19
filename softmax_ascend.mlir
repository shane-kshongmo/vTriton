// RUN: ascend-perf-model-opt %s -assign-op-ids -estimate-cycles -pipeline-analysis | FileCheck %s

// Softmax test in AscendModel dialect
// This is equivalent to the Triton softmax kernel

// CHECK: ascend.reduce_max
// CHECK: ascend.exp
// CHECK: ascend.reduce_sum
// CHECK: ascend.div

module {
  func.func @softmax_16x16(%input: tensor<16x16xf32>) -> tensor<16x16xf32> {
    // Load input from HBM
    %input_loaded = ascend.vector_load %input {bytes = 1024 : i64} 
      : tensor<16x16xf32> -> tensor<16x16xf32>
    
    // Step 1: Find max along axis 1 for numerical stability
    %max = ascend.reduce_max %input_loaded axis = 1 
      : tensor<16x16xf32> -> tensor<16x1xf32>
    
    // Step 2: Broadcast max back to original shape
    %max_broadcast = ascend.broadcast %max to [16, 16] 
      : tensor<16x1xf32> -> tensor<16x16xf32>
    
    // Step 3: Subtract max (x - max)
    %shifted = ascend.sub %input_loaded, %max_broadcast 
      : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    
    // Step 4: Compute exp(x - max)
    %exp_val = ascend.exp %shifted 
      : tensor<16x16xf32> -> tensor<16x16xf32>
    
    // Step 5: Sum along axis 1
    %sum = ascend.reduce_sum %exp_val axis = 1 
      : tensor<16x16xf32> -> tensor<16x1xf32>
    
    // Step 6: Broadcast sum
    %sum_broadcast = ascend.broadcast %sum to [16, 16] 
      : tensor<16x1xf32> -> tensor<16x16xf32>
    
    // Step 7: Divide exp by sum
    %result = ascend.div %exp_val, %sum_broadcast 
      : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    
    // Store result to HBM
    ascend.vector_store %result {bytes = 1024 : i64} : tensor<16x16xf32>
    
    return %result : tensor<16x16xf32>
  }
  
  // Larger softmax for performance testing
  func.func @softmax_256x256(%input: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %input_loaded = ascend.vector_load %input {bytes = 262144 : i64} 
      : tensor<256x256xf32> -> tensor<256x256xf32>
    
    %max = ascend.reduce_max %input_loaded axis = 1 
      : tensor<256x256xf32> -> tensor<256x1xf32>
    
    %max_broadcast = ascend.broadcast %max to [256, 256] 
      : tensor<256x1xf32> -> tensor<256x256xf32>
    
    %shifted = ascend.sub %input_loaded, %max_broadcast 
      : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    
    %exp_val = ascend.exp %shifted 
      : tensor<256x256xf32> -> tensor<256x256xf32>
    
    %sum = ascend.reduce_sum %exp_val axis = 1 
      : tensor<256x256xf32> -> tensor<256x1xf32>
    
    %sum_broadcast = ascend.broadcast %sum to [256, 256] 
      : tensor<256x1xf32> -> tensor<256x256xf32>
    
    %result = ascend.div %exp_val, %sum_broadcast 
      : (tensor<256x256xf32>, tensor<256x256xf32>) -> tensor<256x256xf32>
    
    ascend.vector_store %result {bytes = 262144 : i64} : tensor<256x256xf32>
    
    return %result : tensor<256x256xf32>
  }
}
