// RUN: ascend-perf-model-opt %s -assign-op-ids -estimate-cycles | FileCheck %s

// This tests the AscendModel dialect operations directly

// CHECK: ascend.matmul
// CHECK-SAME: estimated_cycles
// CHECK: ascend.cube_load
// CHECK: ascend.add

module {
  func.func @matmul_test(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> {
    // Cube load for input A
    %a_loaded = ascend.cube_load %a {bytes = 8192 : i64} 
      : tensor<64x64xf16> -> tensor<64x64xf16>
    
    // Cube load for input B
    %b_loaded = ascend.cube_load %b {bytes = 8192 : i64} 
      : tensor<64x64xf16> -> tensor<64x64xf16>
    
    // Matmul on Cube Core
    %c = ascend.matmul %a_loaded, %b_loaded {M = 64 : i64, N = 64 : i64, K = 64 : i64} 
      : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>
    
    // Store result via FixPipe
    ascend.cube_store %c {bytes = 8192 : i64} : tensor<64x64xf16>
    
    return %c : tensor<64x64xf16>
  }
  
  func.func @vector_test(%x: tensor<256xf32>, %y: tensor<256xf32>) -> tensor<256xf32> {
    // Vector load
    %x_loaded = ascend.vector_load %x {bytes = 1024 : i64}
      : tensor<256xf32> -> tensor<256xf32>
    
    %y_loaded = ascend.vector_load %y {bytes = 1024 : i64}
      : tensor<256xf32> -> tensor<256xf32>
    
    // Vector add operation
    %add = ascend.add %x_loaded, %y_loaded 
      : (tensor<256xf32>, tensor<256xf32>) -> tensor<256xf32>
    
    // Vector exp operation
    %exp = ascend.exp %add : tensor<256xf32> -> tensor<256xf32>
    
    // Vector store via MTE3
    ascend.vector_store %exp {bytes = 1024 : i64} : tensor<256xf32>
    
    return %exp : tensor<256xf32>
  }
  
  func.func @softmax_test(%x: tensor<1024xf32>) -> tensor<1024xf32> {
    // Load input
    %x_loaded = ascend.vector_load %x {bytes = 4096 : i64}
      : tensor<1024xf32> -> tensor<1024xf32>
    
    // Find max for numerical stability
    %max = ascend.reduce_max %x_loaded axis 0 : tensor<1024xf32> -> tensor<1xf32>
    
    // Broadcast max
    %max_broadcast = ascend.broadcast %max to [1024] : tensor<1xf32> -> tensor<1024xf32>
    
    // Subtract max
    %x_shifted = ascend.sub %x_loaded, %max_broadcast 
      : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    
    // Exp
    %exp = ascend.exp %x_shifted : tensor<1024xf32> -> tensor<1024xf32>
    
    // Sum
    %sum = ascend.reduce_sum %exp axis 0 : tensor<1024xf32> -> tensor<1xf32>
    
    // Broadcast sum
    %sum_broadcast = ascend.broadcast %sum to [1024] : tensor<1xf32> -> tensor<1024xf32>
    
    // Divide
    %result = ascend.div %exp, %sum_broadcast 
      : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    
    // Store result
    ascend.vector_store %result {bytes = 4096 : i64} : tensor<1024xf32>
    
    return %result : tensor<1024xf32>
  }
}
