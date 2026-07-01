// RUN: ascend-perf-model-opt %s -assign-op-ids -estimate-cycles | FileCheck %s --check-prefix=CYCLES
// RUN: ascend-perf-model-opt %s -assign-op-ids -estimate-cycles -analyze-pipeline | FileCheck %s --check-prefix=SCHED

// Regression fixture for the triton-ascend #337 + #608 back-port:
//   * transfer ops carry optional src_space/dst_space attributes that select
//     the per-(src,dst,corenum) tilesim bandwidth and must round-trip;
//   * PipelineAnalysis serialises the AIV MTE2 (vec load) and MTE3 (vec store)
//     units (tilesim mutex clique) and publishes a loop-multiplied
//     `ascend.scheduled_cycles` summary consumed by the PerfReport pass.

// The src_space/dst_space attributes survive estimate-cycles and the op still
// gets an estimated_cycles annotation from the migrated transfer model.
// CYCLES: ascend.vector_load
// CYCLES-SAME: dst_space = "ub"
// CYCLES-SAME: estimated_cycles
// CYCLES-SAME: src_space = "hbm"
// CYCLES: ascend.vector_store
// CYCLES-SAME: dst_space = "hbm"
// CYCLES-SAME: src_space = "ub"

// PipelineAnalysis publishes the loop-multiplied roofline as scheduled_cycles.
// SCHED: ascend.scheduled_cycles

module {
  // Pure vector kernel: load -> add -> store. The load runs on VecMTE2 and the
  // store on MTE3; because those units are a mutex clique on 910B, the vector
  // path roofline must serialise them (VecMTE2 + MTE3) rather than overlap.
  func.func @vec_load_add_store(%x: tensor<1024xf32>, %y: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %xl = ascend.vector_load %x
        {bytes = 4096 : i64, src_space = "hbm", dst_space = "ub"}
        : tensor<1024xf32> -> tensor<1024xf32>
    %yl = ascend.vector_load %y
        {bytes = 4096 : i64, src_space = "hbm", dst_space = "ub"}
        : tensor<1024xf32> -> tensor<1024xf32>
    %s = ascend.add %xl, %yl
        : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    ascend.vector_store %s
        {bytes = 4096 : i64, src_space = "ub", dst_space = "hbm"}
        : tensor<1024xf32>
    return %s : tensor<1024xf32>
  }
}
