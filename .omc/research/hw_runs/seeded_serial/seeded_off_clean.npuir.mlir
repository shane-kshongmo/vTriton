#map = affine_map<()[s0, s1] -> (s1, s0)>
#map1 = affine_map<()[s0, s1] -> (s1 + 2048, s0)>
#map2 = affine_map<()[s0, s1] -> (s0 - s1)>
module {
  func.func @seeded_serial_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, false, false, false, false, false]> : vector<12xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
    %c8192_i64 = arith.constant 8192 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c2048 = arith.constant 2048 : index
    %c512_i32 = arith.constant 512 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 9.99999974E-5 : f32
    %c2048_i32 = arith.constant 2048 : i32
    %cst_1 = arith.constant 1.000100e+00 : f32
    %c0 = arith.constant 0 : index
    hivm.hir.set_mask_norm
    %0 = arith.muli %arg9, %arg10 : i32
    %1 = arith.muli %0, %arg11 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = arith.muli %arg11, %arg10 : i32
    %5 = arith.divsi %3, %4 : i32
    %6 = arith.remsi %5, %arg9 : i32
    %7 = arith.muli %6, %c2048_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %9 = hivm.hir.pointer_cast(%c0_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %10 = arith.index_cast %arg7 : i32 to index
    %11 = affine.max #map()[%8, %10]
    %12 = affine.min #map1()[%11, %8]
    %13 = affine.apply #map2()[%12, %8]
    %14 = arith.cmpi slt, %13, %c2048 : index
    %subview = memref.subview %reinterpret_cast[0] [%13] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_2 = memref.subview %9[0] [%13] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %14 {
      hivm.hir.vbrc ins(%cst : f32) outs(%9 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%8], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_4 = memref.subview %reinterpret_cast_3[0] [%13] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_4 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%8], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %15 = hivm.hir.pointer_cast(%c16384_i64) : memref<2048xf32, #hivm.address_space<ub>>
    %16 = arith.index_cast %arg8 : i32 to index
    %17 = affine.max #map()[%8, %16]
    %18 = affine.min #map1()[%17, %8]
    %19 = affine.apply #map2()[%18, %8]
    %20 = arith.cmpi slt, %19, %c2048 : index
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%19] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_7 = memref.subview %15[0] [%19] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %20 {
      hivm.hir.vbrc ins(%cst : f32) outs(%15 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_6 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_7 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %21 = scf.for %arg12 = %c0_i32 to %c512_i32 step %c1_i32 iter_args(%arg13 = %15) -> (memref<2048xf32, #hivm.address_space<ub>>)  : i32 {
      %22 = hivm.hir.pointer_cast(%c8192_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.vmul ins(%arg13, %cst_1 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%22 : memref<2048xf32, #hivm.address_space<ub>>)
      %23 = hivm.hir.pointer_cast(%c16384_i64) : memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%22, %cst_0 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%23 : memref<2048xf32, #hivm.address_space<ub>>)
      scf.yield %23 : memref<2048xf32, #hivm.address_space<ub>>
    }
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%8], sizes: [2048], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_9 = memref.subview %21[0] [%19] [1] : memref<2048xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %subview_10 = memref.subview %reinterpret_cast_8[0] [%19] [1] : memref<2048xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.store ins(%subview_9 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_10 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    return
  }
}

