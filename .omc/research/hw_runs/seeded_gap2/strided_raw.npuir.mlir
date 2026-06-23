// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @copy_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, false, false, false, false]> : vector<9xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c2048 = arith.constant 2048 : index
  %c0 = arith.constant 0 : index
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg6, %arg7 : i32
  %1 = arith.muli %0, %arg8 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg8, %arg7 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg6 : i32
  %7 = arith.index_cast %arg5 : i32 to index
  %8 = arith.index_cast %6 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [2048], strides: [%7] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
  %9 = hivm.hir.pointer_cast(%c0_i64) : memref<2048x8xf32, #hivm.address_space<ub>>
  %subview = memref.subview %9[0, 0] [2048, 1] [1, 1] : memref<2048x8xf32, #hivm.address_space<ub>> to memref<2048xf32, strided<[8]>, #hivm.address_space<ub>>
  %10 = affine.max affine_map<()[s0] -> (0, s0)>()[%7]
  %11 = affine.min affine_map<()[s0] -> (2048, s0)>()[%10]
  %12 = arith.cmpi slt, %11, %c2048 : index
  %subview_0 = memref.subview %reinterpret_cast[0] [%11] [1] : memref<2048xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
  %subview_1 = memref.subview %subview[0] [%11] [1] : memref<2048xf32, strided<[8]>, #hivm.address_space<ub>> to memref<?xf32, strided<[8]>, #hivm.address_space<ub>>
  scf.if %12 {
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %subview : memref<2048xf32, strided<[8]>, #hivm.address_space<ub>> -> memref<f32, #hivm.address_space<ub>>, index, index, index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [2048, 1], strides: [8, 1] : memref<f32, #hivm.address_space<ub>> to memref<2048x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%reinterpret_cast_4 : memref<2048x1xf32, strided<[8, 1]>, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview_0 : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>) outs(%subview_1 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%8], sizes: [2048], strides: [%7] : memref<?xf32, #hivm.address_space<gm>> to memref<2048xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
  %subview_3 = memref.subview %reinterpret_cast_2[0] [%11] [1] : memref<2048xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%subview_1 : memref<?xf32, strided<[8]>, #hivm.address_space<ub>>) outs(%subview_3 : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
