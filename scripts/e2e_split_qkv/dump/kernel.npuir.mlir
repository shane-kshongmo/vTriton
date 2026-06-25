func.func @split_qkv_rmsnorm_mrope_kernel(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg12: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<20xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.storage_aligned, mix_mode = "aiv", parallel_mode = "simd"} {
  %c32 = arith.constant 32 : index
  %c139776_i64 = arith.constant 139776 : i64
  %c138752_i64 = arith.constant 138752 : i64
  %c140032_i64 = arith.constant 140032 : i64
  %c136704_i64 = arith.constant 136704 : i64
  %c136448_i64 = arith.constant 136448 : i64
  %c135424_i64 = arith.constant 135424 : i64
  %c135168_i64 = arith.constant 135168 : i64
  %c134144_i64 = arith.constant 134144 : i64
  %c124928_i64 = arith.constant 124928 : i64
  %c120832_i64 = arith.constant 120832 : i64
  %c125952_i64 = arith.constant 125952 : i64
  %c112640_i64 = arith.constant 112640 : i64
  %c111616_i64 = arith.constant 111616 : i64
  %c107520_i64 = arith.constant 107520 : i64
  %c106496_i64 = arith.constant 106496 : i64
  %c102400_i64 = arith.constant 102400 : i64
  %c102144_i64 = arith.constant 102144 : i64
  %c171520_i64 = arith.constant 171520 : i64
  %c98048_i64 = arith.constant 98048 : i64
  %c98016_i64 = arith.constant 98016 : i64
  %c95968_i64 = arith.constant 95968 : i64
  %c95936_i64 = arith.constant 95936 : i64
  %c91840_i64 = arith.constant 91840 : i64
  %c90816_i64 = arith.constant 90816 : i64
  %c155136_i64 = arith.constant 155136 : i64
  %c74432_i64 = arith.constant 74432 : i64
  %c74400_i64 = arith.constant 74400 : i64
  %c66208_i64 = arith.constant 66208 : i64
  %c66080_i64 = arith.constant 66080 : i64
  %c49696_i64 = arith.constant 49696 : i64
  %c49440_i64 = arith.constant 49440 : i64
  %c49184_i64 = arith.constant 49184 : i64
  %c49152_i64 = arith.constant 49152 : i64
  %c49024_i64 = arith.constant 49024 : i64
  %c155008_i64 = arith.constant 155008 : i64
  %c48896_i64 = arith.constant 48896 : i64
  %c48864_i64 = arith.constant 48864 : i64
  %c48736_i64 = arith.constant 48736 : i64
  %c154880_i64 = arith.constant 154880 : i64
  %c48608_i64 = arith.constant 48608 : i64
  %c154752_i64 = arith.constant 154752 : i64
  %c48480_i64 = arith.constant 48480 : i64
  %c48448_i64 = arith.constant 48448 : i64
  %c48320_i64 = arith.constant 48320 : i64
  %c154624_i64 = arith.constant 154624 : i64
  %c48192_i64 = arith.constant 48192 : i64
  %c48160_i64 = arith.constant 48160 : i64
  %c48032_i64 = arith.constant 48032 : i64
  %c154496_i64 = arith.constant 154496 : i64
  %c47904_i64 = arith.constant 47904 : i64
  %c154368_i64 = arith.constant 154368 : i64
  %c47776_i64 = arith.constant 47776 : i64
  %c152320_i64 = arith.constant 152320 : i64
  %c45728_i64 = arith.constant 45728 : i64
  %c41632_i64 = arith.constant 41632 : i64
  %c150272_i64 = arith.constant 150272 : i64
  %c39584_i64 = arith.constant 39584 : i64
  %c23200_i64 = arith.constant 23200 : i64
  %c142080_i64 = arith.constant 142080 : i64
  %c15008_i64 = arith.constant 15008 : i64
  %c14752_i64 = arith.constant 14752 : i64
  %c14624_i64 = arith.constant 14624 : i64
  %c14496_i64 = arith.constant 14496 : i64
  %c14240_i64 = arith.constant 14240 : i64
  %c14112_i64 = arith.constant 14112 : i64
  %c13984_i64 = arith.constant 13984 : i64
  %c13952_i64 = arith.constant 13952 : i64
  %c13824_i64 = arith.constant 13824 : i64
  %c13312_i64 = arith.constant 13312 : i64
  %c12800_i64 = arith.constant 12800 : i64
  %c12672_i64 = arith.constant 12672 : i64
  %c12736_i64 = arith.constant 12736 : i64
  %c12416_i64 = arith.constant 12416 : i64
  %c12352_i64 = arith.constant 12352 : i64
  %c12288_i64 = arith.constant 12288 : i64
  %c12032_i64 = arith.constant 12032 : i64
  %c11904_i64 = arith.constant 11904 : i64
  %c11968_i64 = arith.constant 11968 : i64
  %c11648_i64 = arith.constant 11648 : i64
  %c11520_i64 = arith.constant 11520 : i64
  %c11584_i64 = arith.constant 11584 : i64
  %c11264_i64 = arith.constant 11264 : i64
  %c11008_i64 = arith.constant 11008 : i64
  %c10880_i64 = arith.constant 10880 : i64
  %c10624_i64 = arith.constant 10624 : i64
  %c10368_i64 = arith.constant 10368 : i64
  %c10240_i64 = arith.constant 10240 : i64
  %c2048_i64 = arith.constant 2048 : i64
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f16
  %c1_i32 = arith.constant 1 : i32
  %cst_0 = arith.constant -1.000000e+00 : f32
  %c64_i64 = arith.constant 64 : i64
  %c47_i64 = arith.constant 47 : i64
  %c48_i64 = arith.constant 48 : i64
  %c23_i64 = arith.constant 23 : i64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst_1 = arith.constant 0.000000e+00 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  %cst_3 = arith.constant 9.99999997E-7 : f32
  %cst_4 = arith.constant 1.280000e+02 : f32
  %c64_i32 = arith.constant 64 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c4096_i32 = arith.constant 4096 : i32
  %c6144_i32 = arith.constant 6144 : i32
  %c0_i32 = arith.constant 0 : i32
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg17, %arg18 : i32
  %1 = arith.muli %0, %arg19 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.muli %arg19, %arg18 : i32
  %5 = arith.divsi %3, %4 : i32
  %6 = arith.remsi %5, %arg17 : i32
  %7 = hivm.hir.pointer_cast(%c0_i64) : memref<8x64xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %7 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape : memref<512xf32, #hivm.address_space<ub>>)
  %8 = hivm.hir.pointer_cast(%c2048_i64) : memref<32x64xf32, #hivm.address_space<ub>>
  %collapse_shape_5 = memref.collapse_shape %8 [[0, 1]] : memref<32x64xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%collapse_shape_5 : memref<2048xf32, #hivm.address_space<ub>>)
  %9 = hivm.hir.pointer_cast(%c10240_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_1 : f32) outs(%9 : memref<32xf32, #hivm.address_space<ub>>)
  %10 = arith.cmpi sge, %6, %arg14 : i32
  %11 = arith.select %10, %arg16, %arg15 : i32
  %12 = arith.muli %arg15, %6 : i32
  %13 = scf.if %10 -> (i32) {
    %52 = arith.muli %arg15, %arg14 : i32
    %53 = arith.subi %6, %arg14 : i32
    %54 = arith.muli %53, %arg16 : i32
    %55 = arith.addi %52, %54 : i32
    scf.yield %55 : i32
  } else {
    scf.yield %12 : i32
  }
  %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1]>, #hivm.address_space<gm>>
  %14 = hivm.hir.pointer_cast(%c10368_i64) : memref<128xbf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast : memref<128xbf16, strided<[1]>, #hivm.address_space<gm>>) outs(%14 : memref<128xbf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [0], sizes: [128], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<128xbf16, strided<[1]>, #hivm.address_space<gm>>
  %15 = hivm.hir.pointer_cast(%c10624_i64) : memref<128xbf16, #hivm.address_space<ub>>
  hivm.hir.load ins(%reinterpret_cast_6 : memref<128xbf16, strided<[1]>, #hivm.address_space<gm>>) outs(%15 : memref<128xbf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
  %16 = hivm.hir.pointer_cast(%c10880_i64) : memref<32xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%16 : memref<32xi32, #hivm.address_space<ub>>)
  %17 = hivm.hir.pointer_cast(%c11008_i64) : memref<32xi64, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%16 : memref<32xi32, #hivm.address_space<ub>>) outs(%17 : memref<32xi64, #hivm.address_space<ub>>)
  %18 = hivm.hir.pointer_cast(%c11264_i64) : memref<32xi64, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%c23_i64 : i64) outs(%18 : memref<32xi64, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  %19 = hivm.hir.pointer_cast(%c11584_i64) : memref<32xi1, #hivm.address_space<ub>>
  %20 = hivm.hir.pointer_cast(%c11264_i64) : memref<32xi8, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  scf.for %arg20 = %c0 to %c32 step %c1 {
    %52 = memref.load %17[%arg20] : memref<32xi64, #hivm.address_space<ub>>
    %53 = memref.load %18[%arg20] : memref<32xi64, #hivm.address_space<ub>>
    %54 = arith.cmpi sgt, %52, %53 : i64
    %55 = arith.extui %54 : i1 to i8
    memref.store %55, %20[%arg20] : memref<32xi8, #hivm.address_space<ub>>
  }
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  %21 = hivm.hir.pointer_cast(%c11520_i64) : memref<32xf16, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%20 : memref<32xi8, #hivm.address_space<ub>>) outs(%21 : memref<32xf16, #hivm.address_space<ub>>)
  %22 = hivm.hir.pointer_cast(%c11584_i64) : memref<32xf16, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f16) outs(%22 : memref<32xf16, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%21, %22 : memref<32xf16, #hivm.address_space<ub>>, memref<32xf16, #hivm.address_space<ub>>) outs(%19 : memref<32xi1, #hivm.address_space<ub>>) compare_mode = <ne>
  %23 = hivm.hir.pointer_cast(%c11648_i64) : memref<32xi64, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%c48_i64 : i64) outs(%23 : memref<32xi64, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
  %24 = hivm.hir.pointer_cast(%c11968_i64) : memref<32xi1, #hivm.address_space<ub>>
  %25 = hivm.hir.pointer_cast(%c11648_i64) : memref<32xi8, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
  scf.for %arg20 = %c0 to %c32 step %c1 {
    %52 = memref.load %17[%arg20] : memref<32xi64, #hivm.address_space<ub>>
    %53 = memref.load %23[%arg20] : memref<32xi64, #hivm.address_space<ub>>
    %54 = arith.cmpi slt, %52, %53 : i64
    %55 = arith.extui %54 : i1 to i8
    memref.store %55, %25[%arg20] : memref<32xi8, #hivm.address_space<ub>>
  }
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
  %26 = hivm.hir.pointer_cast(%c11904_i64) : memref<32xf16, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%25 : memref<32xi8, #hivm.address_space<ub>>) outs(%26 : memref<32xf16, #hivm.address_space<ub>>)
  %27 = hivm.hir.pointer_cast(%c11968_i64) : memref<32xf16, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f16) outs(%27 : memref<32xf16, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%26, %27 : memref<32xf16, #hivm.address_space<ub>>, memref<32xf16, #hivm.address_space<ub>>) outs(%24 : memref<32xi1, #hivm.address_space<ub>>) compare_mode = <ne>
  %28 = hivm.hir.pointer_cast(%c11584_i64) : memref<32xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vand ins(%19, %24 : memref<32xi1, #hivm.address_space<ub>>, memref<32xi1, #hivm.address_space<ub>>) outs(%28 : memref<32xi1, #hivm.address_space<ub>>)
  %29 = hivm.hir.pointer_cast(%c12032_i64) : memref<32xi64, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%c47_i64 : i64) outs(%29 : memref<32xi64, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID2>]
  %30 = hivm.hir.pointer_cast(%c12288_i64) : memref<32xi1, #hivm.address_space<ub>>
  %31 = hivm.hir.pointer_cast(%c12032_i64) : memref<32xi8, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID2>]
  scf.for %arg20 = %c0 to %c32 step %c1 {
    %52 = memref.load %17[%arg20] : memref<32xi64, #hivm.address_space<ub>>
    %53 = memref.load %29[%arg20] : memref<32xi64, #hivm.address_space<ub>>
    %54 = arith.cmpi sgt, %52, %53 : i64
    %55 = arith.extui %54 : i1 to i8
    memref.store %55, %31[%arg20] : memref<32xi8, #hivm.address_space<ub>>
  }
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID2>]
  %32 = hivm.hir.pointer_cast(%c12288_i64) : memref<32xf16, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID2>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%31 : memref<32xi8, #hivm.address_space<ub>>) outs(%32 : memref<32xf16, #hivm.address_space<ub>>)
  %33 = hivm.hir.pointer_cast(%c12352_i64) : memref<32xf16, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f16) outs(%33 : memref<32xf16, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%32, %33 : memref<32xf16, #hivm.address_space<ub>>, memref<32xf16, #hivm.address_space<ub>>) outs(%30 : memref<32xi1, #hivm.address_space<ub>>) compare_mode = <ne>
  %34 = hivm.hir.pointer_cast(%c12416_i64) : memref<32xi64, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%c64_i64 : i64) outs(%34 : memref<32xi64, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID3>]
  %35 = hivm.hir.pointer_cast(%c12736_i64) : memref<32xi1, #hivm.address_space<ub>>
  %36 = hivm.hir.pointer_cast(%c12416_i64) : memref<32xi8, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID3>]
  scf.for %arg20 = %c0 to %c32 step %c1 {
    %52 = memref.load %17[%arg20] : memref<32xi64, #hivm.address_space<ub>>
    %53 = memref.load %34[%arg20] : memref<32xi64, #hivm.address_space<ub>>
    %54 = arith.cmpi slt, %52, %53 : i64
    %55 = arith.extui %54 : i1 to i8
    memref.store %55, %36[%arg20] : memref<32xi8, #hivm.address_space<ub>>
  }
  hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID3>]
  %37 = hivm.hir.pointer_cast(%c12672_i64) : memref<32xf16, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID3>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%36 : memref<32xi8, #hivm.address_space<ub>>) outs(%37 : memref<32xf16, #hivm.address_space<ub>>)
  %38 = hivm.hir.pointer_cast(%c12736_i64) : memref<32xf16, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f16) outs(%38 : memref<32xf16, #hivm.address_space<ub>>)
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%37, %38 : memref<32xf16, #hivm.address_space<ub>>, memref<32xf16, #hivm.address_space<ub>>) outs(%35 : memref<32xi1, #hivm.address_space<ub>>) compare_mode = <ne>
  %39 = hivm.hir.pointer_cast(%c12288_i64) : memref<32xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vand ins(%30, %35 : memref<32xi1, #hivm.address_space<ub>>, memref<32xi1, #hivm.address_space<ub>>) outs(%39 : memref<32xi1, #hivm.address_space<ub>>)
  %40 = arith.muli %arg13, %c64_i32 : i32
  %41 = hivm.hir.pointer_cast(%c12800_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vcast ins(%14 : memref<128xbf16, #hivm.address_space<ub>>) outs(%41 : memref<128xf32, #hivm.address_space<ub>>)
  %expand_shape = memref.expand_shape %41 [[0, 1]] output_shape [1, 128] : memref<128xf32, #hivm.address_space<ub>> into memref<1x128xf32, #hivm.address_space<ub>>
  %42 = hivm.hir.pointer_cast(%c13312_i64) : memref<128xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.vcast ins(%15 : memref<128xbf16, #hivm.address_space<ub>>) outs(%42 : memref<128xf32, #hivm.address_space<ub>>)
  %expand_shape_7 = memref.expand_shape %42 [[0, 1]] output_shape [1, 128] : memref<128xf32, #hivm.address_space<ub>> into memref<1x128xf32, #hivm.address_space<ub>>
  %43 = arith.index_cast %40 : i32 to index
  %44 = hivm.hir.pointer_cast(%c13824_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%44 : memref<32xf32, #hivm.address_space<ub>>)
  %45 = hivm.hir.pointer_cast(%c13952_i64) : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst_2 : f32) outs(%45 : memref<8xf32, #hivm.address_space<ub>>)
  %subview = memref.subview %14[0] [32] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<32xbf16, strided<[1]>, #hivm.address_space<ub>>
  %46 = hivm.hir.pointer_cast(%c13984_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%subview : memref<32xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%46 : memref<32xf32, #hivm.address_space<ub>>)
  %expand_shape_8 = memref.expand_shape %46 [[0, 1]] output_shape [1, 32] : memref<32xf32, #hivm.address_space<ub>> into memref<1x32xf32, #hivm.address_space<ub>>
  %subview_9 = memref.subview %14[32] [32] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<32xbf16, strided<[1], offset: 32>, #hivm.address_space<ub>>
  %47 = hivm.hir.pointer_cast(%c14112_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%subview_9 : memref<32xbf16, strided<[1], offset: 32>, #hivm.address_space<ub>>) outs(%47 : memref<32xf32, #hivm.address_space<ub>>)
  %expand_shape_10 = memref.expand_shape %47 [[0, 1]] output_shape [1, 32] : memref<32xf32, #hivm.address_space<ub>> into memref<1x32xf32, #hivm.address_space<ub>>
  %subview_11 = memref.subview %14[0] [64] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1]>, #hivm.address_space<ub>>
  %48 = hivm.hir.pointer_cast(%c14240_i64) : memref<2x32xf32, #hivm.address_space<ub>>
  %collapse_shape_12 = memref.collapse_shape %48 [[0, 1]] : memref<2x32xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%subview_11 : memref<64xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_12 : memref<64xf32, #hivm.address_space<ub>>)
  %expand_shape_13 = memref.expand_shape %48 [[0, 1], [2]] output_shape [1, 2, 32] : memref<2x32xf32, #hivm.address_space<ub>> into memref<1x2x32xf32, #hivm.address_space<ub>>
  %subview_14 = memref.subview %15[0] [32] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<32xbf16, strided<[1]>, #hivm.address_space<ub>>
  %49 = hivm.hir.pointer_cast(%c14496_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%subview_14 : memref<32xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%49 : memref<32xf32, #hivm.address_space<ub>>)
  %expand_shape_15 = memref.expand_shape %49 [[0, 1]] output_shape [1, 32] : memref<32xf32, #hivm.address_space<ub>> into memref<1x32xf32, #hivm.address_space<ub>>
  %subview_16 = memref.subview %15[32] [32] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<32xbf16, strided<[1], offset: 32>, #hivm.address_space<ub>>
  %50 = hivm.hir.pointer_cast(%c14624_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%subview_16 : memref<32xbf16, strided<[1], offset: 32>, #hivm.address_space<ub>>) outs(%50 : memref<32xf32, #hivm.address_space<ub>>)
  %expand_shape_17 = memref.expand_shape %50 [[0, 1]] output_shape [1, 32] : memref<32xf32, #hivm.address_space<ub>> into memref<1x32xf32, #hivm.address_space<ub>>
  %subview_18 = memref.subview %15[0] [64] [1] : memref<128xbf16, #hivm.address_space<ub>> to memref<64xbf16, strided<[1]>, #hivm.address_space<ub>>
  %51 = hivm.hir.pointer_cast(%c14752_i64) : memref<2x32xf32, #hivm.address_space<ub>>
  %collapse_shape_19 = memref.collapse_shape %51 [[0, 1]] : memref<2x32xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%subview_18 : memref<64xbf16, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_19 : memref<64xf32, #hivm.address_space<ub>>)
  %expand_shape_20 = memref.expand_shape %51 [[0, 1], [2]] output_shape [1, 2, 32] : memref<2x32xf32, #hivm.address_space<ub>> into memref<1x2x32xf32, #hivm.address_space<ub>>
  scf.for %arg20 = %c0_i32 to %11 step %c1_i32  : i32 {
    %52 = arith.index_cast %arg20 : i32 to index
    %53 = arith.index_cast %c0_i32 : i32 to index
    %54 = arith.index_cast %11 : i32 to index
    %55 = arith.index_cast %c1_i32 : i32 to index
    %56 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%52, %53, %55]
    %57 = arith.index_cast %56 : index to i1
    %c0_i64_21 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %58 = arith.select %57, %c0_i64_21, %c1_i64 : i64
    %59 = hivm.hir.pointer_cast(%c39584_i64, %c150272_i64) : memref<8x128xbf16, #hivm.address_space<ub>>
    annotation.mark %59 {hivm.multi_buffer = 2 : i32} : memref<8x128xbf16, #hivm.address_space<ub>>
    %60 = hivm.hir.pointer_cast(%c47904_i64, %c154496_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %60 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %61 = hivm.hir.pointer_cast(%c48192_i64, %c154624_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %61 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %62 = hivm.hir.pointer_cast(%c48480_i64, %c154752_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %62 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %63 = hivm.hir.pointer_cast(%c48608_i64, %c154880_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %63 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %64 = hivm.hir.pointer_cast(%c47776_i64, %c154368_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %64 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %65 = hivm.hir.pointer_cast(%c45728_i64, %c152320_i64) : memref<1024xbf16, #hivm.address_space<ub>>
    annotation.mark %65 {hivm.multi_buffer = 2 : i32} : memref<1024xbf16, #hivm.address_space<ub>>
    %66 = hivm.hir.pointer_cast(%c98048_i64, %c171520_i64) : memref<8x128xbf16, #hivm.address_space<ub>>
    annotation.mark %66 {hivm.multi_buffer = 2 : i32} : memref<8x128xbf16, #hivm.address_space<ub>>
    %67 = hivm.hir.pointer_cast(%c48896_i64, %c155008_i64) : memref<32xf32, #hivm.address_space<ub>>
    annotation.mark %67 {hivm.multi_buffer = 2 : i32} : memref<32xf32, #hivm.address_space<ub>>
    %68 = hivm.hir.pointer_cast(%c15008_i64, %c142080_i64) : memref<32x128xbf16, #hivm.address_space<ub>>
    annotation.mark %68 {hivm.multi_buffer = 2 : i32} : memref<32x128xbf16, #hivm.address_space<ub>>
    %69 = hivm.hir.pointer_cast(%c74432_i64, %c155136_i64) : memref<32x128xbf16, #hivm.address_space<ub>>
    annotation.mark %69 {hivm.multi_buffer = 2 : i32} : memref<32x128xbf16, #hivm.address_space<ub>>
    %70 = arith.addi %13, %arg20 : i32
    %71 = arith.muli %70, %c6144_i32 : i32
    %72 = arith.index_cast %71 : i32 to index
    %reinterpret_cast_22 = memref.reinterpret_cast %arg3 to offset: [%72], sizes: [32, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<32x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_23 = memref.collapse_shape %reinterpret_cast_22 [[0, 1]] : memref<32x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<4096xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_24 = memref.collapse_shape %68 [[0, 1]] : memref<32x128xbf16, #hivm.address_space<ub>> into memref<4096xbf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_23 : memref<4096xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_24 : memref<4096xbf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
    %73 = hivm.hir.pointer_cast(%c23200_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_25 = memref.collapse_shape %73 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.vcast ins(%collapse_shape_24 : memref<4096xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<4096xf32, #hivm.address_space<ub>>)
    %74 = affine.apply affine_map<()[s0] -> (s0 + 4096)>()[%72]
    %reinterpret_cast_26 = memref.reinterpret_cast %arg3 to offset: [%74], sizes: [8, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<8x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_27 = memref.collapse_shape %reinterpret_cast_26 [[0, 1]] : memref<8x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_28 = memref.collapse_shape %59 [[0, 1]] : memref<8x128xbf16, #hivm.address_space<ub>> into memref<1024xbf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_27 : memref<1024xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_28 : memref<1024xbf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
    %75 = hivm.hir.pointer_cast(%c41632_i64) : memref<8x128xf32, #hivm.address_space<ub>>
    %collapse_shape_29 = memref.collapse_shape %75 [[0, 1]] : memref<8x128xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.vcast ins(%collapse_shape_28 : memref<1024xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_29 : memref<1024xf32, #hivm.address_space<ub>>)
    %76 = affine.apply affine_map<()[s0] -> (s0 + 5120)>()[%72]
    %reinterpret_cast_30 = memref.reinterpret_cast %arg3 to offset: [%76], sizes: [1024], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1024xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %58]
    hivm.hir.load ins(%reinterpret_cast_30 : memref<1024xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%65 : memref<1024xbf16, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %77 = arith.muli %70, %c64_i32 : i32
    %78 = arith.index_cast %77 : i32 to index
    %79 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%78, %43]
    %80 = affine.apply affine_map<()[s0, s1] -> (s0 * 2 + s1)>()[%43, %78]
    %81 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%78]
    %82 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 + 32)>()[%43, %78]
    %83 = affine.apply affine_map<()[s0, s1] -> (s0 * 2 + s1 + 32)>()[%43, %78]
    %reinterpret_cast_31 = memref.reinterpret_cast %arg8 to offset: [%78], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_32 = memref.subview %reinterpret_cast_31[0] [24] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<24xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_33 = memref.subview %64[0] [24] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<24xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_1 : f32) outs(%64 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%subview_32 : memref<24xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_33 : memref<24xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %reinterpret_cast_34 = memref.reinterpret_cast %arg8 to offset: [%79], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_34 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%60 : memref<32xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID4>]
    %84 = hivm.hir.pointer_cast(%c48032_i64) : memref<32xf32, #hivm.address_space<ub>>
    %85 = hivm.hir.pointer_cast(%c48160_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID4>]
    hivm.hir.vsel ins(%28, %60, %9 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%84 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%85 : memref<8xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_35 = memref.reinterpret_cast %arg8 to offset: [%80], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_35 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%61 : memref<32xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID5>]
    %86 = hivm.hir.pointer_cast(%c48320_i64) : memref<32xf32, #hivm.address_space<ub>>
    %87 = hivm.hir.pointer_cast(%c48448_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID5>]
    hivm.hir.vsel ins(%39, %61, %9 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%86 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%87 : memref<8xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_36 = memref.reinterpret_cast %arg8 to offset: [%81], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_37 = memref.subview %reinterpret_cast_36[0] [24] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<24xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_38 = memref.subview %62[0] [24] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<24xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_1 : f32) outs(%62 : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.load ins(%subview_37 : memref<24xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_38 : memref<24xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_1 : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %reinterpret_cast_39 = memref.reinterpret_cast %arg8 to offset: [%82], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_39 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%63 : memref<32xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID6>]
    %88 = hivm.hir.pointer_cast(%c48736_i64) : memref<32xf32, #hivm.address_space<ub>>
    %89 = hivm.hir.pointer_cast(%c48864_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID6>]
    hivm.hir.vsel ins(%28, %63, %9 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%88 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%89 : memref<8xf32, #hivm.address_space<ub>>)
    %reinterpret_cast_40 = memref.reinterpret_cast %arg8 to offset: [%83], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.load ins(%reinterpret_cast_40 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%67 : memref<32xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %90 = hivm.hir.pointer_cast(%c49024_i64) : memref<32xf32, #hivm.address_space<ub>>
    %91 = hivm.hir.pointer_cast(%c49152_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vsel ins(%39, %67, %9 : memref<32xi1, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%90 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%91 : memref<8xf32, #hivm.address_space<ub>>)
    %92 = hivm.hir.pointer_cast(%c48032_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%64, %84 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%92 : memref<32xf32, #hivm.address_space<ub>>)
    %93 = hivm.hir.pointer_cast(%c48032_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%92, %86 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%93 : memref<32xf32, #hivm.address_space<ub>>)
    %expand_shape_41 = memref.expand_shape %93 [[0, 1]] output_shape [1, 32] : memref<32xf32, #hivm.address_space<ub>> into memref<1x32xf32, #hivm.address_space<ub>>
    %94 = hivm.hir.pointer_cast(%c49184_i64) : memref<2x32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_41 : memref<1x32xf32, #hivm.address_space<ub>>) outs(%94 : memref<2x32xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    hivm.hir.vadd ins(%62, %88 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%62 : memref<32xf32, #hivm.address_space<ub>>)
    %95 = hivm.hir.pointer_cast(%c49024_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%62, %90 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%95 : memref<32xf32, #hivm.address_space<ub>>)
    %expand_shape_42 = memref.expand_shape %95 [[0, 1]] output_shape [1, 32] : memref<32xf32, #hivm.address_space<ub>> into memref<1x32xf32, #hivm.address_space<ub>>
    %96 = hivm.hir.pointer_cast(%c49440_i64) : memref<2x32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%expand_shape_42 : memref<1x32xf32, #hivm.address_space<ub>>) outs(%96 : memref<2x32xf32, #hivm.address_space<ub>>) broadcast_dims = [0]
    %97 = hivm.hir.pointer_cast(%c49696_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %collapse_shape_43 = memref.collapse_shape %97 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_25, %collapse_shape_25 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_43 : memref<4096xf32, #hivm.address_space<ub>>)
    %98 = hivm.hir.pointer_cast(%c66080_i64) : memref<32x1xf32, #hivm.address_space<ub>>
    %99 = hivm.hir.pointer_cast(%c66208_i64) : memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%97 : memref<32x128xf32, #hivm.address_space<ub>>) outs(%98 : memref<32x1xf32, #hivm.address_space<ub>>) temp_buffer(%99 : memref<2048xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_44 = memref.collapse_shape %98 [[0, 1]] : memref<32x1xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %100 = hivm.hir.pointer_cast(%c66080_i64) : memref<32xf32, #hivm.address_space<ub>>
    %101 = hivm.hir.pointer_cast(%c74400_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%collapse_shape_44, %cst_4 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%100 : memref<32xf32, #hivm.address_space<ub>>) temp_buffer(%101 : memref<8xf32, #hivm.address_space<ub>>)
    %102 = hivm.hir.pointer_cast(%c66080_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%100, %cst_3 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%102 : memref<32xf32, #hivm.address_space<ub>>)
    %103 = hivm.hir.pointer_cast(%c66080_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsqrt ins(%102 : memref<32xf32, #hivm.address_space<ub>>) outs(%103 : memref<32xf32, #hivm.address_space<ub>>)
    %104 = hivm.hir.pointer_cast(%c66080_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%44, %103 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%104 : memref<32xf32, #hivm.address_space<ub>>)
    %expand_shape_45 = memref.expand_shape %104 [[0, 1]] output_shape [32, 1] : memref<32xf32, #hivm.address_space<ub>> into memref<32x1xf32, #hivm.address_space<ub>>
    %105 = hivm.hir.pointer_cast(%c74432_i64, %c155136_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    %106 = hivm.hir.pointer_cast(%c90816_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%73, %expand_shape_45 : memref<32x128xf32, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%105 : memref<32x128xf32, #hivm.address_space<ub>>) temp_buffer(%106 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    %107 = hivm.hir.pointer_cast(%c74432_i64, %c155136_i64) : memref<32x128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%105, %expand_shape : memref<32x128xf32, #hivm.address_space<ub>>, memref<1x128xf32, #hivm.address_space<ub>>) outs(%107 : memref<32x128xf32, #hivm.address_space<ub>>) broadcast = [0]
    %108 = hivm.hir.pointer_cast(%c91840_i64) : memref<8x128xf32, #hivm.address_space<ub>>
    %collapse_shape_46 = memref.collapse_shape %108 [[0, 1]] : memref<8x128xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_29, %collapse_shape_29 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_46 : memref<1024xf32, #hivm.address_space<ub>>)
    %109 = hivm.hir.pointer_cast(%c95936_i64) : memref<8x1xf32, #hivm.address_space<ub>>
    %110 = hivm.hir.pointer_cast(%c95968_i64) : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%108 : memref<8x128xf32, #hivm.address_space<ub>>) outs(%109 : memref<8x1xf32, #hivm.address_space<ub>>) temp_buffer(%110 : memref<512xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_47 = memref.collapse_shape %109 [[0, 1]] : memref<8x1xf32, #hivm.address_space<ub>> into memref<8xf32, #hivm.address_space<ub>>
    %111 = hivm.hir.pointer_cast(%c95936_i64) : memref<8xf32, #hivm.address_space<ub>>
    %112 = hivm.hir.pointer_cast(%c98016_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%collapse_shape_47, %cst_4 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%111 : memref<8xf32, #hivm.address_space<ub>>) temp_buffer(%112 : memref<8xf32, #hivm.address_space<ub>>)
    %113 = hivm.hir.pointer_cast(%c95936_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%111, %cst_3 : memref<8xf32, #hivm.address_space<ub>>, f32) outs(%113 : memref<8xf32, #hivm.address_space<ub>>)
    %114 = hivm.hir.pointer_cast(%c95936_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsqrt ins(%113 : memref<8xf32, #hivm.address_space<ub>>) outs(%114 : memref<8xf32, #hivm.address_space<ub>>)
    %115 = hivm.hir.pointer_cast(%c95936_i64) : memref<8xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vdiv ins(%45, %114 : memref<8xf32, #hivm.address_space<ub>>, memref<8xf32, #hivm.address_space<ub>>) outs(%115 : memref<8xf32, #hivm.address_space<ub>>)
    %expand_shape_48 = memref.expand_shape %115 [[0, 1]] output_shape [8, 1] : memref<8xf32, #hivm.address_space<ub>> into memref<8x1xf32, #hivm.address_space<ub>>
    %116 = hivm.hir.pointer_cast(%c98048_i64, %c171520_i64) : memref<8x128xf32, #hivm.address_space<ub>>
    %117 = hivm.hir.pointer_cast(%c102144_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%75, %expand_shape_48 : memref<8x128xf32, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%116 : memref<8x128xf32, #hivm.address_space<ub>>) temp_buffer(%117 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %118 = hivm.hir.pointer_cast(%c98048_i64, %c171520_i64) : memref<8x128xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%116, %expand_shape_7 : memref<8x128xf32, #hivm.address_space<ub>>, memref<1x128xf32, #hivm.address_space<ub>>) outs(%118 : memref<8x128xf32, #hivm.address_space<ub>>) broadcast = [0]
    %subview_49 = memref.subview %73[0, 0] [32, 32] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<32x32xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %119 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %120 = hivm.hir.pointer_cast(%c106496_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%subview_49, %expand_shape_45 : memref<32x32xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%119 : memref<32x32xf32, #hivm.address_space<ub>>) temp_buffer(%120 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    %121 = hivm.hir.pointer_cast(%c102400_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%119, %expand_shape_8 : memref<32x32xf32, #hivm.address_space<ub>>, memref<1x32xf32, #hivm.address_space<ub>>) outs(%121 : memref<32x32xf32, #hivm.address_space<ub>>) broadcast = [0]
    %subview_50 = memref.subview %73[0, 32] [32, 32] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<32x32xf32, strided<[128, 1], offset: 32>, #hivm.address_space<ub>>
    %122 = hivm.hir.pointer_cast(%c107520_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %123 = hivm.hir.pointer_cast(%c111616_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%subview_50, %expand_shape_45 : memref<32x32xf32, strided<[128, 1], offset: 32>, #hivm.address_space<ub>>, memref<32x1xf32, #hivm.address_space<ub>>) outs(%122 : memref<32x32xf32, #hivm.address_space<ub>>) temp_buffer(%123 : memref<256xf32, #hivm.address_space<ub>>) broadcast = [1]
    %124 = hivm.hir.pointer_cast(%c107520_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%122, %expand_shape_10 : memref<32x32xf32, #hivm.address_space<ub>>, memref<1x32xf32, #hivm.address_space<ub>>) outs(%124 : memref<32x32xf32, #hivm.address_space<ub>>) broadcast = [0]
    %125 = hivm.hir.pointer_cast(%c107520_i64) : memref<32x32xf32, #hivm.address_space<ub>>
    %collapse_shape_51 = memref.collapse_shape %124 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %collapse_shape_52 = memref.collapse_shape %125 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_51, %cst_0 : memref<1024xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_52 : memref<1024xf32, #hivm.address_space<ub>>)
    %subview_53 = memref.subview %8[0, 0] [32, 32] [1, 1] : memref<32x64xf32, #hivm.address_space<ub>> to memref<32x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%125, %cst_1 : memref<32x32xf32, #hivm.address_space<ub>>, f32) outs(%subview_53 : memref<32x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>)
    %126 = hivm.hir.pointer_cast(%c112640_i64) : memref<32x64xf32, #hivm.address_space<ub>>
    %collapse_shape_54 = memref.collapse_shape %126 [[0, 1]] : memref<32x64xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_5 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_54 : memref<2048xf32, #hivm.address_space<ub>>)
    %subview_55 = memref.subview %126[0, 0] [32, 32] [1, 1] : memref<32x64xf32, #hivm.address_space<ub>> to memref<32x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%subview_53 : memref<32x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_55 : memref<32x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>)
    %subview_56 = memref.subview %126[0, 32] [32, 32] [1, 1] : memref<32x64xf32, #hivm.address_space<ub>> to memref<32x32xf32, strided<[64, 1], offset: 32>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%121 : memref<32x32xf32, #hivm.address_space<ub>>) outs(%subview_56 : memref<32x32xf32, strided<[64, 1], offset: 32>, #hivm.address_space<ub>>)
    %subview_57 = memref.subview %73[0, 0] [32, 64] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<32x64xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %127 = hivm.hir.pointer_cast(%c125952_i64) : memref<32x2x32xf32, #hivm.address_space<ub>>
    %128 = hivm.hir.pointer_cast(%c120832_i64) : memref<32x1x32xf32, #hivm.address_space<ub>>
    %collapse_shape_58 = memref.collapse_shape %128 [[0, 1], [2]] : memref<32x1x32xf32, #hivm.address_space<ub>> into memref<32x32xf32, #hivm.address_space<ub>>
    %129 = hivm.hir.pointer_cast(%c124928_i64) : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%expand_shape_45 : memref<32x1xf32, #hivm.address_space<ub>>) outs(%collapse_shape_58 : memref<32x32xf32, #hivm.address_space<ub>>) temp_buffer(%129 : memref<256xf32, #hivm.address_space<ub>>) broadcast_dims = [1]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%128 : memref<32x1x32xf32, #hivm.address_space<ub>>) outs(%127 : memref<32x2x32xf32, #hivm.address_space<ub>>) broadcast_dims = [1]
    %130 = hivm.hir.pointer_cast(%c23200_i64) : memref<32x2x32xf32, #hivm.address_space<ub>>
    %collapse_shape_59 = memref.collapse_shape %127 [[0], [1, 2]] : memref<32x2x32xf32, #hivm.address_space<ub>> into memref<32x64xf32, #hivm.address_space<ub>>
    %collapse_shape_60 = memref.collapse_shape %130 [[0], [1, 2]] : memref<32x2x32xf32, #hivm.address_space<ub>> into memref<32x64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%subview_57, %collapse_shape_59 : memref<32x64xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<32x64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_60 : memref<32x64xf32, #hivm.address_space<ub>>)
    %131 = hivm.hir.pointer_cast(%c23200_i64) : memref<32x2x32xf32, #hivm.address_space<ub>>
    %collapse_shape_61 = memref.collapse_shape %expand_shape_13 [[0], [1, 2]] : memref<1x2x32xf32, #hivm.address_space<ub>> into memref<1x64xf32, #hivm.address_space<ub>>
    %collapse_shape_62 = memref.collapse_shape %131 [[0], [1, 2]] : memref<32x2x32xf32, #hivm.address_space<ub>> into memref<32x64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_60, %collapse_shape_61 : memref<32x64xf32, #hivm.address_space<ub>>, memref<1x64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_62 : memref<32x64xf32, #hivm.address_space<ub>>) broadcast = [0]
    %expand_shape_63 = memref.expand_shape %96 [[0, 1], [2]] output_shape [1, 2, 32] : memref<2x32xf32, #hivm.address_space<ub>> into memref<1x2x32xf32, #hivm.address_space<ub>>
    %132 = hivm.hir.pointer_cast(%c112640_i64) : memref<32x2x32xf32, #hivm.address_space<ub>>
    %collapse_shape_64 = memref.collapse_shape %expand_shape_63 [[0], [1, 2]] : memref<1x2x32xf32, #hivm.address_space<ub>> into memref<1x64xf32, #hivm.address_space<ub>>
    %collapse_shape_65 = memref.collapse_shape %132 [[0], [1, 2]] : memref<32x2x32xf32, #hivm.address_space<ub>> into memref<32x64xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%126, %collapse_shape_64 : memref<32x64xf32, #hivm.address_space<ub>>, memref<1x64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_65 : memref<32x64xf32, #hivm.address_space<ub>>) broadcast = [0]
    %expand_shape_66 = memref.expand_shape %94 [[0, 1], [2]] output_shape [1, 2, 32] : memref<2x32xf32, #hivm.address_space<ub>> into memref<1x2x32xf32, #hivm.address_space<ub>>
    %133 = hivm.hir.pointer_cast(%c23200_i64) : memref<32x2x32xf32, #hivm.address_space<ub>>
    %collapse_shape_67 = memref.collapse_shape %expand_shape_66 [[0], [1, 2]] : memref<1x2x32xf32, #hivm.address_space<ub>> into memref<1x64xf32, #hivm.address_space<ub>>
    %collapse_shape_68 = memref.collapse_shape %133 [[0], [1, 2]] : memref<32x2x32xf32, #hivm.address_space<ub>> into memref<32x64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_62, %collapse_shape_67 : memref<32x64xf32, #hivm.address_space<ub>>, memref<1x64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_68 : memref<32x64xf32, #hivm.address_space<ub>>) broadcast = [0]
    %134 = hivm.hir.pointer_cast(%c23200_i64) : memref<32x2x32xf32, #hivm.address_space<ub>>
    %collapse_shape_69 = memref.collapse_shape %132 [[0, 1, 2]] : memref<32x2x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_70 = memref.collapse_shape %133 [[0, 1, 2]] : memref<32x2x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_71 = memref.collapse_shape %134 [[0, 1, 2]] : memref<32x2x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_69, %collapse_shape_70 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_71 : memref<2048xf32, #hivm.address_space<ub>>)
    %collapse_shape_72 = memref.collapse_shape %134 [[0], [1, 2]] : memref<32x2x32xf32, #hivm.address_space<ub>> into memref<32x64xf32, #hivm.address_space<ub>>
    %subview_73 = memref.subview %75[0, 0] [8, 32] [1, 1] : memref<8x128xf32, #hivm.address_space<ub>> to memref<8x32xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %135 = hivm.hir.pointer_cast(%c134144_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %136 = hivm.hir.pointer_cast(%c135168_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%subview_73, %expand_shape_48 : memref<8x32xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%135 : memref<8x32xf32, #hivm.address_space<ub>>) temp_buffer(%136 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %137 = hivm.hir.pointer_cast(%c134144_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%135, %expand_shape_15 : memref<8x32xf32, #hivm.address_space<ub>>, memref<1x32xf32, #hivm.address_space<ub>>) outs(%137 : memref<8x32xf32, #hivm.address_space<ub>>) broadcast = [0]
    %subview_74 = memref.subview %75[0, 32] [8, 32] [1, 1] : memref<8x128xf32, #hivm.address_space<ub>> to memref<8x32xf32, strided<[128, 1], offset: 32>, #hivm.address_space<ub>>
    %138 = hivm.hir.pointer_cast(%c135424_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %139 = hivm.hir.pointer_cast(%c136448_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%subview_74, %expand_shape_48 : memref<8x32xf32, strided<[128, 1], offset: 32>, #hivm.address_space<ub>>, memref<8x1xf32, #hivm.address_space<ub>>) outs(%138 : memref<8x32xf32, #hivm.address_space<ub>>) temp_buffer(%139 : memref<64xf32, #hivm.address_space<ub>>) broadcast = [1]
    %140 = hivm.hir.pointer_cast(%c135424_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%138, %expand_shape_17 : memref<8x32xf32, #hivm.address_space<ub>>, memref<1x32xf32, #hivm.address_space<ub>>) outs(%140 : memref<8x32xf32, #hivm.address_space<ub>>) broadcast = [0]
    %141 = hivm.hir.pointer_cast(%c135424_i64) : memref<8x32xf32, #hivm.address_space<ub>>
    %collapse_shape_75 = memref.collapse_shape %140 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    %collapse_shape_76 = memref.collapse_shape %141 [[0, 1]] : memref<8x32xf32, #hivm.address_space<ub>> into memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_75, %cst_0 : memref<256xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_76 : memref<256xf32, #hivm.address_space<ub>>)
    %subview_77 = memref.subview %7[0, 0] [8, 32] [1, 1] : memref<8x64xf32, #hivm.address_space<ub>> to memref<8x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%141, %cst_1 : memref<8x32xf32, #hivm.address_space<ub>>, f32) outs(%subview_77 : memref<8x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>)
    %142 = hivm.hir.pointer_cast(%c136704_i64) : memref<8x64xf32, #hivm.address_space<ub>>
    %collapse_shape_78 = memref.collapse_shape %142 [[0, 1]] : memref<8x64xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape : memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_78 : memref<512xf32, #hivm.address_space<ub>>)
    %subview_79 = memref.subview %142[0, 0] [8, 32] [1, 1] : memref<8x64xf32, #hivm.address_space<ub>> to memref<8x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%subview_77 : memref<8x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>) outs(%subview_79 : memref<8x32xf32, strided<[64, 1]>, #hivm.address_space<ub>>)
    %subview_80 = memref.subview %142[0, 32] [8, 32] [1, 1] : memref<8x64xf32, #hivm.address_space<ub>> to memref<8x32xf32, strided<[64, 1], offset: 32>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%137 : memref<8x32xf32, #hivm.address_space<ub>>) outs(%subview_80 : memref<8x32xf32, strided<[64, 1], offset: 32>, #hivm.address_space<ub>>)
    %subview_81 = memref.subview %75[0, 0] [8, 64] [1, 1] : memref<8x128xf32, #hivm.address_space<ub>> to memref<8x64xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    %143 = hivm.hir.pointer_cast(%c140032_i64) : memref<8x2x32xf32, #hivm.address_space<ub>>
    %144 = hivm.hir.pointer_cast(%c138752_i64) : memref<8x1x32xf32, #hivm.address_space<ub>>
    %collapse_shape_82 = memref.collapse_shape %144 [[0, 1], [2]] : memref<8x1x32xf32, #hivm.address_space<ub>> into memref<8x32xf32, #hivm.address_space<ub>>
    %145 = hivm.hir.pointer_cast(%c139776_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%expand_shape_48 : memref<8x1xf32, #hivm.address_space<ub>>) outs(%collapse_shape_82 : memref<8x32xf32, #hivm.address_space<ub>>) temp_buffer(%145 : memref<64xf32, #hivm.address_space<ub>>) broadcast_dims = [1]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vbrc ins(%144 : memref<8x1x32xf32, #hivm.address_space<ub>>) outs(%143 : memref<8x2x32xf32, #hivm.address_space<ub>>) broadcast_dims = [1]
    %146 = hivm.hir.pointer_cast(%c140032_i64) : memref<8x2x32xf32, #hivm.address_space<ub>>
    %collapse_shape_83 = memref.collapse_shape %143 [[0], [1, 2]] : memref<8x2x32xf32, #hivm.address_space<ub>> into memref<8x64xf32, #hivm.address_space<ub>>
    %collapse_shape_84 = memref.collapse_shape %146 [[0], [1, 2]] : memref<8x2x32xf32, #hivm.address_space<ub>> into memref<8x64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%subview_81, %collapse_shape_83 : memref<8x64xf32, strided<[128, 1]>, #hivm.address_space<ub>>, memref<8x64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_84 : memref<8x64xf32, #hivm.address_space<ub>>)
    %147 = hivm.hir.pointer_cast(%c140032_i64) : memref<8x2x32xf32, #hivm.address_space<ub>>
    %collapse_shape_85 = memref.collapse_shape %expand_shape_20 [[0], [1, 2]] : memref<1x2x32xf32, #hivm.address_space<ub>> into memref<1x64xf32, #hivm.address_space<ub>>
    %collapse_shape_86 = memref.collapse_shape %147 [[0], [1, 2]] : memref<8x2x32xf32, #hivm.address_space<ub>> into memref<8x64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_84, %collapse_shape_85 : memref<8x64xf32, #hivm.address_space<ub>>, memref<1x64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_86 : memref<8x64xf32, #hivm.address_space<ub>>) broadcast = [0]
    %148 = hivm.hir.pointer_cast(%c136704_i64) : memref<8x2x32xf32, #hivm.address_space<ub>>
    %collapse_shape_87 = memref.collapse_shape %148 [[0], [1, 2]] : memref<8x2x32xf32, #hivm.address_space<ub>> into memref<8x64xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%142, %collapse_shape_64 : memref<8x64xf32, #hivm.address_space<ub>>, memref<1x64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_87 : memref<8x64xf32, #hivm.address_space<ub>>) broadcast = [0]
    %149 = hivm.hir.pointer_cast(%c140032_i64) : memref<8x2x32xf32, #hivm.address_space<ub>>
    %collapse_shape_88 = memref.collapse_shape %149 [[0], [1, 2]] : memref<8x2x32xf32, #hivm.address_space<ub>> into memref<8x64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_86, %collapse_shape_67 : memref<8x64xf32, #hivm.address_space<ub>>, memref<1x64xf32, #hivm.address_space<ub>>) outs(%collapse_shape_88 : memref<8x64xf32, #hivm.address_space<ub>>) broadcast = [0]
    %150 = hivm.hir.pointer_cast(%c136704_i64) : memref<8x2x32xf32, #hivm.address_space<ub>>
    %collapse_shape_89 = memref.collapse_shape %148 [[0, 1, 2]] : memref<8x2x32xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    %collapse_shape_90 = memref.collapse_shape %149 [[0, 1, 2]] : memref<8x2x32xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    %collapse_shape_91 = memref.collapse_shape %150 [[0, 1, 2]] : memref<8x2x32xf32, #hivm.address_space<ub>> into memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_89, %collapse_shape_90 : memref<512xf32, #hivm.address_space<ub>>, memref<512xf32, #hivm.address_space<ub>>) outs(%collapse_shape_91 : memref<512xf32, #hivm.address_space<ub>>)
    %collapse_shape_92 = memref.collapse_shape %150 [[0], [1, 2]] : memref<8x2x32xf32, #hivm.address_space<ub>> into memref<8x64xf32, #hivm.address_space<ub>>
    %subview_93 = memref.subview %107[0, 0] [32, 64] [1, 1] : memref<32x128xf32, #hivm.address_space<ub>> to memref<32x64xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_72 : memref<32x64xf32, #hivm.address_space<ub>>) outs(%subview_93 : memref<32x64xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
    %subview_94 = memref.subview %118[0, 0] [8, 64] [1, 1] : memref<8x128xf32, #hivm.address_space<ub>> to memref<8x64xf32, strided<[128, 1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_92 : memref<8x64xf32, #hivm.address_space<ub>>) outs(%subview_94 : memref<8x64xf32, strided<[128, 1]>, #hivm.address_space<ub>>)
    %151 = arith.muli %70, %c4096_i32 : i32
    %152 = arith.index_cast %151 : i32 to index
    %reinterpret_cast_95 = memref.reinterpret_cast %arg9 to offset: [%152], sizes: [32, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<32x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_96 = memref.collapse_shape %107 [[0, 1]] : memref<32x128xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %collapse_shape_97 = memref.collapse_shape %69 [[0, 1]] : memref<32x128xbf16, #hivm.address_space<ub>> into memref<4096xbf16, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%collapse_shape_96 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_97 : memref<4096xbf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %collapse_shape_98 = memref.collapse_shape %reinterpret_cast_95 [[0, 1]] : memref<32x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<4096xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_MTE3>]
    hivm.hir.store ins(%collapse_shape_97 : memref<4096xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_98 : memref<4096xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %153 = arith.muli %70, %c1024_i32 : i32
    %154 = arith.index_cast %153 : i32 to index
    %reinterpret_cast_99 = memref.reinterpret_cast %arg10 to offset: [%154], sizes: [8, 128], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<8x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %collapse_shape_100 = memref.collapse_shape %118 [[0, 1]] : memref<8x128xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
    %collapse_shape_101 = memref.collapse_shape %66 [[0, 1]] : memref<8x128xbf16, #hivm.address_space<ub>> into memref<1024xbf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_100 : memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_101 : memref<1024xbf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    %collapse_shape_102 = memref.collapse_shape %reinterpret_cast_99 [[0, 1]] : memref<8x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<1024xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID1>]
    hivm.hir.store ins(%collapse_shape_101 : memref<1024xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_102 : memref<1024xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    %reinterpret_cast_103 = memref.reinterpret_cast %arg11 to offset: [%154], sizes: [1024], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1024xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%65 : memref<1024xbf16, #hivm.address_space<ub>>) outs(%reinterpret_cast_103 : memref<1024xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %58]
  }
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
  return
}

warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
loc("/root/.triton/cache/bjTh98HpNersAeW1vTHWWqpRe7_m2_XUnkrY0U7GoTo/split_qkv_rmsnorm_mrope_kernel.ttadapter":76:11): warning: Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency
loc("/root/.triton/cache/bjTh98HpNersAeW1vTHWWqpRe7_m2_XUnkrY0U7GoTo/split_qkv_rmsnorm_mrope_kernel.ttadapter":77:11): warning: Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency
loc("/root/.triton/cache/bjTh98HpNersAeW1vTHWWqpRe7_m2_XUnkrY0U7GoTo/split_qkv_rmsnorm_mrope_kernel.ttadapter":79:11): warning: Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency
loc("/root/.triton/cache/bjTh98HpNersAeW1vTHWWqpRe7_m2_XUnkrY0U7GoTo/split_qkv_rmsnorm_mrope_kernel.ttadapter":80:11): warning: Op 'hivm.hir.vcmp' will execute by scalar instruction with low effiency
