func.func @chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2_mix_aic(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg14: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg15: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg16: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg17: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg18: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg19: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg20: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg21: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg22: f32, %arg23: i32, %arg24: f32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<29xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c0_i64 = arith.constant 0 : i64
  %c0_i64_0 = arith.constant 0 : i64
  %c0_i64_1 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c1_i64_2 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c3_i64 = arith.constant 3 : i64
  %c4_i64 = arith.constant 4 : i64
  %c0_i64_3 = arith.constant 0 : i64
  %c90112_i64 = arith.constant 90112 : i64
  %c83968_i64 = arith.constant 83968 : i64
  %c73728_i64 = arith.constant 73728 : i64
  %c75776_i64 = arith.constant 75776 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %c49152_i64 = arith.constant 49152 : i64
  %c131072_i64 = arith.constant 131072 : i64
  %c71680_i64 = arith.constant 71680 : i64
  %c126976_i64 = arith.constant 126976 : i64
  %c67584_i64 = arith.constant 67584 : i64
  %c122880_i64 = arith.constant 122880 : i64
  %c63488_i64 = arith.constant 63488 : i64
  %c118784_i64 = arith.constant 118784 : i64
  %c59392_i64 = arith.constant 59392 : i64
  %c114688_i64 = arith.constant 114688 : i64
  %c55296_i64 = arith.constant 55296 : i64
  %c110592_i64 = arith.constant 110592 : i64
  %c51200_i64 = arith.constant 51200 : i64
  %c106496_i64 = arith.constant 106496 : i64
  %c47104_i64 = arith.constant 47104 : i64
  %c104448_i64 = arith.constant 104448 : i64
  %c45056_i64 = arith.constant 45056 : i64
  %c102400_i64 = arith.constant 102400 : i64
  %c43008_i64 = arith.constant 43008 : i64
  %c100352_i64 = arith.constant 100352 : i64
  %c40960_i64 = arith.constant 40960 : i64
  %c96256_i64 = arith.constant 96256 : i64
  %c36864_i64 = arith.constant 36864 : i64
  %c92160_i64 = arith.constant 92160 : i64
  %c32768_i64 = arith.constant 32768 : i64
  %c24576_i64 = arith.constant 24576 : i64
  %c16384_i64 = arith.constant 16384 : i64
  %c8192_i64 = arith.constant 8192 : i64
  %c0_i64_4 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c32_i64 = arith.constant 32 : i64
  %c128_i64 = arith.constant 128 : i64
  %c64_i32 = arith.constant 64 : i32
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %c4_i32 = arith.constant 4 : i32
  %true = arith.constant true
  %c1_i32 = arith.constant 1 : i32
  %c8192 = arith.constant 8192 : index
  %c16384 = arith.constant 16384 : index
  %c24576 = arith.constant 24576 : index
  %c32768 = arith.constant 32768 : index
  %c36864 = arith.constant 36864 : index
  %c40960 = arith.constant 40960 : index
  %c43008 = arith.constant 43008 : index
  %c45056 = arith.constant 45056 : index
  %c47104 = arith.constant 47104 : index
  %c51200 = arith.constant 51200 : index
  %c55296 = arith.constant 55296 : index
  %c59392 = arith.constant 59392 : index
  %c67584 = arith.constant 67584 : index
  %c75776 = arith.constant 75776 : index
  %c83968 = arith.constant 83968 : index
  %c100352 = arith.constant 100352 : index
  %c108544 = arith.constant 108544 : index
  %c112640 = arith.constant 112640 : index
  %c116736 = arith.constant 116736 : index
  %c120832 = arith.constant 120832 : index
  %c137216 = arith.constant 137216 : index
  %c145408 = arith.constant 145408 : index
  %c153600 = arith.constant 153600 : index
  %c161792 = arith.constant 161792 : index
  hivm.hir.set_ffts_base_addr %arg0
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg26, %arg27 : i32
  %1 = arith.muli %0, %arg28 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.divsi %3, %arg28 : i32
  %5 = arith.remsi %4, %arg27 : i32
  %6 = arith.muli %arg28, %arg27 : i32
  %7 = arith.divsi %3, %6 : i32
  %8 = arith.remsi %7, %arg26 : i32
  %9 = arith.remsi %5, %c32_i32 : i32
  %10 = arith.muli %8, %c2_i32 : i32
  %11 = arith.index_cast %10 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg21 to offset: [%11], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %12 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %13 = arith.trunci %12 : i64 to i32
  %14 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%11]
  %reinterpret_cast_5 = memref.reinterpret_cast %arg21 to offset: [%14], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %15 = memref.load %reinterpret_cast_5[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %16 = arith.trunci %15 : i64 to i32
  %17 = arith.index_cast %13 : i32 to index
  %reinterpret_cast_6 = memref.reinterpret_cast %arg20 to offset: [%17], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %18 = memref.load %reinterpret_cast_6[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %19 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%17]
  %reinterpret_cast_7 = memref.reinterpret_cast %arg20 to offset: [%19], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %20 = memref.load %reinterpret_cast_7[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %21 = arith.subi %20, %18 : i64
  %22 = arith.trunci %21 : i64 to i32
  %23 = arith.muli %16, %c64_i32 : i32
  %24 = arith.muli %18, %c32_i64 : i64
  %25 = arith.extsi %9 : i32 to i64
  %26 = arith.addi %24, %25 : i64
  %27 = arith.muli %26, %c128_i64 : i64
  %28 = arith.index_cast %27 : i64 to index
  %29 = arith.index_cast %23 : i32 to index
  %30 = arith.index_cast %22 : i32 to index
  %31 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%29, %30]
  %32 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%31, %29]
  %33 = affine.max affine_map<()[s0] -> (0, s0)>()[%29]
  %34 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%33, %29]
  %35 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%34, %29]
  %36 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%32, %29]
  %37 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%36, %35]
  %38 = arith.cmpi slt, %37, %c64 : index
  %39 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %40 = hivm.hir.pointer_cast(%c0_i64_4) : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
  %cast = memref.cast %40 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%39 : memref<64x64xbf16, #hivm.address_space<gm>>) outs(%cast : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
  %41 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %42 = hivm.hir.pointer_cast(%c8192_i64) : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
  %cast_8 = memref.cast %42 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%41 : memref<64x64xbf16, #hivm.address_space<gm>>) outs(%cast_8 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
  %43 = memref_ext.alloc_workspace() from %arg2 offset = [%c16384] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %44 = hivm.hir.pointer_cast(%c16384_i64) : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
  %cast_9 = memref.cast %44 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%43 : memref<64x64xbf16, #hivm.address_space<gm>>) outs(%cast_9 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
  %45 = memref_ext.alloc_workspace() from %arg2 offset = [%c24576] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %46 = hivm.hir.pointer_cast(%c24576_i64) : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
  %cast_10 = memref.cast %46 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%45 : memref<64x64xbf16, #hivm.address_space<gm>>) outs(%cast_10 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID7>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
  scf.for %arg29 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
    %54 = arith.index_cast %arg29 : i32 to index
    %55 = arith.index_cast %c0_i32 : i32 to index
    %56 = arith.index_cast %c4_i32 : i32 to index
    %57 = arith.index_cast %c1_i32 : i32 to index
    %58 = affine.apply affine_map<()[s0, s1, s2] -> ((s0 - s1) floordiv s2)>()[%54, %55, %57]
    %59 = arith.index_cast %58 : index to i64
    %60 = arith.index_cast %arg29 : i32 to index
    %61 = arith.index_cast %c0_i32 : i32 to index
    %62 = arith.index_cast %c4_i32 : i32 to index
    %63 = arith.index_cast %c1_i32 : i32 to index
    %64 = affine.apply affine_map<()[s0, s1, s2] -> (((s0 - s1) floordiv s2) mod 2)>()[%60, %61, %63]
    %65 = arith.index_cast %64 : index to i1
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %66 = arith.select %65, %c6_i64, %c7_i64 : i64
    %67 = arith.index_cast %64 : index to i1
    %c4_i64_16 = arith.constant 4 : i64
    %c5_i64 = arith.constant 5 : i64
    %68 = arith.select %67, %c4_i64_16, %c5_i64 : i64
    %69 = hivm.hir.pointer_cast(%c71680_i64, %c131072_i64) : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %69 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
    %70 = hivm.hir.pointer_cast(%c67584_i64, %c126976_i64) : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %70 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
    %71 = hivm.hir.pointer_cast(%c63488_i64, %c122880_i64) : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
    annotation.mark %71 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
    %72 = arith.cmpi eq, %arg29, %c0_i32 : i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 8
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 9
    scf.for %arg30 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %80 = arith.index_cast %arg30 : i32 to index
      %81 = arith.index_cast %c0_i32 : i32 to index
      %82 = arith.index_cast %c4_i32 : i32 to index
      %83 = arith.index_cast %c1_i32 : i32 to index
      %84 = arith.index_cast %arg29 : i32 to index
      %85 = arith.index_cast %c0_i32 : i32 to index
      %86 = arith.index_cast %c4_i32 : i32 to index
      %87 = arith.index_cast %c1_i32 : i32 to index
      %88 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> ((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3))>()[%80, %81, %82, %83, %84, %85, %87]
      %89 = arith.index_cast %88 : index to i64
      %90 = arith.index_cast %arg30 : i32 to index
      %91 = arith.index_cast %c0_i32 : i32 to index
      %92 = arith.index_cast %c4_i32 : i32 to index
      %93 = arith.index_cast %c1_i32 : i32 to index
      %94 = arith.index_cast %arg29 : i32 to index
      %95 = arith.index_cast %c0_i32 : i32 to index
      %96 = arith.index_cast %c4_i32 : i32 to index
      %97 = arith.index_cast %c1_i32 : i32 to index
      %98 = affine.apply affine_map<()[s0, s1, s2, s3, s4, s5, s6] -> (((s0 - s1) floordiv s3 + ((s4 - s5) floordiv s6) * ((-s1 + s2 + s3 - 1) floordiv s3)) mod 2)>()[%90, %91, %92, %93, %94, %95, %97]
      %99 = arith.index_cast %98 : index to i1
      %c2_i64_24 = arith.constant 2 : i64
      %c3_i64_25 = arith.constant 3 : i64
      %100 = arith.select %99, %c2_i64_24, %c3_i64_25 : i64
      %101 = arith.index_cast %98 : index to i1
      %c0_i64_26 = arith.constant 0 : i64
      %c1_i64_27 = arith.constant 1 : i64
      %102 = arith.select %101, %c0_i64_26, %c1_i64_27 : i64
      %103 = hivm.hir.pointer_cast(%c59392_i64, %c118784_i64) : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %103 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %104 = hivm.hir.pointer_cast(%c55296_i64, %c114688_i64) : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %104 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %105 = hivm.hir.pointer_cast(%c51200_i64, %c110592_i64) : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %105 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %106 = hivm.hir.pointer_cast(%c47104_i64, %c106496_i64) : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %106 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %107 = hivm.hir.pointer_cast(%c45056_i64, %c104448_i64) : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %107 {hivm.multi_buffer = 2 : i32} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
      %108 = hivm.hir.pointer_cast(%c43008_i64, %c102400_i64) : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %108 {hivm.multi_buffer = 2 : i32} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
      %109 = hivm.hir.pointer_cast(%c40960_i64, %c100352_i64) : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %109 {hivm.multi_buffer = 2 : i32} : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>>
      %110 = hivm.hir.pointer_cast(%c36864_i64, %c96256_i64) : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %110 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %111 = hivm.hir.pointer_cast(%c32768_i64, %c92160_i64) : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %111 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>>
      %112 = arith.muli %arg30, %c32_i32 : i32
      %113 = arith.index_cast %112 : i32 to index
      %114 = affine.max affine_map<()[s0] -> (128, s0)>()[%113]
      %115 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%114, %113]
      %116 = affine.max affine_map<()[s0] -> (0, s0)>()[%113]
      %117 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%116, %113]
      %118 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%117, %113]
      %119 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%115, %113]
      %120 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%119, %118]
      %121 = arith.cmpi slt, %120, %c32 : index
      %122 = arith.ori %38, %121 : i1
      %123 = memref_ext.alloc_workspace() from %arg2 offset = [%c32768] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_28 = memref.cast %111 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %102]
      hivm.hir.nd2nz {dst_continuous} ins(%123 : memref<64x32xbf16, #hivm.address_space<gm>>) outs(%cast_28 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      %124 = memref_ext.alloc_workspace() from %arg2 offset = [%c36864] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_29 = memref.cast %110 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%124 : memref<64x32xbf16, #hivm.address_space<gm>>) outs(%cast_29 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID3>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 3
      %125 = memref_ext.alloc_workspace() from %arg2 offset = [%c40960] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_30 = memref.cast %109 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%125 : memref<32x32xbf16, #hivm.address_space<gm>>) outs(%cast_30 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID4>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 4
      %126 = memref_ext.alloc_workspace() from %arg2 offset = [%c43008] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_31 = memref.cast %108 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%126 : memref<32x32xbf16, #hivm.address_space<gm>>) outs(%cast_31 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 5
      %127 = memref_ext.alloc_workspace() from %arg2 offset = [%c45056] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_32 = memref.cast %107 : memref<2x2x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%127 : memref<32x32xbf16, #hivm.address_space<gm>>) outs(%cast_32 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID2>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 6
      %128 = memref_ext.alloc_workspace() from %arg2 offset = [%c47104] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_33 = memref.cast %106 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%128 : memref<64x32xbf16, #hivm.address_space<gm>>) outs(%cast_33 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 7
      %129 = memref_ext.alloc_workspace() from %arg2 offset = [%c51200] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_34 = memref.cast %105 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %100]
      hivm.hir.nd2nz {dst_continuous} ins(%129 : memref<64x32xbf16, #hivm.address_space<gm>>) outs(%cast_34 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 8
      %130 = memref_ext.alloc_workspace() from %arg2 offset = [%c55296] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %cast_35 = memref.cast %104 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
      hivm.hir.pipe_barrier[<PIPE_MTE2>]
      hivm.hir.nd2nz {dst_continuous} ins(%130 : memref<64x32xbf16, #hivm.address_space<gm>>) outs(%cast_35 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
      hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 9
      %131 = hivm.hir.pointer_cast(%c0_i64_4) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %cast_36 = memref.cast %131 : memref<2x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
      %c-1_i64_37 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast_29, %cast_30, %true, %c64, %c32, %c32 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_36 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c3_i64, %c4_i64, %c-1_i64_37, %c-1_i64_37, %89, %c-1_i64_37, %c-1_i64_37 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %132 = memref_ext.alloc_workspace() from %arg2 offset = [%c59392] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 10
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_36 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%132 : memref<64x32xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
      annotation.mark %132 : memref<64x32xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
      %133 = hivm.hir.pointer_cast(%c8192_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %cast_38 = memref.cast %133 : memref<2x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
      %c-1_i64_39 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast_28, %cast_32, %true, %c64, %c32, %c32 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_38 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_39, %c2_i64, %c-1_i64_39, %c-1_i64_39, %89, %c-1_i64_39, %c-1_i64_39 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %134 = memref_ext.alloc_workspace() from %arg2 offset = [%c67584] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 11
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_38 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%134 : memref<64x32xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
      annotation.mark %134 : memref<64x32xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
      %135 = hivm.hir.pointer_cast(%c16384_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
      %cast_40 = memref.cast %135 : memref<2x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
      hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
      %c-1_i64_41 = arith.constant -1 : i64
      hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast_33, %cast_31, %true, %c64, %c32, %c32 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_40 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c1_i64_2, %c-1_i64_41, %c-1_i64_41, %102, %89, %c-1_i64_41, %c-1_i64_41 : i64, i64, i64, i64, i64, i64, i64)
      hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      %136 = memref_ext.alloc_workspace() from %arg2 offset = [%c75776] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 12
      hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_40 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%136 : memref<64x32xf32, #hivm.address_space<gm>>)
      hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
      annotation.mark %136 : memref<64x32xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
      scf.if %72 {
        %137 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 4096)>()[%113, %28, %29]
        %reinterpret_cast_42 = memref.reinterpret_cast %arg5 to offset: [%137], sizes: [64, 32], strides: [4096, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64x32xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
        %cast_43 = memref.cast %103 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
        %subview = memref.subview %reinterpret_cast_42[%35, %118] [%37, %120] [1, 1] : memref<64x32xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
        %138 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%119, %118]
        %139 = affine.apply affine_map<()[s0, s1] -> ((s0 - s1 + 15) floordiv 16)>()[%36, %35]
        %140 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%118]
        %141 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%118]
        %142 = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%35]
        %143 = affine.apply affine_map<()[s0] -> (s0 mod 16)>()[%35]
        %subview_44 = memref.subview %103[%140, %142, %143, %141] [%138, %139, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x16x16xbf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
        %cast_45 = memref.cast %subview_44 : memref<?x?x16x16xbf16, strided<[1024, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>
        scf.if %122 {
          %collapse_shape = memref.collapse_shape %103 [[0, 1, 2, 3]] : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>> into memref<2048xbf16, #hivm.address_space<cbuf>>
          hivm.hir.vbrc ins(%cst : bf16) outs(%collapse_shape : memref<2048xbf16, #hivm.address_space<cbuf>>)
          hivm.hir.pipe_barrier[<PIPE_MTE2>]
        } {hivm.unlikely_condition}
        hivm.hir.nd2nz {dst_continuous} ins(%subview : memref<?x?xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>) outs(%cast_45 : memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>, #hivm.address_space<cbuf>>) init_out_buffer = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
        %144 = hivm.hir.pointer_cast(%c24576_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
        %cast_46 = memref.cast %144 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
        %c-1_i64_47 = arith.constant -1 : i64
        hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%cast_34, %cast_43, %true, %c64, %c32, %37 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_46 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_47, %c0_i64_3, %c-1_i64_47, %c-1_i64_47, %89, %c-1_i64_47, %c-1_i64_47 : i64, i64, i64, i64, i64, i64, i64)
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %145 = memref_ext.alloc_workspace() from %arg2 offset = [%c83968] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xf32, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_46 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%145 : memref<64x64xf32, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
        annotation.mark %145 : memref<64x64xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
        %146 = hivm.hir.pointer_cast(%c40960_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
        %cast_48 = memref.cast %146 : memref<2x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast, %cast_35, %true, %c64, %c64, %c32 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_48 : memref<?x?x?x?xf32, #hivm.address_space<cc>>)
        hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        %147 = memref_ext.alloc_workspace() from %arg2 offset = [%c100352] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xf32, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_ALL>]
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 15
        hivm.hir.sync_block_set[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 14
        hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_48 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%147 : memref<64x32xf32, #hivm.address_space<gm>>)
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        annotation.mark %147 : memref<64x32xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
      }
      hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, %100]
    }
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 10
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 11
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 12
    %73 = memref_ext.alloc_workspace() from %arg2 offset = [%c108544] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_17 = memref.cast %71 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %68]
    hivm.hir.nd2nz {dst_continuous} ins(%73 : memref<64x32xbf16, #hivm.address_space<gm>>) outs(%cast_17 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    %74 = memref_ext.alloc_workspace() from %arg2 offset = [%c112640] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_18 = memref.cast %70 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %66]
    hivm.hir.nd2nz {dst_continuous} ins(%74 : memref<64x32xbf16, #hivm.address_space<gm>>) outs(%cast_18 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
    %75 = memref_ext.alloc_workspace() from %arg2 offset = [%c116736] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %cast_19 = memref.cast %69 : memref<2x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
    hivm.hir.nd2nz {dst_continuous} ins(%75 : memref<64x32xbf16, #hivm.address_space<gm>>) outs(%cast_19 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
    %76 = hivm.hir.pointer_cast(%c49152_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
    %cast_20 = memref.cast %76 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
    %c-1_i64_21 = arith.constant -1 : i64
    hivm.hir.mmadL1 {b_transpose, fixpipe_already_inserted = true} ins(%cast_17, %cast_19, %true, %c64, %c32, %c64 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_20 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_1, %c1_i64, %68, %c-1_i64_21, %59, %c-1_i64_21, %c-1_i64_21 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %77 = memref_ext.alloc_workspace() from %arg2 offset = [%c120832] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_20 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%77 : memref<64x64xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
    annotation.mark %77 : memref<64x64xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %78 = hivm.hir.pointer_cast(%c65536_i64) : memref<2x4x16x16xf32, #hivm.address_space<cc>>
    %cast_22 = memref.cast %78 : memref<2x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
    %c-1_i64_23 = arith.constant -1 : i64
    hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast_8, %cast_18, %true, %c64, %c64, %c32 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_22 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_23, %c-1_i64_23, %c-1_i64_23, %66, %59, %c-1_i64_23, %c-1_i64_23 : i64, i64, i64, i64, i64, i64, i64)
    hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    %79 = memref_ext.alloc_workspace() from %arg2 offset = [%c137216] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xf32, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_22 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%79 : memref<64x32xf32, #hivm.address_space<gm>>)
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
    annotation.mark %79 : memref<64x32xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
  }
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID7>]
  %47 = memref_ext.alloc_workspace() from %arg2 offset = [%c145408] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %48 = hivm.hir.pointer_cast(%c75776_i64) : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
  %cast_11 = memref.cast %48 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
  hivm.hir.nd2nz {dst_continuous} ins(%47 : memref<64x64xbf16, #hivm.address_space<gm>>) outs(%cast_11 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
  %49 = hivm.hir.pointer_cast(%c73728_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
  %cast_12 = memref.cast %49 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
  %c-1_i64 = arith.constant -1 : i64
  hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast_11, %cast_9, %true, %c64, %c64, %c64 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_12 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64_0, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
  hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
  %50 = memref_ext.alloc_workspace() from %arg2 offset = [%c153600] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, pre_quant = #hivm.fixpipe_pre_quant_mode<F322BF16>} ins(%cast_12 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%50 : memref<64x64xbf16, #hivm.address_space<gm>>)
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
  %51 = hivm.hir.pointer_cast(%c83968_i64) : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
  %cast_13 = memref.cast %51 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>> to memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.nd2nz {dst_continuous} ins(%50 : memref<64x64xbf16, #hivm.address_space<gm>>) outs(%cast_13 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>) init_out_buffer = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
  %52 = hivm.hir.pointer_cast(%c90112_i64) : memref<4x4x16x16xf32, #hivm.address_space<cc>>
  %cast_14 = memref.cast %52 : memref<4x4x16x16xf32, #hivm.address_space<cc>> to memref<?x?x?x?xf32, #hivm.address_space<cc>>
  %c-1_i64_15 = arith.constant -1 : i64
  hivm.hir.mmadL1 {fixpipe_already_inserted = true} ins(%cast_10, %cast_13, %true, %c64, %c64, %c64 : memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, memref<?x?x?x?xbf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%cast_14 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64_15, %c0_i64, %c-1_i64_15, %c-1_i64_15, %c-1_i64_15, %c-1_i64_15, %c-1_i64_15 : i64, i64, i64, i64, i64, i64, i64)
  hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
  %53 = memref_ext.alloc_workspace() from %arg2 offset = [%c161792] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xf32, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%cast_14 : memref<?x?x?x?xf32, #hivm.address_space<cc>>) outs(%53 : memref<64x64xf32, #hivm.address_space<gm>>)
  annotation.mark %53 : memref<64x64xf32, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_S>] flag = 0
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //
func.func @chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2_mix_aiv(%arg0: i64 {hacc.arg_type = #hacc.arg_type<ffts_base_address>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg2: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg3: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg10: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg14: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg15: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg16: memref<?xbf16, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg17: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg18: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg19: memref<?xf32, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg20: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg21: memref<?xi64, #hivm.address_space<gm>> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg22: f32, %arg23: i32, %arg24: f32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false, false, false, false]> : vector<29xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.storage_aligned, mix_mode = "mix", parallel_mode = "simd"} {
  %c24832_i64 = arith.constant 24832 : i64
  %c131840_i64 = arith.constant 131840 : i64
  %c123648_i64 = arith.constant 123648 : i64
  %c139904_i64 = arith.constant 139904 : i64
  %c169984_i64 = arith.constant 169984 : i64
  %c161792_i64 = arith.constant 161792 : i64
  %c57728_i64 = arith.constant 57728 : i64
  %c49536_i64 = arith.constant 49536 : i64
  %c41216_i64 = arith.constant 41216 : i64
  %c70144_i64 = arith.constant 70144 : i64
  %c61952_i64 = arith.constant 61952 : i64
  %c5216_i64 = arith.constant 5216 : i64
  %c98688_i64 = arith.constant 98688 : i64
  %c90496_i64 = arith.constant 90496 : i64
  %c74112_i64 = arith.constant 74112 : i64
  %c70016_i64 = arith.constant 70016 : i64
  %c61824_i64 = arith.constant 61824 : i64
  %c53632_i64 = arith.constant 53632 : i64
  %c45440_i64 = arith.constant 45440 : i64
  %c115072_i64 = arith.constant 115072 : i64
  %c5088_i64 = arith.constant 5088 : i64
  %c110976_i64 = arith.constant 110976 : i64
  %c106880_i64 = arith.constant 106880 : i64
  %c5024_i64 = arith.constant 5024 : i64
  %c41344_i64 = arith.constant 41344 : i64
  %c119168_i64 = arith.constant 119168 : i64
  %c39296_i64 = arith.constant 39296 : i64
  %c37248_i64 = arith.constant 37248 : i64
  %c4960_i64 = arith.constant 4960 : i64
  %c33152_i64 = arith.constant 33152 : i64
  %c29056_i64 = arith.constant 29056 : i64
  %c4896_i64 = arith.constant 4896 : i64
  %c24960_i64 = arith.constant 24960 : i64
  %c20864_i64 = arith.constant 20864 : i64
  %c800_i64 = arith.constant 800 : i64
  %c704_i64 = arith.constant 704 : i64
  %c640_i64 = arith.constant 640 : i64
  %c512_i64 = arith.constant 512 : i64
  %c384_i64 = arith.constant 384 : i64
  %c170112_i64 = arith.constant 170112 : i64
  %c161920_i64 = arith.constant 161920 : i64
  %c153728_i64 = arith.constant 153728 : i64
  %c153600_i64 = arith.constant 153600 : i64
  %c153472_i64 = arith.constant 153472 : i64
  %c20736_i64 = arith.constant 20736 : i64
  %c12544_i64 = arith.constant 12544 : i64
  %c8448_i64 = arith.constant 8448 : i64
  %c123264_i64 = arith.constant 123264 : i64
  %c178304_i64 = arith.constant 178304 : i64
  %c60384_i64 = arith.constant 60384 : i64
  %c153216_i64 = arith.constant 153216 : i64
  %c58336_i64 = arith.constant 58336 : i64
  %c149120_i64 = arith.constant 149120 : i64
  %c58240_i64 = arith.constant 58240 : i64
  %c58112_i64 = arith.constant 58112 : i64
  %c148992_i64 = arith.constant 148992 : i64
  %c256_i64 = arith.constant 256 : i64
  %c0_i64 = arith.constant 0 : i64
  %c57856_i64 = arith.constant 57856 : i64
  %c57600_i64 = arith.constant 57600 : i64
  %c148736_i64 = arith.constant 148736 : i64
  %c148480_i64 = arith.constant 148480 : i64
  %c140288_i64 = arith.constant 140288 : i64
  %c140160_i64 = arith.constant 140160 : i64
  %c123776_i64 = arith.constant 123776 : i64
  %c123520_i64 = arith.constant 123520 : i64
  %c0_i32 = arith.constant 0 : i32
  %c64_i64 = arith.constant 64 : i64
  %c32_i64 = arith.constant 32 : i64
  %c128_i64 = arith.constant 128 : i64
  %c64_i32 = arith.constant 64 : i32
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c4096_i32 = arith.constant 4096 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %c128_i32 = arith.constant 128 : i32
  %c4096_i64 = arith.constant 4096 : i64
  %cst_0 = arith.constant 0.000000e+00 : bf16
  %c16384_i64 = arith.constant 16384 : i64
  %c4_i32 = arith.constant 4 : i32
  %cst_1 = arith.constant 0.000000e+00 : f16
  %cst_2 = arith.constant 0.693147182 : f32
  %cst_3 = arith.constant -1.000000e+00 : f32
  %c1_i32 = arith.constant 1 : i32
  %c8192 = arith.constant 8192 : index
  %c16384 = arith.constant 16384 : index
  %c24576 = arith.constant 24576 : index
  %c32768 = arith.constant 32768 : index
  %c36864 = arith.constant 36864 : index
  %c40960 = arith.constant 40960 : index
  %c43008 = arith.constant 43008 : index
  %c45056 = arith.constant 45056 : index
  %c47104 = arith.constant 47104 : index
  %c51200 = arith.constant 51200 : index
  %c55296 = arith.constant 55296 : index
  %c59392 = arith.constant 59392 : index
  %c67584 = arith.constant 67584 : index
  %c75776 = arith.constant 75776 : index
  %c83968 = arith.constant 83968 : index
  %c100352 = arith.constant 100352 : index
  %c108544 = arith.constant 108544 : index
  %c112640 = arith.constant 112640 : index
  %c116736 = arith.constant 116736 : index
  %c120832 = arith.constant 120832 : index
  %c137216 = arith.constant 137216 : index
  %c145408 = arith.constant 145408 : index
  %c161792 = arith.constant 161792 : index
  hivm.hir.set_ffts_base_addr %arg0
  hivm.hir.set_mask_norm
  %0 = arith.muli %arg26, %arg27 : i32
  %1 = arith.muli %0, %arg28 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  %4 = arith.divsi %3, %arg28 : i32
  %5 = arith.remsi %4, %arg27 : i32
  %6 = arith.muli %arg28, %arg27 : i32
  %7 = arith.divsi %3, %6 : i32
  %8 = arith.remsi %7, %arg26 : i32
  %9 = hivm.hir.pointer_cast(%c123520_i64) : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%9 : memref<64xf32, #hivm.address_space<ub>>)
  %10 = hivm.hir.pointer_cast(%c123776_i64) : memref<64x64xf32, #hivm.address_space<ub>>
  %collapse_shape = memref.collapse_shape %10 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>)
  %11 = hivm.hir.pointer_cast(%c140160_i64) : memref<32xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%11 : memref<32xf32, #hivm.address_space<ub>>)
  %12 = hivm.hir.pointer_cast(%c140288_i64) : memref<64x32xf32, #hivm.address_space<ub>>
  %collapse_shape_4 = memref.collapse_shape %12 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_4 : memref<2048xf32, #hivm.address_space<ub>>)
  %13 = arith.remsi %5, %c32_i32 : i32
  %14 = arith.extsi %8 : i32 to i64
  %15 = arith.muli %8, %c2_i32 : i32
  %16 = arith.index_cast %15 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg21 to offset: [%16], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %17 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %18 = arith.trunci %17 : i64 to i32
  %19 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%16]
  %reinterpret_cast_5 = memref.reinterpret_cast %arg21 to offset: [%19], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %20 = memref.load %reinterpret_cast_5[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %21 = arith.trunci %20 : i64 to i32
  %22 = arith.index_cast %18 : i32 to index
  %reinterpret_cast_6 = memref.reinterpret_cast %arg20 to offset: [%22], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %23 = memref.load %reinterpret_cast_6[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %24 = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%22]
  %reinterpret_cast_7 = memref.reinterpret_cast %arg20 to offset: [%24], sizes: [1], strides: [1] : memref<?xi64, #hivm.address_space<gm>> to memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %25 = memref.load %reinterpret_cast_7[%c0] : memref<1xi64, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %26 = arith.subi %25, %23 : i64
  %27 = arith.trunci %26 : i64 to i32
  %28 = arith.muli %21, %c64_i32 : i32
  %29 = hivm.hir.pointer_cast(%c148480_i64) : memref<64xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%29 : memref<64xi32, #hivm.address_space<ub>>)
  %30 = hivm.hir.pointer_cast(%c148480_i64) : memref<64xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%29, %28 : memref<64xi32, #hivm.address_space<ub>>, i32) outs(%30 : memref<64xi32, #hivm.address_space<ub>>)
  %31 = hivm.hir.pointer_cast(%c148736_i64) : memref<64xi32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vmax ins(%30, %27 : memref<64xi32, #hivm.address_space<ub>>, i32) outs(%31 : memref<64xi32, #hivm.address_space<ub>>)
  %32 = hivm.hir.pointer_cast(%c148736_i64) : memref<64xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%31, %30 : memref<64xi32, #hivm.address_space<ub>>, memref<64xi32, #hivm.address_space<ub>>) outs(%32 : memref<64xi1, #hivm.address_space<ub>>)
  %33 = hivm.hir.pointer_cast(%c148736_i64) : memref<64xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%32 : memref<64xi1, #hivm.address_space<ub>>) outs(%33 : memref<64xi1, #hivm.address_space<ub>>)
  %34 = arith.addi %28, %c64_i32 : i32
  %35 = arith.minsi %27, %34 : i32
  %36 = arith.subi %35, %c1_i32 : i32
  %37 = hivm.hir.pointer_cast(%c57600_i64) : memref<64xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%36 : i32) outs(%37 : memref<64xi32, #hivm.address_space<ub>>)
  %38 = hivm.hir.pointer_cast(%c57600_i64) : memref<64xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%30, %37 : memref<64xi32, #hivm.address_space<ub>>, memref<64xi32, #hivm.address_space<ub>>) outs(%38 : memref<64xi1, #hivm.address_space<ub>>)
  %39 = arith.muli %13, %arg25 : i32
  %40 = arith.extsi %39 : i32 to i64
  %41 = arith.addi %40, %23 : i64
  %42 = arith.muli %41, %c128_i64 : i64
  %43 = arith.index_cast %42 : i64 to index
  %44 = arith.muli %23, %c32_i64 : i64
  %45 = arith.extsi %13 : i32 to i64
  %46 = arith.addi %44, %45 : i64
  %47 = arith.muli %46, %c128_i64 : i64
  %48 = arith.index_cast %47 : i64 to index
  %49 = arith.index_cast %41 : i64 to index
  %50 = arith.muli %41, %c64_i64 : i64
  %51 = arith.index_cast %50 : i64 to index
  %52 = arith.muli %14, %c32_i64 : i64
  %53 = arith.addi %52, %45 : i64
  %54 = arith.muli %53, %c16384_i64 : i64
  %55 = arith.index_cast %54 : i64 to index
  %56 = hivm.hir.pointer_cast(%c57856_i64) : memref<64xi32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%30, %c0_i32 : memref<64xi32, #hivm.address_space<ub>>, i32) outs(%56 : memref<64xi32, #hivm.address_space<ub>>)
  %57 = hivm.hir.pointer_cast(%c57856_i64) : memref<64xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%56, %30 : memref<64xi32, #hivm.address_space<ub>>, memref<64xi32, #hivm.address_space<ub>>) outs(%57 : memref<64xi1, #hivm.address_space<ub>>)
  %58 = hivm.hir.pointer_cast(%c57856_i64) : memref<64xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vand ins(%33, %57 : memref<64xi1, #hivm.address_space<ub>>, memref<64xi1, #hivm.address_space<ub>>) outs(%58 : memref<64xi1, #hivm.address_space<ub>>)
  %59 = arith.index_cast %28 : i32 to index
  %60 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%49, %59]
  %reinterpret_cast_8 = memref.reinterpret_cast %arg8 to offset: [%60], sizes: [1, 64], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
  %61 = hivm.hir.pointer_cast(%c0_i64) : memref<1x64xf32, #hivm.address_space<ub>>
  %62 = arith.index_cast %27 : i32 to index
  %63 = affine.max affine_map<()[s0, s1] -> (s1, s0)>()[%59, %62]
  %64 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%63, %59]
  %65 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%64, %59]
  %66 = affine.max affine_map<()[s0] -> (0, s0)>()[%59]
  %67 = affine.min affine_map<()[s0, s1] -> (s1 + 64, s0)>()[%66, %59]
  %68 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%67, %59]
  %69 = affine.min affine_map<()[s0, s1] -> (64, s0 - s1)>()[%64, %59]
  %70 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%69, %68]
  %71 = arith.cmpi slt, %70, %c64 : index
  %subview = memref.subview %reinterpret_cast_8[0, %68] [1, %70] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<1x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
  %subview_9 = memref.subview %61[0, %68] [1, %70] [1, 1] : memref<1x64xf32, #hivm.address_space<ub>> to memref<1x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
  scf.if %71 {
    %collapse_shape_52 = memref.collapse_shape %61 [[0, 1]] : memref<1x64xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_52 : memref<64xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  %collapse_shape_10 = memref.collapse_shape %subview [[0, 1]] : memref<1x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %collapse_shape_11 = memref.collapse_shape %subview_9 [[0, 1]] : memref<1x?xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  hivm.hir.load ins(%collapse_shape_10 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%collapse_shape_11 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %68 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
  %72 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%51, %59]
  %reinterpret_cast_12 = memref.reinterpret_cast %arg9 to offset: [%72], sizes: [64, 64], strides: [1, 64] : memref<?xbf16, #hivm.address_space<gm>> to memref<64x64xbf16, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
  %73 = hivm.hir.pointer_cast(%c256_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
  %subview_13 = memref.subview %reinterpret_cast_12[0, %68] [64, %70] [1, 1] : memref<64x64xbf16, strided<[1, 64], offset: ?>, #hivm.address_space<gm>> to memref<64x?xbf16, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>
  %subview_14 = memref.subview %73[0, %68] [64, %70] [1, 1] : memref<64x64xbf16, #hivm.address_space<ub>> to memref<64x?xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
  scf.if %71 {
    %collapse_shape_52 = memref.collapse_shape %73 [[0, 1]] : memref<64x64xbf16, #hivm.address_space<ub>> into memref<4096xbf16, #hivm.address_space<ub>>
    hivm.hir.vbrc ins(%cst_0 : bf16) outs(%collapse_shape_52 : memref<4096xbf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  } {hivm.unlikely_condition}
  hivm.hir.load ins(%subview_13 : memref<64x?xbf16, strided<[1, 64], offset: ?>, #hivm.address_space<gm>>) outs(%subview_14 : memref<64x?xbf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : bf16 left_padding_num = %68 : index init_out_buffer = false may_implicit_transpose_with_last_axis = true
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  annotation.mark %73 {MayImplicitTransposeWithLastAxis} : memref<64x64xbf16, #hivm.address_space<ub>>
  %74 = memref_ext.alloc_workspace() from %arg2 offset = [%c0] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xbf16, #hivm.address_space<gm>>
  %75 = hivm.hir.get_sub_block_idx -> i64
  %76 = arith.index_cast %75 : i64 to index
  %77 = arith.cmpi eq, %76, %c0 : index
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
  scf.if %77 {
    %collapse_shape_52 = memref.collapse_shape %73 [[0, 1]] : memref<64x64xbf16, #hivm.address_space<ub>> into memref<4096xbf16, #hivm.address_space<ub>>
    %collapse_shape_53 = memref.collapse_shape %74 [[0, 1]] : memref<64x64xbf16, #hivm.address_space<gm>> into memref<4096xbf16, #hivm.address_space<gm>>
    hivm.hir.store ins(%collapse_shape_52 : memref<4096xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_53 : memref<4096xbf16, #hivm.address_space<gm>>)
  }
  annotation.mark %74 : memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %78 = memref_ext.alloc_workspace() from %arg2 offset = [%c8192] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xbf16, #hivm.address_space<gm>>
  scf.if %77 {
    %collapse_shape_52 = memref.collapse_shape %73 [[0, 1]] : memref<64x64xbf16, #hivm.address_space<ub>> into memref<4096xbf16, #hivm.address_space<ub>>
    %collapse_shape_53 = memref.collapse_shape %78 [[0, 1]] : memref<64x64xbf16, #hivm.address_space<gm>> into memref<4096xbf16, #hivm.address_space<gm>>
    hivm.hir.store ins(%collapse_shape_52 : memref<4096xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_53 : memref<4096xbf16, #hivm.address_space<gm>>)
  }
  annotation.mark %78 : memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %79 = memref_ext.alloc_workspace() from %arg2 offset = [%c16384] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xbf16, #hivm.address_space<gm>>
  scf.if %77 {
    %collapse_shape_52 = memref.collapse_shape %73 [[0, 1]] : memref<64x64xbf16, #hivm.address_space<ub>> into memref<4096xbf16, #hivm.address_space<ub>>
    %collapse_shape_53 = memref.collapse_shape %79 [[0, 1]] : memref<64x64xbf16, #hivm.address_space<gm>> into memref<4096xbf16, #hivm.address_space<gm>>
    hivm.hir.store ins(%collapse_shape_52 : memref<4096xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_53 : memref<4096xbf16, #hivm.address_space<gm>>)
  }
  annotation.mark %79 : memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %80 = memref_ext.alloc_workspace() from %arg2 offset = [%c24576] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xbf16, #hivm.address_space<gm>>
  scf.if %77 {
    %collapse_shape_52 = memref.collapse_shape %73 [[0, 1]] : memref<64x64xbf16, #hivm.address_space<ub>> into memref<4096xbf16, #hivm.address_space<ub>>
    %collapse_shape_53 = memref.collapse_shape %80 [[0, 1]] : memref<64x64xbf16, #hivm.address_space<gm>> into memref<4096xbf16, #hivm.address_space<gm>>
    hivm.hir.store ins(%collapse_shape_52 : memref<4096xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_53 : memref<4096xbf16, #hivm.address_space<gm>>)
  }
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  annotation.mark %80 : memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  annotation.mark %73 {MayImplicitTransposeWithLastAxis} : memref<64x64xbf16, #hivm.address_space<ub>>
  %81 = hivm.hir.pointer_cast(%c148992_i64) : memref<32xi32, #hivm.address_space<ub>>
  hivm.hir.varange offset[%c0] strides[%c1] outs(%81 : memref<32xi32, #hivm.address_space<ub>>)
  %82 = arith.extsi %36 : i32 to i64
  %83 = arith.muli %82, %c4096_i64 : i64
  %84 = arith.addi %47, %83 : i64
  %85 = arith.index_cast %84 : i64 to index
  %86 = hivm.hir.pointer_cast(%c58112_i64) : memref<64xf16, #hivm.address_space<ub>>
  %87 = hivm.hir.pointer_cast(%c58240_i64) : memref<48xf16, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%58 : memref<64xi1, #hivm.address_space<ub>>) outs(%86 : memref<64xf16, #hivm.address_space<ub>>) temp_buffer(%87 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
  %expand_shape = memref.expand_shape %86 [[0, 1]] output_shape [64, 1] : memref<64xf16, #hivm.address_space<ub>> into memref<64x1xf16, #hivm.address_space<ub>>
  %88 = hivm.hir.pointer_cast(%c149120_i64) : memref<64x32xf16, #hivm.address_space<ub>>
  %89 = hivm.hir.pointer_cast(%c58336_i64) : memref<1024xf16, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape : memref<64x1xf16, #hivm.address_space<ub>>) outs(%88 : memref<64x32xf16, #hivm.address_space<ub>>) temp_buffer(%89 : memref<1024xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
  %90 = hivm.hir.pointer_cast(%c149120_i64) : memref<64x32xi1, #hivm.address_space<ub>>
  %collapse_shape_15 = memref.collapse_shape %88 [[0, 1]] : memref<64x32xf16, #hivm.address_space<ub>> into memref<2048xf16, #hivm.address_space<ub>>
  %collapse_shape_16 = memref.collapse_shape %90 [[0, 1]] : memref<64x32xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_15, %cst_1 : memref<2048xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_16 : memref<2048xi1, #hivm.address_space<ub>>)
  %91 = hivm.hir.pointer_cast(%c149120_i64) : memref<64x32xi1, #hivm.address_space<ub>>
  %collapse_shape_17 = memref.collapse_shape %91 [[0, 1]] : memref<64x32xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_16 : memref<2048xi1, #hivm.address_space<ub>>) outs(%collapse_shape_17 : memref<2048xi1, #hivm.address_space<ub>>)
  %expand_shape_18 = memref.expand_shape %61 [[0], [1, 2]] output_shape [1, 64, 1] : memref<1x64xf32, #hivm.address_space<ub>> into memref<1x64x1xf32, #hivm.address_space<ub>>
  %collapse_shape_19 = memref.collapse_shape %expand_shape_18 [[0, 1], [2]] : memref<1x64x1xf32, #hivm.address_space<ub>> into memref<64x1xf32, #hivm.address_space<ub>>
  %92 = hivm.hir.pointer_cast(%c153216_i64) : memref<64x1xf32, #hivm.address_space<ub>>
  %collapse_shape_20 = memref.collapse_shape %92 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
  %93 = hivm.hir.pointer_cast(%c60384_i64) : memref<24xf32, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%38 : memref<64xi1, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<64xf32, #hivm.address_space<ub>>) temp_buffer(%93 : memref<24xf32, #hivm.address_space<ub>>) cast = <cast_unsigned>
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  %94 = arith.maxsi %28, %c0_i32 : i32
  %95 = arith.index_cast %94 : i32 to index
  %96 = arith.subi %c0_i32, %28 : i32
  %97 = arith.maxsi %96, %c0_i32 : i32
  %98 = arith.index_cast %97 : i32 to index
  %99 = affine.min affine_map<()[s0] -> (64, s0)>()[%68]
  %100 = affine.min affine_map<()[s0, s1, s2] -> (64, s0 + s1 - s2)>()[%99, %69, %68]
  %101 = hivm.hir.pointer_cast(%c178304_i64) : memref<64x64xf32, #hivm.address_space<ub>>
  %collapse_shape_21 = memref.collapse_shape %101 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.copy ins(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_21 : memref<4096xf32, #hivm.address_space<ub>>)
  %102 = hivm.hir.pointer_cast(%c123264_i64) : memref<64xf32, #hivm.address_space<ub>>
  hivm.hir.copy ins(%9 : memref<64xf32, #hivm.address_space<ub>>) outs(%102 : memref<64xf32, #hivm.address_space<ub>>)
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  %103:2 = scf.for %arg29 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg30 = %101, %arg31 = %102) -> (memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>)  : i32 {
    %137 = arith.muli %arg29, %c32_i32 : i32
    %138 = arith.maxsi %137, %c0_i32 : i32
    %139 = arith.index_cast %138 : i32 to index
    %140 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 4096)>()[%139, %48, %95]
    %reinterpret_cast_52 = memref.reinterpret_cast %arg4 to offset: [%140], sizes: [64, 32], strides: [4096, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64x32xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_53 = memref.reinterpret_cast %arg7 to offset: [%140], sizes: [64, 32], strides: [4096, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x32xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
    %141 = hivm.hir.pointer_cast(%c8448_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
    %142 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 4096, 0)>()[%62, %139, %95]
    %143 = affine.min affine_map<()[s0] -> (64, s0)>()[%142]
    %144 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 4096) * 4096 + 128, 0)>()[%139]
    %145 = affine.min affine_map<()[s0] -> (32, s0)>()[%144]
    %146 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%98, %142]
    %147 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%143, %146]
    %148 = arith.subi %c0_i32, %137 : i32
    %149 = arith.maxsi %148, %c0_i32 : i32
    %150 = arith.index_cast %149 : i32 to index
    %151 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%150, %144]
    %152 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%145, %151]
    %153 = arith.cmpi slt, %147, %c64 : index
    %154 = arith.cmpi slt, %152, %c32 : index
    %155 = arith.ori %153, %154 : i1
    %subview_54 = memref.subview %reinterpret_cast_52[0, 0] [%147, %152] [1, 1] : memref<64x32xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_55 = memref.subview %141[%146, %151] [%147, %152] [1, 1] : memref<64x32xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %155 {
      %collapse_shape_127 = memref.collapse_shape %141 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : bf16) outs(%collapse_shape_127 : memref<2048xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_54 : memref<?x?xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_55 : memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : bf16 left_padding_num = %151 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %156 = hivm.hir.pointer_cast(%c12544_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %subview_56 = memref.subview %reinterpret_cast_53[0, 0] [%147, %152] [1, 1] : memref<64x32xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_57 = memref.subview %156[%146, %151] [%147, %152] [1, 1] : memref<64x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %155 {
      %collapse_shape_127 = memref.collapse_shape %156 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst : f32) outs(%collapse_shape_127 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_56 : memref<?x?xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_57 : memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %151 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    %157 = arith.index_cast %137 : i32 to index
    %158 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%85, %157]
    %reinterpret_cast_58 = memref.reinterpret_cast %arg7 to offset: [%158], sizes: [32], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %159 = hivm.hir.pointer_cast(%c20736_i64) : memref<32xf32, #hivm.address_space<ub>>
    %160 = affine.max affine_map<()[s0] -> (128, s0)>()[%157]
    %161 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%160, %157]
    %162 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%161, %157]
    %163 = arith.cmpi slt, %162, %c32 : index
    %subview_59 = memref.subview %reinterpret_cast_58[0] [%162] [1] : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    %subview_60 = memref.subview %159[0] [%162] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    scf.if %163 {
      hivm.hir.vbrc ins(%cst : f32) outs(%159 : memref<32xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_59 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_60 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst : f32 left_padding_num = %c0 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %164 = hivm.hir.pointer_cast(%c153472_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%11 : memref<32xf32, #hivm.address_space<ub>>) outs(%164 : memref<32xf32, #hivm.address_space<ub>>)
    %subview_61 = memref.subview %164[0] [%162] [1] : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%subview_60 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%subview_61 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
    %165 = arith.cmpi eq, %arg29, %c0_i32 : i32
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 10
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 11
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 12
    %166 = arith.extsi %137 : i32 to i64
    %167 = arith.muli %166, %c128_i64 : i64
    %168 = hivm.hir.pointer_cast(%c153600_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%11 : memref<32xf32, #hivm.address_space<ub>>) outs(%168 : memref<32xf32, #hivm.address_space<ub>>)
    %169 = hivm.hir.pointer_cast(%c153728_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_62 = memref.collapse_shape %169 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_4 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_62 : memref<2048xf32, #hivm.address_space<ub>>)
    %170 = hivm.hir.pointer_cast(%c161920_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_63 = memref.collapse_shape %170 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_4 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_63 : memref<2048xf32, #hivm.address_space<ub>>)
    %171 = hivm.hir.pointer_cast(%c170112_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_64 = memref.collapse_shape %171 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_4 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_64 : memref<2048xf32, #hivm.address_space<ub>>)
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    %172:6 = scf.for %arg32 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg33 = %168, %arg34 = %169, %arg35 = %170, %arg36 = %171, %arg37 = %arg30, %arg38 = %arg31) -> (memref<32xf32, #hivm.address_space<ub>>, memref<64x32xf32, #hivm.address_space<ub>>, memref<64x32xf32, #hivm.address_space<ub>>, memref<64x32xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>)  : i32 {
      %235 = arith.muli %arg32, %c32_i32 : i32
      %236 = hivm.hir.pointer_cast(%c256_i64) : memref<32xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%81, %235 : memref<32xi32, #hivm.address_space<ub>>, i32) outs(%236 : memref<32xi32, #hivm.address_space<ub>>)
      %237 = hivm.hir.pointer_cast(%c384_i64) : memref<32xi32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmax ins(%236, %c128_i32 : memref<32xi32, #hivm.address_space<ub>>, i32) outs(%237 : memref<32xi32, #hivm.address_space<ub>>)
      %238 = hivm.hir.pointer_cast(%c384_i64) : memref<32xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%237, %236 : memref<32xi32, #hivm.address_space<ub>>, memref<32xi32, #hivm.address_space<ub>>) outs(%238 : memref<32xi1, #hivm.address_space<ub>>)
      %239 = hivm.hir.pointer_cast(%c384_i64) : memref<32xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%238 : memref<32xi1, #hivm.address_space<ub>>) outs(%239 : memref<32xi1, #hivm.address_space<ub>>)
      %240 = hivm.hir.pointer_cast(%c512_i64) : memref<32xi32, #hivm.address_space<ub>>
      hivm.hir.vmax ins(%236, %c0_i32 : memref<32xi32, #hivm.address_space<ub>>, i32) outs(%240 : memref<32xi32, #hivm.address_space<ub>>)
      %241 = hivm.hir.pointer_cast(%c512_i64) : memref<32xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%240, %236 : memref<32xi32, #hivm.address_space<ub>>, memref<32xi32, #hivm.address_space<ub>>) outs(%241 : memref<32xi1, #hivm.address_space<ub>>)
      %242 = hivm.hir.pointer_cast(%c384_i64) : memref<32xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vand ins(%239, %241 : memref<32xi1, #hivm.address_space<ub>>, memref<32xi1, #hivm.address_space<ub>>) outs(%242 : memref<32xi1, #hivm.address_space<ub>>)
      %243 = arith.index_cast %235 : i32 to index
      %244 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>()[%243, %43, %59]
      %reinterpret_cast_127 = memref.reinterpret_cast %arg6 to offset: [%244], sizes: [64, 32], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64x32xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %reinterpret_cast_128 = memref.reinterpret_cast %arg11 to offset: [%244], sizes: [64, 32], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64x32xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %245 = hivm.hir.pointer_cast(%c640_i64) : memref<32xf16, #hivm.address_space<ub>>
      %246 = hivm.hir.pointer_cast(%c704_i64) : memref<48xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcast ins(%242 : memref<32xi1, #hivm.address_space<ub>>) outs(%245 : memref<32xf16, #hivm.address_space<ub>>) temp_buffer(%246 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
      %expand_shape_129 = memref.expand_shape %245 [[0, 1]] output_shape [1, 32] : memref<32xf16, #hivm.address_space<ub>> into memref<1x32xf16, #hivm.address_space<ub>>
      %247 = hivm.hir.pointer_cast(%c800_i64) : memref<64x32xf16, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vbrc ins(%expand_shape_129 : memref<1x32xf16, #hivm.address_space<ub>>) outs(%247 : memref<64x32xf16, #hivm.address_space<ub>>) broadcast_dims = [0]
      %248 = hivm.hir.pointer_cast(%c800_i64) : memref<64x32xi1, #hivm.address_space<ub>>
      %collapse_shape_130 = memref.collapse_shape %247 [[0, 1]] : memref<64x32xf16, #hivm.address_space<ub>> into memref<2048xf16, #hivm.address_space<ub>>
      %collapse_shape_131 = memref.collapse_shape %248 [[0, 1]] : memref<64x32xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vcmp ins(%collapse_shape_130, %cst_1 : memref<2048xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_131 : memref<2048xi1, #hivm.address_space<ub>>)
      %249 = hivm.hir.pointer_cast(%c800_i64) : memref<64x32xi1, #hivm.address_space<ub>>
      %collapse_shape_132 = memref.collapse_shape %249 [[0, 1]] : memref<64x32xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vnot ins(%collapse_shape_131 : memref<2048xi1, #hivm.address_space<ub>>) outs(%collapse_shape_132 : memref<2048xi1, #hivm.address_space<ub>>)
      %250 = hivm.hir.pointer_cast(%c800_i64) : memref<64x32xi1, #hivm.address_space<ub>>
      %collapse_shape_133 = memref.collapse_shape %250 [[0, 1]] : memref<64x32xi1, #hivm.address_space<ub>> into memref<2048xi1, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vand ins(%collapse_shape_17, %collapse_shape_132 : memref<2048xi1, #hivm.address_space<ub>>, memref<2048xi1, #hivm.address_space<ub>>) outs(%collapse_shape_133 : memref<2048xi1, #hivm.address_space<ub>>)
      %251 = hivm.hir.pointer_cast(%c20864_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
      %252 = affine.max affine_map<()[s0] -> (128, s0)>()[%243]
      %253 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%252, %243]
      %254 = affine.max affine_map<()[s0] -> (0, s0)>()[%243]
      %255 = affine.min affine_map<()[s0, s1] -> (s1 + 32, s0)>()[%254, %243]
      %256 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%255, %243]
      %257 = affine.min affine_map<()[s0, s1] -> (32, s0 - s1)>()[%253, %243]
      %258 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%257, %256]
      %259 = arith.cmpi slt, %258, %c32 : index
      %260 = arith.ori %71, %259 : i1
      %subview_134 = memref.subview %reinterpret_cast_127[%68, %256] [%70, %258] [1, 1] : memref<64x32xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_135 = memref.subview %251[%68, %256] [%70, %258] [1, 1] : memref<64x32xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %260 {
        %collapse_shape_168 = memref.collapse_shape %251 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst_0 : bf16) outs(%collapse_shape_168 : memref<2048xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      hivm.hir.load ins(%subview_134 : memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_135 : memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : bf16 left_padding_num = %256 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %261 = hivm.hir.pointer_cast(%c24960_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
      %collapse_shape_136 = memref.collapse_shape %251 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
      %collapse_shape_137 = memref.collapse_shape %261 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
      %262 = hivm.hir.pointer_cast(%c4896_i64) : memref<32xbf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vsel ins(%collapse_shape_133, %collapse_shape_136, %cst_0 : memref<2048xi1, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>, bf16) outs(%collapse_shape_137 : memref<2048xbf16, #hivm.address_space<ub>>) temp_buffer(%262 : memref<32xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
      %263 = memref_ext.alloc_workspace() from %arg2 offset = [%c32768] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %77 {
        %collapse_shape_168 = memref.collapse_shape %263 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<gm>> into memref<2048xbf16, #hivm.address_space<gm>>
        hivm.hir.store ins(%collapse_shape_137 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_168 : memref<2048xbf16, #hivm.address_space<gm>>)
      }
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
      annotation.mark %263 : memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %264 = hivm.hir.pointer_cast(%c29056_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
      %subview_138 = memref.subview %reinterpret_cast_128[%68, %256] [%70, %258] [1, 1] : memref<64x32xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
      %subview_139 = memref.subview %264[%68, %256] [%70, %258] [1, 1] : memref<64x32xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
      scf.if %260 {
        %collapse_shape_168 = memref.collapse_shape %264 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
        hivm.hir.vbrc ins(%cst_0 : bf16) outs(%collapse_shape_168 : memref<2048xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      } {hivm.unlikely_condition}
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      hivm.hir.load ins(%subview_138 : memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_139 : memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : bf16 left_padding_num = %256 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %265 = hivm.hir.pointer_cast(%c33152_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
      %collapse_shape_140 = memref.collapse_shape %264 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
      %collapse_shape_141 = memref.collapse_shape %265 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
      %266 = hivm.hir.pointer_cast(%c4960_i64) : memref<32xbf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vsel ins(%collapse_shape_133, %collapse_shape_140, %cst_0 : memref<2048xi1, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>, bf16) outs(%collapse_shape_141 : memref<2048xbf16, #hivm.address_space<ub>>) temp_buffer(%266 : memref<32xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
      %267 = memref_ext.alloc_workspace() from %arg2 offset = [%c36864] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %77 {
        %collapse_shape_168 = memref.collapse_shape %267 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<gm>> into memref<2048xbf16, #hivm.address_space<gm>>
        hivm.hir.store ins(%collapse_shape_141 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_168 : memref<2048xbf16, #hivm.address_space<gm>>)
      }
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
      annotation.mark %267 : memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %268 = arith.extsi %235 : i32 to i64
      %269 = hivm.hir.pointer_cast(%c37248_i64) : memref<32x32xbf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
      scf.for %arg39 = %c0 to %c32 step %c1 {
        %303 = arith.index_cast %arg39 : index to i64
        %304 = arith.addi %268, %303 : i64
        %305 = arith.addi %304, %167 : i64
        scf.for %arg40 = %c0 to %c32 step %c1 {
          %306 = arith.index_cast %arg40 : index to i64
          %307 = arith.muli %306, %c128_i64 : i64
          %308 = arith.addi %305, %307 : i64
          %309 = arith.index_cast %308 : i64 to index
          %310 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%55, %309]
          %reinterpret_cast_168 = memref.reinterpret_cast %arg10 to offset: [%310], sizes: [1], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %311 = memref.load %reinterpret_cast_168[%c0] : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          memref.store %311, %269[%arg39, %arg40] : memref<32x32xbf16, #hivm.address_space<ub>>
        } {ExtractedLoadOrStore}
      } {ExtractedLoadOrStore}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %270 = memref_ext.alloc_workspace() from %arg2 offset = [%c40960] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %77 {
        %collapse_shape_168 = memref.collapse_shape %269 [[0, 1]] : memref<32x32xbf16, #hivm.address_space<ub>> into memref<1024xbf16, #hivm.address_space<ub>>
        %collapse_shape_169 = memref.collapse_shape %270 [[0, 1]] : memref<32x32xbf16, #hivm.address_space<gm>> into memref<1024xbf16, #hivm.address_space<gm>>
        hivm.hir.store ins(%collapse_shape_168 : memref<1024xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_169 : memref<1024xbf16, #hivm.address_space<gm>>)
      }
      annotation.mark %270 : memref<32x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %271 = memref_ext.alloc_workspace() from %arg2 offset = [%c43008] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
      scf.if %77 {
        %collapse_shape_168 = memref.collapse_shape %269 [[0, 1]] : memref<32x32xbf16, #hivm.address_space<ub>> into memref<1024xbf16, #hivm.address_space<ub>>
        %collapse_shape_169 = memref.collapse_shape %271 [[0, 1]] : memref<32x32xbf16, #hivm.address_space<gm>> into memref<1024xbf16, #hivm.address_space<gm>>
        hivm.hir.store ins(%collapse_shape_168 : memref<1024xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_169 : memref<1024xbf16, #hivm.address_space<gm>>)
      }
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
      annotation.mark %271 : memref<32x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %272 = hivm.hir.pointer_cast(%c39296_i64) : memref<32x32xbf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      scf.for %arg39 = %c0 to %c32 step %c1 {
        %303 = arith.index_cast %arg39 : index to i64
        %304 = arith.addi %268, %303 : i64
        %305 = arith.addi %304, %167 : i64
        scf.for %arg40 = %c0 to %c32 step %c1 {
          %306 = arith.index_cast %arg40 : index to i64
          %307 = arith.muli %306, %c128_i64 : i64
          %308 = arith.addi %305, %307 : i64
          %309 = arith.index_cast %308 : i64 to index
          %310 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%55, %309]
          %reinterpret_cast_168 = memref.reinterpret_cast %arg12 to offset: [%310], sizes: [1], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %311 = memref.load %reinterpret_cast_168[%c0] : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          memref.store %311, %272[%arg39, %arg40] : memref<32x32xbf16, #hivm.address_space<ub>>
        } {ExtractedLoadOrStore}
      } {ExtractedLoadOrStore}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      %273 = memref_ext.alloc_workspace() from %arg2 offset = [%c45056] : from memref<?xi8, #hivm.address_space<gm>> to memref<32x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %77 {
        %collapse_shape_168 = memref.collapse_shape %272 [[0, 1]] : memref<32x32xbf16, #hivm.address_space<ub>> into memref<1024xbf16, #hivm.address_space<ub>>
        %collapse_shape_169 = memref.collapse_shape %273 [[0, 1]] : memref<32x32xbf16, #hivm.address_space<gm>> into memref<1024xbf16, #hivm.address_space<gm>>
        hivm.hir.store ins(%collapse_shape_168 : memref<1024xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_169 : memref<1024xbf16, #hivm.address_space<gm>>)
      }
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
      annotation.mark %273 : memref<32x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %274 = affine.min affine_map<()[s0] -> (32, s0)>()[%256]
      %275 = affine.min affine_map<()[s0, s1, s2] -> (32, s0 + s1 - s2)>()[%274, %257, %256]
      %276 = hivm.hir.pointer_cast(%c119168_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
      scf.for %arg39 = %99 to %100 step %c1 {
        %303 = arith.index_cast %arg39 : index to i32
        %304 = arith.addi %28, %303 : i32
        %305 = arith.muli %304, %c4096_i32 : i32
        %306 = arith.extsi %305 : i32 to i64
        %307 = arith.addi %47, %306 : i64
        scf.for %arg40 = %274 to %275 step %c1 {
          %308 = arith.index_cast %arg40 : index to i32
          %309 = arith.addi %235, %308 : i32
          %310 = arith.extsi %309 : i32 to i64
          %311 = arith.addi %307, %310 : i64
          %312 = arith.index_cast %311 : i64 to index
          %reinterpret_cast_168 = memref.reinterpret_cast %arg15 to offset: [%312], sizes: [1], strides: [1] : memref<?xbf16, #hivm.address_space<gm>> to memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          %313 = memref.load %reinterpret_cast_168[%c0] : memref<1xbf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
          memref.store %313, %276[%arg39, %arg40] : memref<64x32xbf16, #hivm.address_space<ub>>
        } {ExtractedLoadOrStore}
      } {ExtractedLoadOrStore}
      hivm.hir.set_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      %277 = hivm.hir.pointer_cast(%c41344_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
      %collapse_shape_142 = memref.collapse_shape %276 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
      %collapse_shape_143 = memref.collapse_shape %277 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
      %278 = hivm.hir.pointer_cast(%c5024_i64) : memref<32xbf16, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
      hivm.hir.wait_flag[<PIPE_S>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vsel ins(%collapse_shape_133, %collapse_shape_142, %cst_0 : memref<2048xi1, #hivm.address_space<ub>>, memref<2048xbf16, #hivm.address_space<ub>>, bf16) outs(%collapse_shape_143 : memref<2048xbf16, #hivm.address_space<ub>>) temp_buffer(%278 : memref<32xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      %279 = memref_ext.alloc_workspace() from %arg2 offset = [%c47104] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      scf.if %77 {
        %collapse_shape_168 = memref.collapse_shape %279 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<gm>> into memref<2048xbf16, #hivm.address_space<gm>>
        hivm.hir.store ins(%collapse_shape_143 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_168 : memref<2048xbf16, #hivm.address_space<gm>>)
      }
      annotation.mark %279 : memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %280 = memref_ext.alloc_workspace() from %arg2 offset = [%c51200] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 8
      scf.if %77 {
        %collapse_shape_168 = memref.collapse_shape %280 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<gm>> into memref<2048xbf16, #hivm.address_space<gm>>
        hivm.hir.store ins(%collapse_shape_143 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_168 : memref<2048xbf16, #hivm.address_space<gm>>)
      }
      annotation.mark %280 : memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %281 = memref_ext.alloc_workspace() from %arg2 offset = [%c55296] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 9
      scf.if %77 {
        %collapse_shape_168 = memref.collapse_shape %281 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<gm>> into memref<2048xbf16, #hivm.address_space<gm>>
        hivm.hir.store ins(%collapse_shape_143 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_168 : memref<2048xbf16, #hivm.address_space<gm>>)
      }
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
      annotation.mark %281 : memref<64x32xbf16, #hivm.address_space<gm>>
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
      %282 = hivm.hir.pointer_cast(%c106880_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %collapse_shape_144 = memref.collapse_shape %269 [[0, 1]] : memref<32x32xbf16, #hivm.address_space<ub>> into memref<1024xbf16, #hivm.address_space<ub>>
      %collapse_shape_145 = memref.collapse_shape %282 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%collapse_shape_144 : memref<1024xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_145 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
      %283 = hivm.hir.pointer_cast(%c110976_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %collapse_shape_146 = memref.collapse_shape %272 [[0, 1]] : memref<32x32xbf16, #hivm.address_space<ub>> into memref<1024xbf16, #hivm.address_space<ub>>
      %collapse_shape_147 = memref.collapse_shape %283 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.vcast ins(%collapse_shape_146 : memref<1024xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_147 : memref<1024xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
      %284 = hivm.hir.pointer_cast(%c110976_i64) : memref<32x32xf32, #hivm.address_space<ub>>
      %collapse_shape_148 = memref.collapse_shape %284 [[0, 1]] : memref<32x32xf32, #hivm.address_space<ub>> into memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vmul ins(%collapse_shape_145, %collapse_shape_147 : memref<1024xf32, #hivm.address_space<ub>>, memref<1024xf32, #hivm.address_space<ub>>) outs(%collapse_shape_148 : memref<1024xf32, #hivm.address_space<ub>>)
      %285 = hivm.hir.pointer_cast(%c5088_i64) : memref<1x32xf32, #hivm.address_space<ub>>
      %286 = hivm.hir.pointer_cast(%c115072_i64) : memref<1024xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vreduce <sum> ins(%284 : memref<32x32xf32, #hivm.address_space<ub>>) outs(%285 : memref<1x32xf32, #hivm.address_space<ub>>) temp_buffer(%286 : memref<1024xf32, #hivm.address_space<ub>>) reduce_dims = [0]
      %collapse_shape_149 = memref.collapse_shape %285 [[0, 1]] : memref<1x32xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
      %287 = hivm.hir.pointer_cast(%c153600_i64) : memref<32xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%arg33, %collapse_shape_149 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%287 : memref<32xf32, #hivm.address_space<ub>>)
      %288 = memref_ext.alloc_workspace() from %arg2 offset = [%c59392] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
      %289 = hivm.hir.pointer_cast(%c45440_i64) : memref<64x32xf32, #hivm.address_space<ub>>
      %collapse_shape_150 = memref.collapse_shape %288 [[0, 1]] : memref<64x32xf32, #hivm.address_space<gm>> into memref<2048xf32, #hivm.address_space<gm>>
      %collapse_shape_151 = memref.collapse_shape %289 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      hivm.hir.load ins(%collapse_shape_150 : memref<2048xf32, #hivm.address_space<gm>>) outs(%collapse_shape_151 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 10
      %290 = hivm.hir.pointer_cast(%c45440_i64) : memref<64x32xf32, #hivm.address_space<ub>>
      %collapse_shape_152 = memref.collapse_shape %290 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_151, %arg24 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_152 : memref<2048xf32, #hivm.address_space<ub>>)
      %291 = hivm.hir.pointer_cast(%c153728_i64) : memref<64x32xf32, #hivm.address_space<ub>>
      %collapse_shape_153 = memref.collapse_shape %arg34 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      %collapse_shape_154 = memref.collapse_shape %291 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_153, %collapse_shape_152 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_154 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
      %292 = memref_ext.alloc_workspace() from %arg2 offset = [%c67584] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
      %293 = hivm.hir.pointer_cast(%c53632_i64) : memref<64x32xf32, #hivm.address_space<ub>>
      %collapse_shape_155 = memref.collapse_shape %292 [[0, 1]] : memref<64x32xf32, #hivm.address_space<gm>> into memref<2048xf32, #hivm.address_space<gm>>
      %collapse_shape_156 = memref.collapse_shape %293 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
      hivm.hir.load ins(%collapse_shape_155 : memref<2048xf32, #hivm.address_space<gm>>) outs(%collapse_shape_156 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 11
      %294 = hivm.hir.pointer_cast(%c53632_i64) : memref<64x32xf32, #hivm.address_space<ub>>
      %collapse_shape_157 = memref.collapse_shape %294 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_156, %arg24 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_157 : memref<2048xf32, #hivm.address_space<ub>>)
      %295 = hivm.hir.pointer_cast(%c161920_i64) : memref<64x32xf32, #hivm.address_space<ub>>
      %collapse_shape_158 = memref.collapse_shape %arg35 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      %collapse_shape_159 = memref.collapse_shape %295 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_158, %collapse_shape_157 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_159 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
      %296 = memref_ext.alloc_workspace() from %arg2 offset = [%c75776] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xf32, #hivm.address_space<gm>>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
      %297 = hivm.hir.pointer_cast(%c61824_i64) : memref<64x32xf32, #hivm.address_space<ub>>
      %collapse_shape_160 = memref.collapse_shape %296 [[0, 1]] : memref<64x32xf32, #hivm.address_space<gm>> into memref<2048xf32, #hivm.address_space<gm>>
      %collapse_shape_161 = memref.collapse_shape %297 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
      hivm.hir.load ins(%collapse_shape_160 : memref<2048xf32, #hivm.address_space<gm>>) outs(%collapse_shape_161 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 12
      %298 = hivm.hir.pointer_cast(%c61824_i64) : memref<64x32xf32, #hivm.address_space<ub>>
      %collapse_shape_162 = memref.collapse_shape %298 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.vmul ins(%collapse_shape_161, %arg24 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_162 : memref<2048xf32, #hivm.address_space<ub>>)
      %299 = hivm.hir.pointer_cast(%c170112_i64) : memref<64x32xf32, #hivm.address_space<ub>>
      %collapse_shape_163 = memref.collapse_shape %arg36 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      %collapse_shape_164 = memref.collapse_shape %299 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%collapse_shape_163, %collapse_shape_162 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_164 : memref<2048xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
      hivm.hir.pipe_barrier[<PIPE_ALL>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
      %300:2 = scf.if %165 -> (memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) {
        %303 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 4096)>()[%243, %48, %59]
        %reinterpret_cast_168 = memref.reinterpret_cast %arg5 to offset: [%303], sizes: [64, 32], strides: [4096, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64x32xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
        %304 = arith.maxsi %235, %c0_i32 : i32
        %305 = arith.index_cast %304 : i32 to index
        %306 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 4096)>()[%305, %48, %95]
        %reinterpret_cast_169 = memref.reinterpret_cast %arg16 to offset: [%306], sizes: [64, 32], strides: [4096, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64x32xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
        %307 = hivm.hir.pointer_cast(%c70016_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
        %subview_170 = memref.subview %reinterpret_cast_168[%68, %256] [%70, %258] [1, 1] : memref<64x32xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
        %subview_171 = memref.subview %307[%68, %256] [%70, %258] [1, 1] : memref<64x32xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
        scf.if %260 {
          %collapse_shape_185 = memref.collapse_shape %307 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
          hivm.hir.vbrc ins(%cst_0 : bf16) outs(%collapse_shape_185 : memref<2048xbf16, #hivm.address_space<ub>>)
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
        } {hivm.unlikely_condition}
        hivm.hir.load ins(%subview_170 : memref<?x?xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_171 : memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : bf16 left_padding_num = %256 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
        %308 = memref_ext.alloc_workspace() from %arg2 offset = [%c83968] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xf32, #hivm.address_space<gm>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
        %309 = hivm.hir.pointer_cast(%c74112_i64) : memref<64x64xf32, #hivm.address_space<ub>>
        %collapse_shape_172 = memref.collapse_shape %308 [[0, 1]] : memref<64x64xf32, #hivm.address_space<gm>> into memref<4096xf32, #hivm.address_space<gm>>
        %collapse_shape_173 = memref.collapse_shape %309 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        hivm.hir.load ins(%collapse_shape_172 : memref<4096xf32, #hivm.address_space<gm>>) outs(%collapse_shape_173 : memref<4096xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %310 = hivm.hir.pointer_cast(%c74112_i64) : memref<64x64xf32, #hivm.address_space<ub>>
        %collapse_shape_174 = memref.collapse_shape %310 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%collapse_shape_173, %arg24 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_174 : memref<4096xf32, #hivm.address_space<ub>>)
        %311 = memref_ext.alloc_workspace() from %arg2 offset = [%c100352] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xf32, #hivm.address_space<gm>>
        hivm.hir.pipe_barrier[<PIPE_ALL>]
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 14
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
        %312 = hivm.hir.pointer_cast(%c90496_i64) : memref<64x32xf32, #hivm.address_space<ub>>
        %collapse_shape_175 = memref.collapse_shape %311 [[0, 1]] : memref<64x32xf32, #hivm.address_space<gm>> into memref<2048xf32, #hivm.address_space<gm>>
        %collapse_shape_176 = memref.collapse_shape %312 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
        hivm.hir.load ins(%collapse_shape_175 : memref<2048xf32, #hivm.address_space<gm>>) outs(%collapse_shape_176 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
        hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        %313 = hivm.hir.pointer_cast(%c98688_i64) : memref<64x32xf32, #hivm.address_space<ub>>
        %314 = hivm.hir.pointer_cast(%c5216_i64) : memref<512xf32, #hivm.address_space<ub>>
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
        hivm.hir.vmul ins(%312, %collapse_shape_19 : memref<64x32xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>) outs(%313 : memref<64x32xf32, #hivm.address_space<ub>>) temp_buffer(%314 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
        %315 = hivm.hir.pointer_cast(%c256_i64) : memref<64x32xf32, #hivm.address_space<ub>>
        %collapse_shape_177 = memref.collapse_shape %307 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
        %collapse_shape_178 = memref.collapse_shape %315 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vcast ins(%collapse_shape_177 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_178 : memref<2048xf32, #hivm.address_space<ub>>)
        %316 = hivm.hir.pointer_cast(%c90496_i64) : memref<64x32xf32, #hivm.address_space<ub>>
        %collapse_shape_179 = memref.collapse_shape %316 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vmul ins(%collapse_shape_176, %collapse_shape_178 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_179 : memref<2048xf32, #hivm.address_space<ub>>)
        %317 = hivm.hir.pointer_cast(%c256_i64) : memref<64x1xf32, #hivm.address_space<ub>>
        hivm.hir.pipe_barrier[<PIPE_V>]
        hivm.hir.vreduce <sum> ins(%316 : memref<64x32xf32, #hivm.address_space<ub>>) outs(%317 : memref<64x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
        %collapse_shape_180 = memref.collapse_shape %317 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
        %318 = hivm.hir.pointer_cast(%c98688_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
        %collapse_shape_181 = memref.collapse_shape %313 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
        %collapse_shape_182 = memref.collapse_shape %318 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
        hivm.hir.vcast ins(%collapse_shape_181 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_182 : memref<2048xbf16, #hivm.address_space<ub>>)
        hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        %319 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 4096, 0)>()[%62, %305, %95]
        %320 = affine.min affine_map<()[s0] -> (64, s0)>()[%319]
        %321 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 4096) * 4096 + 128, 0)>()[%305]
        %322 = affine.min affine_map<()[s0] -> (32, s0)>()[%321]
        %323 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%98, %319]
        %324 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%320, %323]
        %325 = arith.subi %c0_i32, %235 : i32
        %326 = arith.maxsi %325, %c0_i32 : i32
        %327 = arith.index_cast %326 : i32 to index
        %328 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%327, %321]
        %329 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%322, %328]
        %subview_183 = memref.subview %318[%323, %328] [%324, %329] [1, 1] : memref<64x32xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_184 = memref.subview %reinterpret_cast_169[0, 0] [%324, %329] [1, 1] : memref<64x32xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
        scf.if %77 {
          hivm.hir.store ins(%subview_183 : memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_184 : memref<?x?xbf16, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>)
        } {limit_sub_block_id0}
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        scf.yield %310, %collapse_shape_180 : memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
      } else {
        scf.yield %10, %9 : memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
      }
      %301 = hivm.hir.pointer_cast(%c178304_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      %collapse_shape_165 = memref.collapse_shape %arg37 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %collapse_shape_166 = memref.collapse_shape %300#0 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %collapse_shape_167 = memref.collapse_shape %301 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.vadd ins(%collapse_shape_165, %collapse_shape_166 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_167 : memref<4096xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
      %302 = hivm.hir.pointer_cast(%c123264_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.pipe_barrier[<PIPE_V>]
      hivm.hir.vadd ins(%arg38, %300#1 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%302 : memref<64xf32, #hivm.address_space<ub>>)
      scf.yield %287, %291, %295, %299, %301, %302 : memref<32xf32, #hivm.address_space<ub>>, memref<64x32xf32, #hivm.address_space<ub>>, memref<64x32xf32, #hivm.address_space<ub>>, memref<64x32xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
    }
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID3>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 2
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 3
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 4
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 5
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 6
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 7
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 8
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_S>] flag = 9
    %173 = hivm.hir.pointer_cast(%c106880_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_65 = memref.collapse_shape %156 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_66 = memref.collapse_shape %173 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_65, %cst_2 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_66 : memref<2048xf32, #hivm.address_space<ub>>)
    %174 = hivm.hir.pointer_cast(%c106880_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_67 = memref.collapse_shape %174 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%collapse_shape_66 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_67 : memref<2048xf32, #hivm.address_space<ub>>)
    %175 = hivm.hir.pointer_cast(%c115072_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %176 = hivm.hir.pointer_cast(%c256_i64) : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%174, %collapse_shape_19 : memref<64x32xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>) outs(%175 : memref<64x32xf32, #hivm.address_space<ub>>) temp_buffer(%176 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
    %177 = hivm.hir.pointer_cast(%c61824_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%164, %cst_2 : memref<32xf32, #hivm.address_space<ub>>, f32) outs(%177 : memref<32xf32, #hivm.address_space<ub>>)
    %178 = hivm.hir.pointer_cast(%c61824_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%177 : memref<32xf32, #hivm.address_space<ub>>) outs(%178 : memref<32xf32, #hivm.address_space<ub>>)
    %179 = hivm.hir.pointer_cast(%c61824_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%172#0, %178 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%179 : memref<32xf32, #hivm.address_space<ub>>)
    %180 = hivm.hir.pointer_cast(%c256_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_68 = memref.collapse_shape %172#1 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_69 = memref.collapse_shape %180 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_68, %collapse_shape_67 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_69 : memref<2048xf32, #hivm.address_space<ub>>)
    %181 = hivm.hir.pointer_cast(%c256_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_70 = memref.collapse_shape %181 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_69, %arg22 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_70 : memref<2048xf32, #hivm.address_space<ub>>)
    %expand_shape_71 = memref.expand_shape %164 [[0, 1]] output_shape [1, 32] : memref<32xf32, #hivm.address_space<ub>> into memref<1x32xf32, #hivm.address_space<ub>>
    %182 = hivm.hir.pointer_cast(%c12544_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    hivm.hir.vsub ins(%expand_shape_71, %156 : memref<1x32xf32, #hivm.address_space<ub>>, memref<64x32xf32, #hivm.address_space<ub>>) outs(%182 : memref<64x32xf32, #hivm.address_space<ub>>) broadcast = [0]
    %183 = hivm.hir.pointer_cast(%c12544_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_72 = memref.collapse_shape %182 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_73 = memref.collapse_shape %183 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_72, %cst_2 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_73 : memref<2048xf32, #hivm.address_space<ub>>)
    %184 = hivm.hir.pointer_cast(%c12544_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_74 = memref.collapse_shape %184 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vexp ins(%collapse_shape_73 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_74 : memref<2048xf32, #hivm.address_space<ub>>)
    %subview_75 = memref.subview %184[0, 0] [%65, 32] [1, 1] : memref<64x32xf32, #hivm.address_space<ub>> to memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %185 = hivm.hir.pointer_cast(%c61952_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_76 = memref.collapse_shape %185 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.copy ins(%collapse_shape_4 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_76 : memref<2048xf32, #hivm.address_space<ub>>)
    %subview_77 = memref.subview %185[0, 0] [%65, 32] [1, 1] : memref<64x32xf32, #hivm.address_space<ub>> to memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<ub>>
    %collapse_shape_78 = memref.collapse_shape %subview_75 [[0, 1]] : memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    %collapse_shape_79 = memref.collapse_shape %subview_77 [[0, 1]] : memref<?x32xf32, strided<[32, 1]>, #hivm.address_space<ub>> into memref<?xf32, strided<[1]>, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.copy ins(%collapse_shape_78 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>) outs(%collapse_shape_79 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
    %186 = hivm.hir.pointer_cast(%c61952_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_80 = memref.collapse_shape %172#2 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_81 = memref.collapse_shape %186 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_80, %collapse_shape_76 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_81 : memref<2048xf32, #hivm.address_space<ub>>)
    %187 = hivm.hir.pointer_cast(%c70144_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_82 = memref.collapse_shape %141 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
    %collapse_shape_83 = memref.collapse_shape %187 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%collapse_shape_82 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_83 : memref<2048xf32, #hivm.address_space<ub>>)
    %188 = hivm.hir.pointer_cast(%c106880_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_84 = memref.collapse_shape %188 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_83, %collapse_shape_67 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_84 : memref<2048xf32, #hivm.address_space<ub>>)
    %189 = hivm.hir.pointer_cast(%c41216_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
    %collapse_shape_85 = memref.collapse_shape %172#3 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_86 = memref.collapse_shape %189 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcast ins(%collapse_shape_85 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_86 : memref<2048xbf16, #hivm.address_space<ub>>)
    %190 = hivm.hir.pointer_cast(%c20864_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_87 = memref.collapse_shape %190 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_86 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_87 : memref<2048xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    %191 = hivm.hir.pointer_cast(%c20864_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_88 = memref.collapse_shape %191 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_87, %cst_3 : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_88 : memref<2048xf32, #hivm.address_space<ub>>)
    %192 = hivm.hir.pointer_cast(%c20864_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_89 = memref.collapse_shape %192 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_88, %cst : memref<2048xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_89 : memref<2048xf32, #hivm.address_space<ub>>)
    %193 = hivm.hir.pointer_cast(%c20864_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
    %collapse_shape_90 = memref.collapse_shape %193 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vcast ins(%collapse_shape_89 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_90 : memref<2048xbf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %194 = memref_ext.alloc_workspace() from %arg2 offset = [%c108544] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.if %77 {
      %collapse_shape_127 = memref.collapse_shape %194 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<gm>> into memref<2048xbf16, #hivm.address_space<gm>>
      hivm.hir.store ins(%collapse_shape_90 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_127 : memref<2048xbf16, #hivm.address_space<gm>>)
    }
    annotation.mark %194 : memref<64x32xbf16, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %195 = memref_ext.alloc_workspace() from %arg2 offset = [%c112640] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
    scf.if %77 {
      %collapse_shape_127 = memref.collapse_shape %195 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<gm>> into memref<2048xbf16, #hivm.address_space<gm>>
      hivm.hir.store ins(%collapse_shape_90 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_127 : memref<2048xbf16, #hivm.address_space<gm>>)
    }
    annotation.mark %195 : memref<64x32xbf16, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %196 = hivm.hir.pointer_cast(%c29056_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
    %collapse_shape_91 = memref.collapse_shape %196 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
    hivm.hir.vcast ins(%collapse_shape_84 : memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_91 : memref<2048xbf16, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %197 = memref_ext.alloc_workspace() from %arg2 offset = [%c116736] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xbf16, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.if %77 {
      %collapse_shape_127 = memref.collapse_shape %197 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<gm>> into memref<2048xbf16, #hivm.address_space<gm>>
      hivm.hir.store ins(%collapse_shape_91 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_127 : memref<2048xbf16, #hivm.address_space<gm>>)
    }
    annotation.mark %197 : memref<64x32xbf16, #hivm.address_space<gm>>
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
    %198 = memref_ext.alloc_workspace() from %arg2 offset = [%c120832] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %199 = hivm.hir.pointer_cast(%c33152_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %collapse_shape_92 = memref.collapse_shape %198 [[0, 1]] : memref<64x64xf32, #hivm.address_space<gm>> into memref<4096xf32, #hivm.address_space<gm>>
    %collapse_shape_93 = memref.collapse_shape %199 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.load ins(%collapse_shape_92 : memref<4096xf32, #hivm.address_space<gm>>) outs(%collapse_shape_93 : memref<4096xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %200 = hivm.hir.pointer_cast(%c33152_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %collapse_shape_94 = memref.collapse_shape %200 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_93, %arg24 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_94 : memref<4096xf32, #hivm.address_space<ub>>)
    %201 = hivm.hir.pointer_cast(%c178304_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %collapse_shape_95 = memref.collapse_shape %172#4 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    %collapse_shape_96 = memref.collapse_shape %201 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_95, %collapse_shape_94 : memref<4096xf32, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_96 : memref<4096xf32, #hivm.address_space<ub>>)
    %202 = memref_ext.alloc_workspace() from %arg2 offset = [%c137216] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x32xf32, #hivm.address_space<gm>>
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
    %203 = hivm.hir.pointer_cast(%c49536_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_97 = memref.collapse_shape %202 [[0, 1]] : memref<64x32xf32, #hivm.address_space<gm>> into memref<2048xf32, #hivm.address_space<gm>>
    %collapse_shape_98 = memref.collapse_shape %203 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%collapse_shape_97 : memref<2048xf32, #hivm.address_space<gm>>) outs(%collapse_shape_98 : memref<2048xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %204 = hivm.hir.pointer_cast(%c106880_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_99 = memref.collapse_shape %204 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vmul ins(%collapse_shape_98, %collapse_shape_84 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_99 : memref<2048xf32, #hivm.address_space<ub>>)
    %205 = hivm.hir.pointer_cast(%c153472_i64) : memref<64x1xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%204 : memref<64x32xf32, #hivm.address_space<ub>>) outs(%205 : memref<64x1xf32, #hivm.address_space<ub>>) reduce_dims = [1]
    %collapse_shape_100 = memref.collapse_shape %205 [[0, 1]] : memref<64x1xf32, #hivm.address_space<ub>> into memref<64xf32, #hivm.address_space<ub>>
    %206 = hivm.hir.pointer_cast(%c123264_i64) : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%172#5, %collapse_shape_100 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%206 : memref<64xf32, #hivm.address_space<ub>>)
    %207 = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 + s2 * 128)>()[%139, %43, %95]
    %reinterpret_cast_101 = memref.reinterpret_cast %arg3 to offset: [%207], sizes: [64, 32], strides: [128, 1] : memref<?xbf16, #hivm.address_space<gm>> to memref<64x32xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %208 = hivm.hir.pointer_cast(%c57728_i64) : memref<64x32xbf16, #hivm.address_space<ub>>
    %209 = affine.max affine_map<()[s0, s1, s2] -> (s0 - s2 - s1 floordiv 128, 0)>()[%62, %139, %95]
    %210 = affine.min affine_map<()[s0] -> (64, s0)>()[%209]
    %211 = affine.max affine_map<()[s0] -> (-s0 + (s0 floordiv 128) * 128 + 128, 0)>()[%139]
    %212 = affine.min affine_map<()[s0] -> (32, s0)>()[%211]
    %213 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%98, %209]
    %214 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%210, %213]
    %215 = affine.min affine_map<()[s0, s1] -> (32, s1, s0)>()[%150, %211]
    %216 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%212, %215]
    %217 = arith.cmpi slt, %214, %c64 : index
    %218 = arith.cmpi slt, %216, %c32 : index
    %219 = arith.ori %217, %218 : i1
    %subview_102 = memref.subview %reinterpret_cast_101[0, 0] [%214, %216] [1, 1] : memref<64x32xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_103 = memref.subview %208[%213, %215] [%214, %216] [1, 1] : memref<64x32xbf16, #hivm.address_space<ub>> to memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    scf.if %219 {
      %collapse_shape_127 = memref.collapse_shape %208 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
      hivm.hir.vbrc ins(%cst_0 : bf16) outs(%collapse_shape_127 : memref<2048xbf16, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
    } {hivm.unlikely_condition}
    hivm.hir.load ins(%subview_102 : memref<?x?xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%subview_103 : memref<?x?xbf16, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) pad_mode = <PadValue> pad_value = %cst_0 : bf16 left_padding_num = %215 : index init_out_buffer = false may_implicit_transpose_with_last_axis = false
    hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    %220 = hivm.hir.pointer_cast(%c70144_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_104 = memref.collapse_shape %220 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_83, %collapse_shape_81 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_104 : memref<2048xf32, #hivm.address_space<ub>>)
    %221 = hivm.hir.pointer_cast(%c153472_i64) : memref<1x32xf32, #hivm.address_space<ub>>
    %222 = hivm.hir.pointer_cast(%c153600_i64) : memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vreduce <sum> ins(%220 : memref<64x32xf32, #hivm.address_space<ub>>) outs(%221 : memref<1x32xf32, #hivm.address_space<ub>>) temp_buffer(%222 : memref<2048xf32, #hivm.address_space<ub>>) reduce_dims = [0]
    %collapse_shape_105 = memref.collapse_shape %221 [[0, 1]] : memref<1x32xf32, #hivm.address_space<ub>> into memref<32xf32, #hivm.address_space<ub>>
    %223 = hivm.hir.pointer_cast(%c153472_i64) : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%179, %collapse_shape_105 : memref<32xf32, #hivm.address_space<ub>>, memref<32xf32, #hivm.address_space<ub>>) outs(%223 : memref<32xf32, #hivm.address_space<ub>>)
    %224 = hivm.hir.pointer_cast(%c153600_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_106 = memref.collapse_shape %208 [[0, 1]] : memref<64x32xbf16, #hivm.address_space<ub>> into memref<2048xbf16, #hivm.address_space<ub>>
    %collapse_shape_107 = memref.collapse_shape %224 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.vcast ins(%collapse_shape_106 : memref<2048xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_107 : memref<2048xf32, #hivm.address_space<ub>>)
    %225 = hivm.hir.pointer_cast(%c153600_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_108 = memref.collapse_shape %225 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%collapse_shape_107, %collapse_shape_70 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_108 : memref<2048xf32, #hivm.address_space<ub>>)
    %226 = hivm.hir.pointer_cast(%c153600_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_109 = memref.collapse_shape %226 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vsub ins(%collapse_shape_108, %collapse_shape_104 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_109 : memref<2048xf32, #hivm.address_space<ub>>)
    %expand_shape_110 = memref.expand_shape %223 [[0, 1]] output_shape [1, 32] : memref<32xf32, #hivm.address_space<ub>> into memref<1x32xf32, #hivm.address_space<ub>>
    %227 = hivm.hir.pointer_cast(%c161792_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %228 = hivm.hir.pointer_cast(%c169984_i64) : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%92, %expand_shape_110 : memref<64x1xf32, #hivm.address_space<ub>>, memref<1x32xf32, #hivm.address_space<ub>>) outs(%227 : memref<64x32xf32, #hivm.address_space<ub>>) temp_buffer(%228 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [0, 1]
    %229 = hivm.hir.pointer_cast(%c153600_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_111 = memref.collapse_shape %227 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_112 = memref.collapse_shape %229 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_109, %collapse_shape_111 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_112 : memref<2048xf32, #hivm.address_space<ub>>)
    %230 = hivm.hir.pointer_cast(%c106880_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %231 = hivm.hir.pointer_cast(%c161792_i64) : memref<512xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vmul ins(%204, %collapse_shape_19 : memref<64x32xf32, #hivm.address_space<ub>>, memref<64x1xf32, #hivm.address_space<ub>>) outs(%230 : memref<64x32xf32, #hivm.address_space<ub>>) temp_buffer(%231 : memref<512xf32, #hivm.address_space<ub>>) broadcast = [1]
    %232 = hivm.hir.pointer_cast(%c106880_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_113 = memref.collapse_shape %230 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_114 = memref.collapse_shape %232 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_112, %collapse_shape_113 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_114 : memref<2048xf32, #hivm.address_space<ub>>)
    %233 = hivm.hir.pointer_cast(%c115072_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_115 = memref.collapse_shape %175 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    %collapse_shape_116 = memref.collapse_shape %233 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.vmul ins(%collapse_shape_98, %collapse_shape_115 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_116 : memref<2048xf32, #hivm.address_space<ub>>)
    %234 = hivm.hir.pointer_cast(%c115072_i64) : memref<64x32xf32, #hivm.address_space<ub>>
    %collapse_shape_117 = memref.collapse_shape %234 [[0, 1]] : memref<64x32xf32, #hivm.address_space<ub>> into memref<2048xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_V>]
    hivm.hir.vadd ins(%collapse_shape_81, %collapse_shape_116 : memref<2048xf32, #hivm.address_space<ub>>, memref<2048xf32, #hivm.address_space<ub>>) outs(%collapse_shape_117 : memref<2048xf32, #hivm.address_space<ub>>)
    hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    %reinterpret_cast_118 = memref.reinterpret_cast %arg13 to offset: [%140], sizes: [64, 32], strides: [4096, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x32xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_119 = memref.reinterpret_cast %arg14 to offset: [%140], sizes: [64, 32], strides: [4096, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x32xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
    %reinterpret_cast_120 = memref.reinterpret_cast %arg17 to offset: [%140], sizes: [64, 32], strides: [4096, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x32xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
    %subview_121 = memref.subview %181[%146, %151] [%147, %152] [1, 1] : memref<64x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_122 = memref.subview %reinterpret_cast_118[0, 0] [%147, %152] [1, 1] : memref<64x32xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
    scf.if %77 {
      hivm.hir.store ins(%subview_121 : memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_122 : memref<?x?xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>)
    } {limit_sub_block_id0}
    %subview_123 = memref.subview %234[%146, %151] [%147, %152] [1, 1] : memref<64x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_124 = memref.subview %reinterpret_cast_119[0, 0] [%147, %152] [1, 1] : memref<64x32xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
    scf.if %77 {
      hivm.hir.store ins(%subview_123 : memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_124 : memref<?x?xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>)
    } {limit_sub_block_id0}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
    %subview_125 = memref.subview %232[%146, %151] [%147, %152] [1, 1] : memref<64x32xf32, #hivm.address_space<ub>> to memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>
    %subview_126 = memref.subview %reinterpret_cast_120[0, 0] [%147, %152] [1, 1] : memref<64x32xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x?xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>
    scf.if %77 {
      hivm.hir.store ins(%subview_125 : memref<?x?xf32, strided<[32, 1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_126 : memref<?x?xf32, strided<[4096, 1], offset: ?>, #hivm.address_space<gm>>)
    } {limit_sub_block_id0}
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    scf.yield %201, %206 : memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
  }
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_S>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID3>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID4>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID5>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID6>]
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  %expand_shape_22 = memref.expand_shape %30 [[0, 1]] output_shape [64, 1] : memref<64xi32, #hivm.address_space<ub>> into memref<64x1xi32, #hivm.address_space<ub>>
  %expand_shape_23 = memref.expand_shape %30 [[0, 1]] output_shape [1, 64] : memref<64xi32, #hivm.address_space<ub>> into memref<1x64xi32, #hivm.address_space<ub>>
  %104 = hivm.hir.pointer_cast(%c123520_i64) : memref<64x64xi32, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%expand_shape_23 : memref<1x64xi32, #hivm.address_space<ub>>) outs(%104 : memref<64x64xi32, #hivm.address_space<ub>>) broadcast_dims = [0]
  %105 = hivm.hir.pointer_cast(%c148992_i64) : memref<64x64xi32, #hivm.address_space<ub>>
  %106 = hivm.hir.pointer_cast(%c139904_i64) : memref<512xi32, #hivm.address_space<ub>>
  hivm.hir.vmax ins(%expand_shape_22, %expand_shape_23 : memref<64x1xi32, #hivm.address_space<ub>>, memref<1x64xi32, #hivm.address_space<ub>>) outs(%105 : memref<64x64xi32, #hivm.address_space<ub>>) temp_buffer(%106 : memref<512xi32, #hivm.address_space<ub>>) broadcast = [0, 1]
  %107 = hivm.hir.pointer_cast(%c148992_i64) : memref<64x64xi1, #hivm.address_space<ub>>
  %collapse_shape_24 = memref.collapse_shape %105 [[0, 1]] : memref<64x64xi32, #hivm.address_space<ub>> into memref<4096xi32, #hivm.address_space<ub>>
  %collapse_shape_25 = memref.collapse_shape %104 [[0, 1]] : memref<64x64xi32, #hivm.address_space<ub>> into memref<4096xi32, #hivm.address_space<ub>>
  %collapse_shape_26 = memref.collapse_shape %107 [[0, 1]] : memref<64x64xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_24, %collapse_shape_25 : memref<4096xi32, #hivm.address_space<ub>>, memref<4096xi32, #hivm.address_space<ub>>) outs(%collapse_shape_26 : memref<4096xi1, #hivm.address_space<ub>>)
  %108 = hivm.hir.pointer_cast(%c148992_i64) : memref<64x64xi1, #hivm.address_space<ub>>
  %collapse_shape_27 = memref.collapse_shape %108 [[0, 1]] : memref<64x64xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_26 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_27 : memref<4096xi1, #hivm.address_space<ub>>)
  %109 = hivm.hir.pointer_cast(%c123520_i64) : memref<64xf16, #hivm.address_space<ub>>
  %110 = hivm.hir.pointer_cast(%c123648_i64) : memref<48xf16, #hivm.address_space<ub>>
  hivm.hir.vcast ins(%33 : memref<64xi1, #hivm.address_space<ub>>) outs(%109 : memref<64xf16, #hivm.address_space<ub>>) temp_buffer(%110 : memref<48xf16, #hivm.address_space<ub>>) round_mode = <trunc>
  %expand_shape_28 = memref.expand_shape %109 [[0, 1]] output_shape [64, 1] : memref<64xf16, #hivm.address_space<ub>> into memref<64x1xf16, #hivm.address_space<ub>>
  %111 = hivm.hir.pointer_cast(%c123648_i64) : memref<64x64xf16, #hivm.address_space<ub>>
  %112 = hivm.hir.pointer_cast(%c131840_i64) : memref<1024xf16, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vbrc ins(%expand_shape_28 : memref<64x1xf16, #hivm.address_space<ub>>) outs(%111 : memref<64x64xf16, #hivm.address_space<ub>>) temp_buffer(%112 : memref<1024xf16, #hivm.address_space<ub>>) broadcast_dims = [1]
  %113 = hivm.hir.pointer_cast(%c123648_i64) : memref<64x64xi1, #hivm.address_space<ub>>
  %collapse_shape_29 = memref.collapse_shape %111 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
  %collapse_shape_30 = memref.collapse_shape %113 [[0, 1]] : memref<64x64xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_29, %cst_1 : memref<4096xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_30 : memref<4096xi1, #hivm.address_space<ub>>)
  %114 = hivm.hir.pointer_cast(%c123648_i64) : memref<64x64xi1, #hivm.address_space<ub>>
  %collapse_shape_31 = memref.collapse_shape %114 [[0, 1]] : memref<64x64xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_30 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_31 : memref<4096xi1, #hivm.address_space<ub>>)
  %expand_shape_32 = memref.expand_shape %109 [[0, 1]] output_shape [1, 64] : memref<64xf16, #hivm.address_space<ub>> into memref<1x64xf16, #hivm.address_space<ub>>
  %115 = hivm.hir.pointer_cast(%c131840_i64) : memref<64x64xf16, #hivm.address_space<ub>>
  hivm.hir.vbrc ins(%expand_shape_32 : memref<1x64xf16, #hivm.address_space<ub>>) outs(%115 : memref<64x64xf16, #hivm.address_space<ub>>) broadcast_dims = [0]
  %116 = hivm.hir.pointer_cast(%c131840_i64) : memref<64x64xi1, #hivm.address_space<ub>>
  %collapse_shape_33 = memref.collapse_shape %115 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
  %collapse_shape_34 = memref.collapse_shape %116 [[0, 1]] : memref<64x64xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcmp ins(%collapse_shape_33, %cst_1 : memref<4096xf16, #hivm.address_space<ub>>, f16) outs(%collapse_shape_34 : memref<4096xi1, #hivm.address_space<ub>>)
  %117 = hivm.hir.pointer_cast(%c131840_i64) : memref<64x64xi1, #hivm.address_space<ub>>
  %collapse_shape_35 = memref.collapse_shape %117 [[0, 1]] : memref<64x64xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vnot ins(%collapse_shape_34 : memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_35 : memref<4096xi1, #hivm.address_space<ub>>)
  %118 = hivm.hir.pointer_cast(%c123648_i64) : memref<64x64xi1, #hivm.address_space<ub>>
  %collapse_shape_36 = memref.collapse_shape %118 [[0, 1]] : memref<64x64xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vand ins(%collapse_shape_31, %collapse_shape_35 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_36 : memref<4096xi1, #hivm.address_space<ub>>)
  %119 = hivm.hir.pointer_cast(%c123648_i64) : memref<64x64xi1, #hivm.address_space<ub>>
  %collapse_shape_37 = memref.collapse_shape %119 [[0, 1]] : memref<64x64xi1, #hivm.address_space<ub>> into memref<4096xi1, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vand ins(%collapse_shape_27, %collapse_shape_36 : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xi1, #hivm.address_space<ub>>) outs(%collapse_shape_37 : memref<4096xi1, #hivm.address_space<ub>>)
  %120 = hivm.hir.pointer_cast(%c131840_i64) : memref<64x64xf32, #hivm.address_space<ub>>
  hivm.hir.vmul ins(%103#0, %61 : memref<64x64xf32, #hivm.address_space<ub>>, memref<1x64xf32, #hivm.address_space<ub>>) outs(%120 : memref<64x64xf32, #hivm.address_space<ub>>) broadcast = [0]
  %121 = hivm.hir.pointer_cast(%c8448_i64) : memref<64x64xf32, #hivm.address_space<ub>>
  %collapse_shape_38 = memref.collapse_shape %120 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %collapse_shape_39 = memref.collapse_shape %121 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %122 = hivm.hir.pointer_cast(%c123520_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_37, %collapse_shape_38, %cst : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_39 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%122 : memref<16xf32, #hivm.address_space<ub>>)
  %123 = hivm.hir.pointer_cast(%c8448_i64) : memref<64x64xbf16, #hivm.address_space<ub>>
  %collapse_shape_40 = memref.collapse_shape %123 [[0, 1]] : memref<64x64xbf16, #hivm.address_space<ub>> into memref<4096xbf16, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vcast ins(%collapse_shape_39 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_40 : memref<4096xbf16, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  %124 = memref_ext.alloc_workspace() from %arg2 offset = [%c145408] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  scf.if %77 {
    %collapse_shape_52 = memref.collapse_shape %124 [[0, 1]] : memref<64x64xbf16, #hivm.address_space<gm>> into memref<4096xbf16, #hivm.address_space<gm>>
    hivm.hir.store ins(%collapse_shape_40 : memref<4096xbf16, #hivm.address_space<ub>>) outs(%collapse_shape_52 : memref<4096xbf16, #hivm.address_space<gm>>)
  }
  annotation.mark %124 : memref<64x64xbf16, #hivm.address_space<gm>>
  hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_S>] flag = 1
  %125 = memref_ext.alloc_workspace() from %arg2 offset = [%c161792] : from memref<?xi8, #hivm.address_space<gm>> to memref<64x64xf32, #hivm.address_space<gm>>
  hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_S>] flag = 0
  %126 = hivm.hir.pointer_cast(%c24832_i64) : memref<64x64xf32, #hivm.address_space<ub>>
  %collapse_shape_41 = memref.collapse_shape %125 [[0, 1]] : memref<64x64xf32, #hivm.address_space<gm>> into memref<4096xf32, #hivm.address_space<gm>>
  %collapse_shape_42 = memref.collapse_shape %126 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.load ins(%collapse_shape_41 : memref<4096xf32, #hivm.address_space<gm>>) outs(%collapse_shape_42 : memref<4096xf32, #hivm.address_space<ub>>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  %127 = hivm.hir.pointer_cast(%c24832_i64) : memref<64x64xf32, #hivm.address_space<ub>>
  %collapse_shape_43 = memref.collapse_shape %127 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.vmul ins(%collapse_shape_42, %cst_3 : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_43 : memref<4096xf32, #hivm.address_space<ub>>)
  %128 = hivm.hir.pointer_cast(%c24832_i64) : memref<64x64xf32, #hivm.address_space<ub>>
  %collapse_shape_44 = memref.collapse_shape %128 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vadd ins(%collapse_shape_43, %cst : memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_44 : memref<4096xf32, #hivm.address_space<ub>>)
  %129 = hivm.hir.pointer_cast(%c41216_i64) : memref<64x64xf32, #hivm.address_space<ub>>
  %collapse_shape_45 = memref.collapse_shape %129 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
  %130 = hivm.hir.pointer_cast(%c123520_i64) : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.pipe_barrier[<PIPE_V>]
  hivm.hir.vsel ins(%collapse_shape_37, %collapse_shape_44, %cst : memref<4096xi1, #hivm.address_space<ub>>, memref<4096xf32, #hivm.address_space<ub>>, f32) outs(%collapse_shape_45 : memref<4096xf32, #hivm.address_space<ub>>) temp_buffer(%130 : memref<16xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  %131 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%51, %95]
  %reinterpret_cast_46 = memref.reinterpret_cast %arg19 to offset: [%131], sizes: [64, 64], strides: [64, 1] : memref<?xf32, #hivm.address_space<gm>> to memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
  %132 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%95, %49]
  %reinterpret_cast_47 = memref.reinterpret_cast %arg18 to offset: [%132], sizes: [64], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %133 = affine.max affine_map<()[s0, s1] -> (0, s0 - s1)>()[%62, %95]
  %134 = affine.min affine_map<()[s0] -> (64, s0)>()[%133]
  %135 = affine.min affine_map<()[s0, s1] -> (64, s1, s0)>()[%98, %133]
  %136 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%134, %135]
  %subview_48 = memref.subview %129[%135, 0] [%136, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
  %subview_49 = memref.subview %reinterpret_cast_46[0, 0] [%136, 64] [1, 1] : memref<64x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  scf.if %77 {
    %collapse_shape_52 = memref.collapse_shape %subview_48 [[0, 1]] : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    %collapse_shape_53 = memref.collapse_shape %subview_49 [[0, 1]] : memref<?x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
    hivm.hir.store ins(%collapse_shape_52 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) outs(%collapse_shape_53 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  } {limit_sub_block_id0}
  %subview_50 = memref.subview %103#1[%135] [%136] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
  %subview_51 = memref.subview %reinterpret_cast_47[0] [%136] [1] : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  scf.if %77 {
    hivm.hir.store ins(%subview_50 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>) outs(%subview_51 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  } {limit_sub_block_id0}
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}

ld.lld: warning: -z separate-code and -z separate-loadable-segments will be ignored
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.
warning: overriding the module target triple with aarch64-unknown-linux-gnu [-Woverride-module]
1 warning generated.