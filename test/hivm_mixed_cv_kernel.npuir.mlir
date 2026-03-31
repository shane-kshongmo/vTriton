// Synthetic mixed CV HIVM-style fixture.
func.func @mixed_cv_kernel(%a: memref<?xf16, #hivm.address_space<gm>>,
                           %b: memref<?xf16, #hivm.address_space<gm>>,
                           %c: memref<?xf32, #hivm.address_space<gm>>,
                           %out: memref<?xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %true = arith.constant true
  %a_l1 = hivm.hir.pointer_cast(%c0_i64) : memref<128x128xf16, #hivm.address_space<cbuf>>
  %b_l1 = hivm.hir.pointer_cast(%c0_i64) : memref<128x128xf16, #hivm.address_space<cbuf>>
  %c_l0c = hivm.hir.pointer_cast(%c0_i64) : memref<128x128xf32, #hivm.address_space<cc>>
  %cv_ub = hivm.hir.pointer_cast(%c0_i64) : memref<128x128xf32, #hivm.address_space<ub>>
  %gm_a = memref.reinterpret_cast %a to offset: [%c0], sizes: [128, 128], strides: [128, 1]
      : memref<?xf16, #hivm.address_space<gm>>
      to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  %gm_b = memref.reinterpret_cast %b to offset: [%c0], sizes: [128, 128], strides: [128, 1]
      : memref<?xf16, #hivm.address_space<gm>>
      to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  %gm_c = memref.reinterpret_cast %c to offset: [%c0], sizes: [128, 128], strides: [128, 1]
      : memref<?xf32, #hivm.address_space<gm>>
      to memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
  %gm_out = memref.reinterpret_cast %out to offset: [%c0], sizes: [128, 128], strides: [128, 1]
      : memref<?xf32, #hivm.address_space<gm>>
      to memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>

  hivm.hir.nd2nz ins(%gm_a : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
      outs(%a_l1 : memref<128x128xf16, #hivm.address_space<cbuf>>)
  hivm.hir.nd2nz ins(%gm_b : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
      outs(%b_l1 : memref<128x128xf16, #hivm.address_space<cbuf>>)
  hivm.hir.mmadL1 ins(%a_l1, %b_l1, %true, %c128, %c128, %c128
      : memref<128x128xf16, #hivm.address_space<cbuf>>,
        memref<128x128xf16, #hivm.address_space<cbuf>>,
        i1, index, index, index)
      outs(%c_l0c : memref<128x128xf32, #hivm.address_space<cc>>)
  hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
  hivm.hir.fixpipe ins(%c_l0c : memref<128x128xf32, #hivm.address_space<cc>>)
      outs(%cv_ub : memref<128x128xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_FIX>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_V>, <EVENT_ID1>]
  hivm.hir.load ins(%gm_c : memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
      outs(%cv_ub : memref<128x128xf32, #hivm.address_space<ub>>)
  hivm.hir.vadd ins(%cv_ub, %cv_ub : memref<128x128xf32, #hivm.address_space<ub>>,
                    memref<128x128xf32, #hivm.address_space<ub>>)
      outs(%cv_ub : memref<128x128xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID2>]
  hivm.hir.store ins(%cv_ub : memref<128x128xf32, #hivm.address_space<ub>>)
      outs(%gm_out : memref<128x128xf32, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}
