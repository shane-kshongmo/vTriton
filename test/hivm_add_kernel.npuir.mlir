// Minimal HIVM-style fixture for text-based scheduling analysis.
func.func @add_kernel(%arg0: memref<?xf32, #hivm.address_space<gm>>,
                      %arg1: memref<?xf32, #hivm.address_space<gm>>,
                      %arg2: memref<?xf32, #hivm.address_space<gm>>) {
  %c0_i64 = arith.constant 0 : i64
  %src0 = memref.reinterpret_cast %arg0 to offset: [%c0_i64], sizes: [1024], strides: [1]
      : memref<?xf32, #hivm.address_space<gm>>
      to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %src1 = memref.reinterpret_cast %arg1 to offset: [%c0_i64], sizes: [1024], strides: [1]
      : memref<?xf32, #hivm.address_space<gm>>
      to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %dst = memref.reinterpret_cast %arg2 to offset: [%c0_i64], sizes: [1024], strides: [1]
      : memref<?xf32, #hivm.address_space<gm>>
      to memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %ub0 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
  %ub1 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
  %ub2 = hivm.hir.pointer_cast(%c0_i64) : memref<1024xf32, #hivm.address_space<ub>>
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
  %a = hivm.hir.load ins(%src0 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      outs(%ub0 : memref<1024xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]
  %b = hivm.hir.load ins(%src1 : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
      outs(%ub1 : memref<1024xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
  %c = hivm.hir.vadd ins(%ub0, %ub1 : memref<1024xf32, #hivm.address_space<ub>>,
                         memref<1024xf32, #hivm.address_space<ub>>)
      outs(%ub2 : memref<1024xf32, #hivm.address_space<ub>>)
  hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
  hivm.hir.store ins(%ub2 : memref<1024xf32, #hivm.address_space<ub>>)
      outs(%dst : memref<1024xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  hivm.hir.pipe_barrier[<PIPE_ALL>]
  return
}
