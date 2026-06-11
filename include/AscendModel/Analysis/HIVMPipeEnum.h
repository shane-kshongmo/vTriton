//===- HIVMPipeEnum.h - HIVM pipe enumeration -*- C++ -*-===//
//
// Defines the HIVMPipe enum used by both HIVMAnalysis and
// HIVMBottleneckDiagnosis, breaking the circular include dependency.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ANALYSIS_HIVMPIEEENUM_H
#define ASCENDMODEL_ANALYSIS_HIVMPIEEENUM_H

namespace mlir {
namespace ascend {

enum class HIVMPipe {
  Unknown,
  Vector,
  VectorMTE2,
  CubeMTE2,
  MTE3,
  Scalar,
  FixPipe,
  Cube,
  MTE1,
  All
};

} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_ANALYSIS_HIVMPIEEENUM_H