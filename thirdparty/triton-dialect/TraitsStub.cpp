// Stub implementations for Triton trait verification functions.
// The full Traits.cpp depends on TritonGPU which is incompatible with LLVM 20.
// For the modelling pipeline, GPU layout verification is not needed.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

using namespace mlir;

static constexpr int64_t maxTensorNumElements = 131072;

namespace mlir {
namespace OpTrait {
namespace impl {

LogicalResult verifySameOperandsEncoding(Operation *op,
                                         bool allowTensorPointerType) {
  return success();
}

LogicalResult verifySameOperandsAndResultEncoding(
    Operation *op, bool allowTensorPointerType) {
  return success();
}

LogicalResult verifyTensorSize(Operation *op) {
  for (auto opType : op->getOperandTypes()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(opType)) {
      int64_t numElements = 1;
      for (int64_t s : tensorType.getShape())
        numElements *= s;
      if (numElements > maxTensorNumElements)
        return op->emitError("Maximum allowed number of elements is ")
               << maxTensorNumElements << ", but " << *op
               << " has more than that";
    }
  }
  for (auto opType : op->getResultTypes()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(opType)) {
      int64_t numElements = 1;
      for (int64_t s : tensorType.getShape())
        numElements *= s;
      if (numElements > maxTensorNumElements)
        return op->emitError("Maximum allowed number of elements is ")
               << maxTensorNumElements << ", but " << *op
               << " has more than that";
    }
  }
  return success();
}

LogicalResult verifyTensorLayouts(Operation *op) { return success(); }

LogicalResult verifySameLoadStoreOperandsShape(Operation *op) {
  return success();
}

LogicalResult verifySameLoadStoreOperandsAndResultShape(Operation *op) {
  return success();
}

} // namespace impl
} // namespace OpTrait
} // namespace mlir
