//===- HivmOpsEditor.h - HIVM ops CRUD C++ API ----------------*- C++ -*-===//
//
// Provides a C++ API for programmatically creating, reading, updating, and
// deleting HIVM operations on an MLIR ModuleOp.  Designed to be called from
// the upper-level optimization pipeline (C++) to apply optimisation
// suggestions directly to an MLIR file.
//
// Typical workflow:
//   1. Read  .bak file  ->  HivmOpsEditor::loadFromFile(ctx, path)
//   2. Apply optimisation suggestions via the editor methods
//   3. Write result     ->  editor.exportToFile(outputPath)
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_TRANSFORMS_HIVMOPS_EDITOR_H
#define ASCENDMODEL_TRANSFORMS_HIVMOPS_EDITOR_H

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>

namespace mlir {
namespace ascend {

//===----------------------------------------------------------------------===//
// OpInfo - lightweight descriptor for a single HIVM op
//===----------------------------------------------------------------------===//

struct HivmOpInfo {
  unsigned index;
  std::string qualifiedName;
  Operation *op;
};

//===----------------------------------------------------------------------===//
// HivmOpsEditor - main API class
//===----------------------------------------------------------------------===//

class HivmOpsEditor {
public:
  explicit HivmOpsEditor(ModuleOp module) : module(module) {}

  static OwningOpRef<ModuleOp> loadFromFile(MLIRContext &ctx,
                                            llvm::StringRef path);

  LogicalResult exportToFile(llvm::StringRef path) const;
  std::string exportToString() const;

  ModuleOp getModule() const { return module; }

  //===--------------------------------------------------------------------===//
  // READ
  //===--------------------------------------------------------------------===//

  SmallVector<HivmOpInfo> listOps() const;
  std::map<std::string, unsigned> opCounts() const;
  void printSummary(raw_ostream &os) const;

  template <typename OpT>
  SmallVector<OpT> collectOps() const {
    SmallVector<OpT> result;
    module.walk([&](OpT op) { result.push_back(op); });
    return result;
  }

  //===--------------------------------------------------------------------===//
  // CREATE - add new HIVM ops
  //===--------------------------------------------------------------------===//

  template <typename OpT, typename... Args>
  OpT createOpAfter(Operation *target, Location loc, Args &&...args) {
    OpBuilder builder(target->getContext());
    builder.setInsertionPointAfter(target);
    return builder.create<OpT>(loc, std::forward<Args>(args)...);
  }

  template <typename OpT, typename... Args>
  OpT createOpBefore(Operation *target, Location loc, Args &&...args) {
    OpBuilder builder(target->getContext());
    builder.setInsertionPoint(target);
    return builder.create<OpT>(loc, std::forward<Args>(args)...);
  }

  void addSetFlagWaitFlagBefore(Operation *target, PIPE setPipe,
                                PIPE waitPipe, EVENT_ID eventId);
  void addSetFlagWaitFlagAfter(Operation *target, PIPE setPipe,
                               PIPE waitPipe, EVENT_ID eventId);

  hivm::LoadOp addLoadBefore(Operation *target, Value src, Value dst);
  hivm::StoreOp addStoreBefore(Operation *target, Value src, Value dst);
  hivm::VAddOp addVAddAfter(Operation *target, ValueRange src, ValueRange dst);
  hivm::VMulOp addVMulAfter(Operation *target, ValueRange src, ValueRange dst);
  hivm::VSubOp addVSubAfter(Operation *target, ValueRange src, ValueRange dst);

  //===--------------------------------------------------------------------===//
  // DELETE
  //===--------------------------------------------------------------------===//

  void deleteOp(Operation *op);
  void deleteAllOpsWithName(llvm::StringRef opName);
  void deleteNthOpWithName(llvm::StringRef opName, unsigned n);

  template <typename OpT> void deleteAllOpsOfKind() {
    SmallVector<Operation *> toErase;
    module.walk([&](OpT op) { toErase.push_back(op); });
    for (auto *op : toErase)
      op->erase();
  }

  void deleteSyncOpsForOp(Operation *computeOp);
  void deleteRedundantGMTrips(unsigned count);

  //===--------------------------------------------------------------------===//
  // MODIFY - change attributes / replace ops
  //===--------------------------------------------------------------------===//

  template <typename OldOpT, typename NewOpT, typename... ExtraArgs>
  NewOpT replaceOpWith(OldOpT oldOp, ExtraArgs &&...extraArgs) {
    IRRewriter rewriter(oldOp->getContext());
    rewriter.setInsertionPoint(oldOp);
    auto newOp = rewriter.create<NewOpT>(
        oldOp->getLoc(), oldOp->getResultTypes(),
        std::forward<ExtraArgs>(extraArgs)...);
    rewriter.replaceOp(oldOp, newOp->getResults());
    return newOp;
  }

  void changeElementType(Type oldType, Type newType);
  void changeMemorySpace(llvm::StringRef oldSpace, llvm::StringRef newSpace);
  void changePipeAttr(PIPE oldPipe, PIPE newPipe);
  void changeEventAttr(EVENT_ID oldEvent, EVENT_ID newEvent);
  void changeShape(ArrayRef<int64_t> oldShape, ArrayRef<int64_t> newShape);

  void setSetFlagPipe(Operation *setFlagOp, PIPE newPipe);
  void setWaitFlagPipe(Operation *waitFlagOp, PIPE newPipe);
  void setEventId(Operation *syncOp, EVENT_ID newEvent);

  //===--------------------------------------------------------------------===//
  // OPTIMISATION-DRIVEN CONVENIENCE
  //===--------------------------------------------------------------------===//

  void removeRedundantLoadStorePair(unsigned n);
  void fuseConsecutiveComputeOps();
  void insertDoubleBuffering(Value src, Value ub0, Value ub1,
                             PIPE setPipe, PIPE waitPipe,
                             EVENT_ID eventId);

private:
  ModuleOp module;
};

} // namespace ascend
} // namespace mlir

#endif // TRITONSIM_HAS_BISHENGIR_HIVM

#endif // ASCENDMODEL_TRANSFORMS_HIVMOPS_EDITOR_H
