//===- HivmOpsEditor.cpp - HIVM ops CRUD C++ API implementation --------===//
//
// Implements the HivmOpsEditor class that provides a C++ API for
// programmatically creating, reading, updating, and deleting HIVM
// operations on an MLIR ModuleOp.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Transforms/HivmOpsEditor.h"

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::hivm;

//===----------------------------------------------------------------------===//
// File I/O
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> HivmOpsEditor::loadFromFile(MLIRContext &ctx,
                                                   llvm::StringRef path) {
  ctx.loadDialect<HIVMDialect, arith::ArithDialect, func::FuncDialect,
                  memref::MemRefDialect>();
  auto parsed = parseSourceFile<ModuleOp>(path, &ctx);
  if (!parsed) {
    llvm::errs() << "HivmOpsEditor: failed to parse " << path << "\n";
    return nullptr;
  }
  return std::move(*parsed);
}

LogicalResult HivmOpsEditor::exportToFile(llvm::StringRef path) const {
  std::error_code ec;
  llvm::raw_fd_ostream outFile(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "HivmOpsEditor: cannot open " << path << ": "
                 << ec.message() << "\n";
    return failure();
  }
  module.print(outFile);
  return success();
}

std::string HivmOpsEditor::exportToString() const {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  module.print(os);
  return buf;
}

//===----------------------------------------------------------------------===//
// READ
//===----------------------------------------------------------------------===//

SmallVector<HivmOpInfo> HivmOpsEditor::listOps() const {
  SmallVector<HivmOpInfo> result;
  unsigned idx = 0;
  module.walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "hivm") {
      result.push_back(
          {idx++, op->getName().getStringRef().str(), op});
    }
  });
  return result;
}

std::map<std::string, unsigned> HivmOpsEditor::opCounts() const {
  std::map<std::string, unsigned> counts;
  module.walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "hivm")
      counts[op->getName().getStringRef().str()]++;
  });
  return counts;
}

void HivmOpsEditor::printSummary(raw_ostream &os) const {
  auto ops = listOps();
  os << "Found " << ops.size() << " HIVM operations:\n";
  for (auto &info : ops) {
    os << "  [" << info.index << "] " << info.qualifiedName;
    if (auto loc = info.op->getLoc().dyn_cast<FileLineColLoc>())
      os << "  (line " << loc.getLine() << ")";
    os << "\n";
  }
  os << "\nOperation counts:\n";
  for (auto &[name, cnt] : opCounts())
    os << "  " << name << ": " << cnt << "\n";
}

//===----------------------------------------------------------------------===//
// CREATE
//===----------------------------------------------------------------------===//

void HivmOpsEditor::addSetFlagWaitFlagBefore(Operation *target,
                                              PIPE setPipe, PIPE waitPipe,
                                              EVENT_ID eventId) {
  OpBuilder builder(target->getContext());
  auto *ctx = target->getContext();
  builder.setInsertionPoint(target);
  auto loc = target->getLoc();
  auto sp = PipeAttr::get(ctx, setPipe);
  auto wp = PipeAttr::get(ctx, waitPipe);
  auto ev = EventAttr::get(ctx, eventId);
  builder.create<SetFlagOp>(loc, sp, wp, ev, Value());
  builder.create<WaitFlagOp>(loc, sp, wp, ev, Value());
}

void HivmOpsEditor::addSetFlagWaitFlagAfter(Operation *target,
                                             PIPE setPipe, PIPE waitPipe,
                                             EVENT_ID eventId) {
  OpBuilder builder(target->getContext());
  auto *ctx = target->getContext();
  builder.setInsertionPointAfter(target);
  auto loc = target->getLoc();
  auto sp = PipeAttr::get(ctx, setPipe);
  auto wp = PipeAttr::get(ctx, waitPipe);
  auto ev = EventAttr::get(ctx, eventId);
  builder.create<SetFlagOp>(loc, sp, wp, ev, Value());
  builder.create<WaitFlagOp>(loc, sp, wp, ev, Value());
}

hivm::LoadOp HivmOpsEditor::addLoadBefore(Operation *target, Value src,
                                           Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<LoadOp>(target->getLoc(), TypeRange{}, src, dst);
}

hivm::StoreOp HivmOpsEditor::addStoreBefore(Operation *target, Value src,
                                             Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<StoreOp>(target->getLoc(), TypeRange{}, src, dst);
}

hivm::VAddOp HivmOpsEditor::addVAddAfter(Operation *target, ValueRange src,
                                          ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VAddOp>(target->getLoc(), TypeRange{}, src, dst);
}

hivm::VMulOp HivmOpsEditor::addVMulAfter(Operation *target, ValueRange src,
                                          ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VMulOp>(target->getLoc(), TypeRange{}, src, dst);
}

hivm::VSubOp HivmOpsEditor::addVSubAfter(Operation *target, ValueRange src,
                                          ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VSubOp>(target->getLoc(), TypeRange{}, src, dst);
}

//===----------------------------------------------------------------------===//
// DELETE
//===----------------------------------------------------------------------===//

void HivmOpsEditor::deleteOp(Operation *op) { op->erase(); }

void HivmOpsEditor::deleteAllOpsWithName(llvm::StringRef opName) {
  SmallVector<Operation *> toErase;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == opName)
      toErase.push_back(op);
  });
  for (auto *op : toErase)
    op->erase();
}

void HivmOpsEditor::deleteNthOpWithName(llvm::StringRef opName, unsigned n) {
  unsigned count = 0;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == opName) {
      if (count == n) {
        op->erase();
        return WalkResult::interrupt();
      }
      ++count;
    }
    return WalkResult::advance();
  });
}

void HivmOpsEditor::deleteSyncOpsForOp(Operation *computeOp) {
  SmallVector<Operation *> toErase;
  auto *block = computeOp->getBlock();
  if (!block)
    return;

  for (auto &op : *block) {
    if (&op == computeOp)
      continue;
    if (auto sf = dyn_cast<SetFlagOp>(&op)) {
      if (sf.getSetPipe() && sf.getWaitPipe()) {
        auto setP = sf.getSetPipe().getPipe();
        auto waitP = sf.getWaitPipe().getPipe();
        if ((setP == PIPE::PIPE_V && waitP == PIPE::PIPE_MTE2) ||
            (setP == PIPE::PIPE_MTE2 && waitP == PIPE::PIPE_V) ||
            (setP == PIPE::PIPE_V && waitP == PIPE::PIPE_MTE3) ||
            (setP == PIPE::PIPE_MTE3 && waitP == PIPE::PIPE_V))
          toErase.push_back(&op);
      }
    } else if (auto wf = dyn_cast<WaitFlagOp>(&op)) {
      if (wf.getSetPipe() && wf.getWaitPipe()) {
        auto setP = wf.getSetPipe().getPipe();
        auto waitP = wf.getWaitPipe().getPipe();
        if ((setP == PIPE::PIPE_V && waitP == PIPE::PIPE_MTE2) ||
            (setP == PIPE::PIPE_MTE2 && waitP == PIPE::PIPE_V) ||
            (setP == PIPE::PIPE_V && waitP == PIPE::PIPE_MTE3) ||
            (setP == PIPE::PIPE_MTE3 && waitP == PIPE::PIPE_V))
          toErase.push_back(&op);
      }
    }
  }
  for (auto *op : toErase)
    op->erase();
}

void HivmOpsEditor::deleteRedundantGMTrips(unsigned count) {
  SmallVector<Operation *> loadOps;
  SmallVector<Operation *> storeOps;
  module.walk([&](LoadOp op) { loadOps.push_back(op); });
  module.walk([&](StoreOp op) { storeOps.push_back(op); });

  unsigned removed = 0;
  for (auto *load : loadOps) {
    if (removed >= count)
      break;
    auto dstVal = cast<LoadOp>(load).getDst();
    SmallVector<Operation *> users(dstVal.getUsers().begin(),
                                   dstVal.getUsers().end());
    if (users.size() == 1) {
      SmallVector<Operation *> toErase;
      auto *block = load->getBlock();
      if (!block)
        continue;
      bool foundLoad = false;
      for (auto &op : *block) {
        if (&op == load) {
          foundLoad = true;
          continue;
        }
        if (!foundLoad)
          continue;
        if (auto sf = dyn_cast<SetFlagOp>(&op)) {
          toErase.push_back(&op);
          continue;
        }
        if (auto wf = dyn_cast<WaitFlagOp>(&op)) {
          toErase.push_back(&op);
          continue;
        }
        if (&op == users[0])
          break;
      }
      for (auto *e : toErase)
        e->erase();
      load->erase();
      ++removed;
    }
  }
}

//===----------------------------------------------------------------------===//
// MODIFY
//===----------------------------------------------------------------------===//

void HivmOpsEditor::changeElementType(Type oldType, Type newType) {
  module.walk([&](Operation *op) {
    for (auto &region : op->getRegions()) {
      region.walk([&](Operation *innerOp) {
        for (unsigned i = 0; i < innerOp->getNumOperands(); ++i) {
          auto opType = innerOp->getOperand(i).getType();
          if (auto memref = opType.dyn_cast<MemRefType>()) {
            if (memref.getElementType() == oldType) {
              auto newMemref = MemRefType::get(
                  memref.getShape(), newType, memref.getLayout(),
                  memref.getMemorySpace());
              innerOp->getOperand(i).setType(newMemref);
            }
          }
        }
        for (unsigned i = 0; i < innerOp->getNumResults(); ++i) {
          auto resType = innerOp->getResult(i).getType();
          if (auto memref = resType.dyn_cast<MemRefType>()) {
            if (memref.getElementType() == oldType) {
              auto newMemref = MemRefType::get(
                  memref.getShape(), newType, memref.getLayout(),
                  memref.getMemorySpace());
              innerOp->getResult(i).setType(newMemref);
            }
          }
        }
      });
    }
  });
}

void HivmOpsEditor::changeMemorySpace(llvm::StringRef oldSpace,
                                       llvm::StringRef newSpace) {
  auto *ctx = module.getContext();
  AddressSpace newAS;
  if (newSpace == "gm")
    newAS = AddressSpace::GM;
  else if (newSpace == "ub")
    newAS = AddressSpace::UB;
  else if (newSpace == "l1" || newSpace == "cbuf")
    newAS = AddressSpace::L1;
  else
    return;

  auto newSpaceAttr = AddressSpaceAttr::get(ctx, newAS);

  module.walk([&](Operation *op) {
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto opType = op->getOperand(i).getType();
      if (auto memref = opType.dyn_cast<MemRefType>()) {
        auto spaceAttr =
            memref.getMemorySpace().dyn_cast_or_null<AddressSpaceAttr>();
        if (spaceAttr) {
          std::string curSpace;
          switch (spaceAttr.getValue()) {
          case AddressSpace::GM: curSpace = "gm"; break;
          case AddressSpace::UB: curSpace = "ub"; break;
          case AddressSpace::L1: curSpace = "l1"; break;
          default: break;
          }
          if (curSpace == oldSpace.str()) {
            auto newMemref = MemRefType::get(
                memref.getShape(), memref.getElementType(),
                memref.getLayout(), newSpaceAttr);
            op->getOperand(i).setType(newMemref);
          }
        }
      }
    }
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      auto resType = op->getResult(i).getType();
      if (auto memref = resType.dyn_cast<MemRefType>()) {
        auto spaceAttr =
            memref.getMemorySpace().dyn_cast_or_null<AddressSpaceAttr>();
        if (spaceAttr) {
          std::string curSpace;
          switch (spaceAttr.getValue()) {
          case AddressSpace::GM: curSpace = "gm"; break;
          case AddressSpace::UB: curSpace = "ub"; break;
          case AddressSpace::L1: curSpace = "l1"; break;
          default: break;
          }
          if (curSpace == oldSpace.str()) {
            auto newMemref = MemRefType::get(
                memref.getShape(), memref.getElementType(),
                memref.getLayout(), newSpaceAttr);
            op->getResult(i).setType(newMemref);
          }
        }
      }
    }
  });
}

void HivmOpsEditor::changePipeAttr(PIPE oldPipe, PIPE newPipe) {
  auto *ctx = module.getContext();
  auto newPipeAttr = PipeAttr::get(ctx, newPipe);
  module.walk([&](Operation *op) {
    for (unsigned i = 0; i < op->getNumAttrs(); ++i) {
      auto attr = op->getAttr(i);
      if (auto pipeAttr = attr.dyn_cast<PipeAttr>()) {
        if (pipeAttr.getPipe() == oldPipe)
          op->setAttr(op->getAttrNames()[i], newPipeAttr);
      }
    }
  });
}

void HivmOpsEditor::changeEventAttr(EVENT_ID oldEvent, EVENT_ID newEvent) {
  auto *ctx = module.getContext();
  auto newEventAttr = EventAttr::get(ctx, newEvent);
  module.walk([&](Operation *op) {
    for (unsigned i = 0; i < op->getNumAttrs(); ++i) {
      auto attr = op->getAttr(i);
      if (auto eventAttr = attr.dyn_cast<EventAttr>()) {
        if (eventAttr.getValue() == oldEvent)
          op->setAttr(op->getAttrNames()[i], newEventAttr);
      }
    }
  });
}

void HivmOpsEditor::changeShape(ArrayRef<int64_t> oldShape,
                                 ArrayRef<int64_t> newShape) {
  module.walk([&](Operation *op) {
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto opType = op->getOperand(i).getType();
      if (auto memref = opType.dyn_cast<MemRefType>()) {
        if (memref.getShape() == oldShape) {
          auto newMemref = MemRefType::get(
              newShape, memref.getElementType(), memref.getLayout(),
              memref.getMemorySpace());
          op->getOperand(i).setType(newMemref);
        }
      }
    }
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      auto resType = op->getResult(i).getType();
      if (auto memref = resType.dyn_cast<MemRefType>()) {
        if (memref.getShape() == oldShape) {
          auto newMemref = MemRefType::get(
              newShape, memref.getElementType(), memref.getLayout(),
              memref.getMemorySpace());
          op->getResult(i).setType(newMemref);
        }
      }
    }
  });
}

void HivmOpsEditor::setSetFlagPipe(Operation *setFlagOp, PIPE newPipe) {
  auto *ctx = setFlagOp->getContext();
  auto newPipeAttr = PipeAttr::get(ctx, newPipe);
  setFlagOp->setAttr("set_pipe", newPipeAttr);
}

void HivmOpsEditor::setWaitFlagPipe(Operation *waitFlagOp, PIPE newPipe) {
  auto *ctx = waitFlagOp->getContext();
  auto newPipeAttr = PipeAttr::get(ctx, newPipe);
  waitFlagOp->setAttr("wait_pipe", newPipeAttr);
}

void HivmOpsEditor::setEventId(Operation *syncOp, EVENT_ID newEvent) {
  auto *ctx = syncOp->getContext();
  auto newEventAttr = EventAttr::get(ctx, newEvent);
  syncOp->setAttr("static_event_id", newEventAttr);
}

//===----------------------------------------------------------------------===//
// OPTIMISATION-DRIVEN CONVENIENCE
//===----------------------------------------------------------------------===//

void HivmOpsEditor::removeRedundantLoadStorePair(unsigned n) {
  deleteRedundantGMTrips(n);
}

void HivmOpsEditor::fuseConsecutiveComputeOps() {
  SmallVector<VAddOp> vadds;
  module.walk([&](VAddOp op) { vadds.push_back(op); });
  for (auto vadd : vadds) {
    auto *nextOp = vadd->getNextNode();
    if (!nextOp)
      continue;
    auto nextVAdd = dyn_cast<VAddOp>(nextOp);
    if (!nextVAdd)
      continue;
    auto dst0 = vadd.getDst();
    auto src1 = nextVAdd.getSrc();
    if (src1.size() < 1)
      continue;
    bool usesResult = false;
    for (auto s : src1) {
      if (s == dst0[0]) {
        usesResult = true;
        break;
      }
    }
    if (usesResult) {
      IRRewriter rewriter(vadd->getContext());
      rewriter.setInsertionPoint(nextVAdd);
      auto fused = rewriter.create<VAddOp>(
          nextVAdd->getLoc(), nextVAdd->getResultTypes(),
          vadd.getSrc(), nextVAdd.getDst());
      rewriter.replaceOp(nextVAdd, fused->getResults());
    }
  }
}

void HivmOpsEditor::insertDoubleBuffering(Value src, Value ub0, Value ub1,
                                           PIPE setPipe, PIPE waitPipe,
                                           EVENT_ID eventId) {
  auto *ctx = module.getContext();
  OpBuilder builder(ctx);

  module.walk([&](func::FuncOp func) {
    auto *entry = &func.getBody().front();
    builder.setInsertionPointToStart(entry);
    auto loc = builder.getUnknownLoc();

    auto sp = PipeAttr::get(ctx, setPipe);
    auto wp = PipeAttr::get(ctx, waitPipe);
    auto ev0 = EventAttr::get(ctx, eventId);
    auto ev1 = EventAttr::get(ctx, static_cast<EVENT_ID>(
        static_cast<int>(eventId) + 1));

    builder.create<SetFlagOp>(loc, sp, wp, ev0, Value());
    builder.create<WaitFlagOp>(loc, sp, wp, ev0, Value());
    builder.create<LoadOp>(loc, TypeRange{}, src, ub0);

    builder.create<SetFlagOp>(loc, sp, wp, ev1, Value());
    builder.create<WaitFlagOp>(loc, sp, wp, ev1, Value());
    builder.create<LoadOp>(loc, TypeRange{}, src, ub1);
  });
}

#endif // TRITONSIM_HAS_BISHENGIR_HIVM
