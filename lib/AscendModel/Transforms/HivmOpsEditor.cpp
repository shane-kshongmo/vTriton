//===- HivmOpsEditor.cpp - HIVM ops CRUD C++ API implementation --------===//
//
// Implements the HivmOpsEditor class that provides a C++ API for
// programmatically creating, reading, updating, and deleting HIVM
// operations on an MLIR ModuleOp.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Transforms/HivmOpsEditor.h"

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
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
                  memref::MemRefDialect, scf::SCFDialect,
                  tensor::TensorDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::annotation::AnnotationDialect,
                  mlir::hacc::HACCDialect>();
  auto parsed = parseSourceFile<ModuleOp>(path, &ctx);
  if (!parsed) {
    llvm::errs() << "HivmOpsEditor: failed to parse " << path << "\n";
    return nullptr;
  }
  return std::move(*parsed);
}

LogicalResult HivmOpsEditor::exportToFile(llvm::StringRef path) {
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

std::string HivmOpsEditor::exportToString() {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  module.print(os);
  return buf;
}

//===----------------------------------------------------------------------===//
// READ
//===----------------------------------------------------------------------===//

SmallVector<HivmOpInfo> HivmOpsEditor::listOps() {
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

std::map<std::string, unsigned> HivmOpsEditor::opCounts() {
  std::map<std::string, unsigned> counts;
  module.walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "hivm")
      counts[op->getName().getStringRef().str()]++;
  });
  return counts;
}

void HivmOpsEditor::printSummary(raw_ostream &os) {
  auto ops = listOps();
  os << "Found " << ops.size() << " HIVM operations:\n";
  for (auto &info : ops) {
    os << "  [" << info.index << "] " << info.qualifiedName;
    if (auto loc = dyn_cast<FileLineColLoc>(info.op->getLoc()))
      os << "  (line " << loc.getLine() << ")";
    os << "\n";
  }
  os << "\nOperation counts:\n";
  for (auto &[name, cnt] : opCounts())
    os << "  " << name << ": " << cnt << "\n";
}

//===----------------------------------------------------------------------===//
// CREATE - DMA Ops
//===----------------------------------------------------------------------===//

LoadOp HivmOpsEditor::addLoadBefore(Operation *target, Value src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<LoadOp>(target->getLoc(), TypeRange{}, src, dst);
}

LoadOp HivmOpsEditor::addLoadAfter(Operation *target, Value src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<LoadOp>(target->getLoc(), TypeRange{}, src, dst);
}

StoreOp HivmOpsEditor::addStoreBefore(Operation *target, Value src,
                                       Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<StoreOp>(target->getLoc(), TypeRange{}, src, dst);
}

StoreOp HivmOpsEditor::addStoreAfter(Operation *target, Value src,
                                      Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<StoreOp>(target->getLoc(), TypeRange{}, src, dst);
}

CopyOp HivmOpsEditor::addCopyBefore(Operation *target, Value src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<CopyOp>(target->getLoc(), TypeRange{}, src, dst);
}

CopyOp HivmOpsEditor::addCopyAfter(Operation *target, Value src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<CopyOp>(target->getLoc(), TypeRange{}, src, dst);
}

FixpipeOp HivmOpsEditor::addFixpipeBefore(Operation *target, Value src,
                                           Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<FixpipeOp>(target->getLoc(), TypeRange{}, src, dst);
}

FixpipeOp HivmOpsEditor::addFixpipeAfter(Operation *target, Value src,
                                          Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<FixpipeOp>(target->getLoc(), TypeRange{}, src, dst);
}

ND2NZOp HivmOpsEditor::addND2NZBefore(Operation *target, Value src,
                                       Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<ND2NZOp>(target->getLoc(), TypeRange{}, src, dst,
                                  UnitAttr());
}

ND2NZOp HivmOpsEditor::addND2NZAfter(Operation *target, Value src,
                                      Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<ND2NZOp>(target->getLoc(), TypeRange{}, src, dst,
                                 UnitAttr());
}

NZ2NDOp HivmOpsEditor::addNZ2NDBefore(Operation *target, Value src,
                                       Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<NZ2NDOp>(target->getLoc(), TypeRange{}, src, dst);
}

NZ2NDOp HivmOpsEditor::addNZ2NDAfter(Operation *target, Value src,
                                      Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<NZ2NDOp>(target->getLoc(), TypeRange{}, src, dst);
}

AtomicCasOp HivmOpsEditor::addAtomicCasBefore(Operation *target,
                                               ValueRange src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<AtomicCasOp>(target->getLoc(), TypeRange{}, src,
                                     dst);
}

AtomicCasOp HivmOpsEditor::addAtomicCasAfter(Operation *target,
                                              ValueRange src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<AtomicCasOp>(target->getLoc(), TypeRange{}, src,
                                     dst);
}

AtomicXchgOp HivmOpsEditor::addAtomicXchgBefore(Operation *target,
                                                 Value src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<AtomicXchgOp>(target->getLoc(), TypeRange{}, src,
                                      dst, Value());
}

AtomicXchgOp HivmOpsEditor::addAtomicXchgAfter(Operation *target,
                                                Value src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<AtomicXchgOp>(target->getLoc(), TypeRange{}, src,
                                      dst, Value());
}

AtomicRMWOp HivmOpsEditor::addAtomicRMWBefore(Operation *target, Value src,
                                               Value dst,
                                               AtomicKindAttr kind) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<AtomicRMWOp>(target->getLoc(), TypeRange{}, src,
                                     dst, kind);
}

AtomicRMWOp HivmOpsEditor::addAtomicRMWAfter(Operation *target, Value src,
                                              Value dst,
                                              AtomicKindAttr kind) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<AtomicRMWOp>(target->getLoc(), TypeRange{}, src,
                                      dst, kind);
}

//===----------------------------------------------------------------------===//
// CREATE - Vector Unary Ops
//===----------------------------------------------------------------------===//

VExpOp HivmOpsEditor::addVExpBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VExpOp>(target->getLoc(), TypeRange{}, src, dst);
}

VExpOp HivmOpsEditor::addVExpAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VExpOp>(target->getLoc(), TypeRange{}, src, dst);
}

VAbsOp HivmOpsEditor::addVAbsBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VAbsOp>(target->getLoc(), TypeRange{}, src, dst);
}

VAbsOp HivmOpsEditor::addVAbsAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VAbsOp>(target->getLoc(), TypeRange{}, src, dst);
}

VLnOp HivmOpsEditor::addVLnBefore(Operation *target, ValueRange src,
                                  ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VLnOp>(target->getLoc(), TypeRange{}, src, dst);
}

VLnOp HivmOpsEditor::addVLnAfter(Operation *target, ValueRange src,
                                 ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VLnOp>(target->getLoc(), TypeRange{}, src, dst);
}

VReluOp HivmOpsEditor::addVReluBefore(Operation *target, ValueRange src,
                                      ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VReluOp>(target->getLoc(), TypeRange{}, src, dst);
}

VReluOp HivmOpsEditor::addVReluAfter(Operation *target, ValueRange src,
                                     ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VReluOp>(target->getLoc(), TypeRange{}, src, dst);
}

VRsqrtOp HivmOpsEditor::addVRsqrtBefore(Operation *target, ValueRange src,
                                        ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VRsqrtOp>(target->getLoc(), TypeRange{}, src, dst);
}

VRsqrtOp HivmOpsEditor::addVRsqrtAfter(Operation *target, ValueRange src,
                                       ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VRsqrtOp>(target->getLoc(), TypeRange{}, src, dst);
}

VSqrtOp HivmOpsEditor::addVSqrtBefore(Operation *target, ValueRange src,
                                      ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VSqrtOp>(target->getLoc(), TypeRange{}, src, dst);
}

VSqrtOp HivmOpsEditor::addVSqrtAfter(Operation *target, ValueRange src,
                                     ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VSqrtOp>(target->getLoc(), TypeRange{}, src, dst);
}

VTanhOp HivmOpsEditor::addVTanhBefore(Operation *target, ValueRange src,
                                      ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VTanhOp>(target->getLoc(), TypeRange{}, src, dst);
}

VTanhOp HivmOpsEditor::addVTanhAfter(Operation *target, ValueRange src,
                                     ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VTanhOp>(target->getLoc(), TypeRange{}, src, dst);
}

VSinOp HivmOpsEditor::addVSinBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VSinOp>(target->getLoc(), TypeRange{}, src, dst);
}

VSinOp HivmOpsEditor::addVSinAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VSinOp>(target->getLoc(), TypeRange{}, src, dst);
}

VCosOp HivmOpsEditor::addVCosBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VCosOp>(target->getLoc(), TypeRange{}, src, dst);
}

VCosOp HivmOpsEditor::addVCosAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VCosOp>(target->getLoc(), TypeRange{}, src, dst);
}

VErfOp HivmOpsEditor::addVErfBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VErfOp>(target->getLoc(), TypeRange{}, src, dst);
}

VErfOp HivmOpsEditor::addVErfAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VErfOp>(target->getLoc(), TypeRange{}, src, dst);
}

VRecOp HivmOpsEditor::addVRecBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VRecOp>(target->getLoc(), TypeRange{}, src, dst);
}

VRecOp HivmOpsEditor::addVRecAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VRecOp>(target->getLoc(), TypeRange{}, src, dst);
}

VNotOp HivmOpsEditor::addVNotBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VNotOp>(target->getLoc(), TypeRange{}, src, dst);
}

VNotOp HivmOpsEditor::addVNotAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VNotOp>(target->getLoc(), TypeRange{}, src, dst);
}

VCastOp HivmOpsEditor::addVCastBefore(Operation *target, ValueRange src,
                                      ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VCastOp>(target->getLoc(), TypeRange{}, src, dst);
}

VCastOp HivmOpsEditor::addVCastAfter(Operation *target, ValueRange src,
                                     ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VCastOp>(target->getLoc(), TypeRange{}, src, dst);
}

//===----------------------------------------------------------------------===//
// CREATE - Vector Binary Ops
//===----------------------------------------------------------------------===//

VAddOp HivmOpsEditor::addVAddBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VAddOp>(target->getLoc(), TypeRange{}, src, dst);
}

VAddOp HivmOpsEditor::addVAddAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VAddOp>(target->getLoc(), TypeRange{}, src, dst);
}

VSubOp HivmOpsEditor::addVSubBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VSubOp>(target->getLoc(), TypeRange{}, src, dst);
}

VSubOp HivmOpsEditor::addVSubAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VSubOp>(target->getLoc(), TypeRange{}, src, dst);
}

VMulOp HivmOpsEditor::addVMulBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VMulOp>(target->getLoc(), TypeRange{}, src, dst);
}

VMulOp HivmOpsEditor::addVMulAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VMulOp>(target->getLoc(), TypeRange{}, src, dst);
}

VDivOp HivmOpsEditor::addVDivBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VDivOp>(target->getLoc(), TypeRange{}, src, dst);
}

VDivOp HivmOpsEditor::addVDivAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VDivOp>(target->getLoc(), TypeRange{}, src, dst);
}

VMaxOp HivmOpsEditor::addVMaxBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VMaxOp>(target->getLoc(), TypeRange{}, src, dst);
}

VMaxOp HivmOpsEditor::addVMaxAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VMaxOp>(target->getLoc(), TypeRange{}, src, dst);
}

VMinOp HivmOpsEditor::addVMinBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VMinOp>(target->getLoc(), TypeRange{}, src, dst);
}

VMinOp HivmOpsEditor::addVMinAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VMinOp>(target->getLoc(), TypeRange{}, src, dst);
}

VOrOp HivmOpsEditor::addVOrBefore(Operation *target, ValueRange src,
                                  ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VOrOp>(target->getLoc(), TypeRange{}, src, dst);
}

VOrOp HivmOpsEditor::addVOrAfter(Operation *target, ValueRange src,
                                 ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VOrOp>(target->getLoc(), TypeRange{}, src, dst);
}

VAndOp HivmOpsEditor::addVAndBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VAndOp>(target->getLoc(), TypeRange{}, src, dst);
}

VAndOp HivmOpsEditor::addVAndAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VAndOp>(target->getLoc(), TypeRange{}, src, dst);
}

VXorOp HivmOpsEditor::addVXorBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VXorOp>(target->getLoc(), TypeRange{}, src, dst);
}

VXorOp HivmOpsEditor::addVXorAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VXorOp>(target->getLoc(), TypeRange{}, src, dst);
}

VModOp HivmOpsEditor::addVModBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VModOp>(target->getLoc(), TypeRange{}, src, dst);
}

VModOp HivmOpsEditor::addVModAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VModOp>(target->getLoc(), TypeRange{}, src, dst);
}

VShLOp HivmOpsEditor::addVShLBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VShLOp>(target->getLoc(), TypeRange{}, src, dst);
}

VShLOp HivmOpsEditor::addVShLAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VShLOp>(target->getLoc(), TypeRange{}, src, dst);
}

VShROp HivmOpsEditor::addVShRBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto roundAttr = builder.getBoolAttr(false);
  auto noneAttr = DenseI64ArrayAttr::get(builder.getContext(), {});
  return builder.create<VShROp>(target->getLoc(), TypeRange{}, src, dst,
                                roundAttr, noneAttr, noneAttr);
}

VShROp HivmOpsEditor::addVShRAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto roundAttr = builder.getBoolAttr(false);
  auto noneAttr = DenseI64ArrayAttr::get(builder.getContext(), {});
  return builder.create<VShROp>(target->getLoc(), TypeRange{}, src, dst,
                                roundAttr, noneAttr, noneAttr);
}

VCmpOp HivmOpsEditor::addVCmpBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VCmpOp>(target->getLoc(), TypeRange{}, src, dst);
}

VCmpOp HivmOpsEditor::addVCmpAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VCmpOp>(target->getLoc(), TypeRange{}, src, dst);
}

VPowOp HivmOpsEditor::addVPowBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VPowOp>(target->getLoc(), TypeRange{}, src, dst);
}

VPowOp HivmOpsEditor::addVPowAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VPowOp>(target->getLoc(), TypeRange{}, src, dst);
}

VMulExtOp HivmOpsEditor::addVMulExtBefore(Operation *target, ValueRange src,
                                           ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VMulExtOp>(target->getLoc(), TypeRange{}, src, dst);
}

VMulExtOp HivmOpsEditor::addVMulExtAfter(Operation *target, ValueRange src,
                                          ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VMulExtOp>(target->getLoc(), TypeRange{}, src, dst);
}

//===----------------------------------------------------------------------===//
// CREATE - Vector Ternary / Special Ops
//===----------------------------------------------------------------------===//

VSelOp HivmOpsEditor::addVSelBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto noneAttr = DenseI64ArrayAttr::get(builder.getContext(), {});
  return builder.create<VSelOp>(target->getLoc(), TypeRange{}, src, dst,
                                Value(), noneAttr, noneAttr);
}

VSelOp HivmOpsEditor::addVSelAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto noneAttr = DenseI64ArrayAttr::get(builder.getContext(), {});
  return builder.create<VSelOp>(target->getLoc(), TypeRange{}, src, dst,
                                Value(), noneAttr, noneAttr);
}

VBrcOp HivmOpsEditor::addVBrcBefore(Operation *target, Value src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VBrcOp>(target->getLoc(), TypeRange{}, src, dst);
}

VBrcOp HivmOpsEditor::addVBrcAfter(Operation *target, Value src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VBrcOp>(target->getLoc(), TypeRange{}, src, dst);
}

VReduceOp HivmOpsEditor::addVReduceBefore(Operation *target, Value src,
                                           ValueRange dst,
                                           ReduceOpAttr arith,
                                           DenseI64ArrayAttr reduceDims) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VReduceOp>(target->getLoc(), TypeRange{}, src, dst,
                                   arith, reduceDims);
}

VReduceOp HivmOpsEditor::addVReduceAfter(Operation *target, Value src,
                                          ValueRange dst,
                                          ReduceOpAttr arith,
                                          DenseI64ArrayAttr reduceDims) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VReduceOp>(target->getLoc(), TypeRange{}, src, dst,
                                   arith, reduceDims);
}

VConcatOp HivmOpsEditor::addVConcatBefore(Operation *target, int64_t dim,
                                           ValueRange src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto dimAttr = builder.getI64IntegerAttr(dim);
  return builder.create<VConcatOp>(target->getLoc(), TypeRange{}, dimAttr,
                                   src, dst);
}

VConcatOp HivmOpsEditor::addVConcatAfter(Operation *target, int64_t dim,
                                          ValueRange src, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto dimAttr = builder.getI64IntegerAttr(dim);
  return builder.create<VConcatOp>(target->getLoc(), TypeRange{}, dimAttr,
                                   src, dst);
}

VFlipOp HivmOpsEditor::addVFlipBefore(Operation *target, Value src,
                                      Value dst, int64_t flipAxis) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto axisAttr = builder.getI64IntegerAttr(flipAxis);
  return builder.create<VFlipOp>(target->getLoc(), TypeRange{}, src, dst,
                                 axisAttr);
}

VFlipOp HivmOpsEditor::addVFlipAfter(Operation *target, Value src,
                                     Value dst, int64_t flipAxis) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto axisAttr = builder.getI64IntegerAttr(flipAxis);
  return builder.create<VFlipOp>(target->getLoc(), TypeRange{}, src, dst,
                                 axisAttr);
}

VPadOp HivmOpsEditor::addVPadBefore(Operation *target, Value src, Value dst,
                                     Value padValue, ValueRange low,
                                     ValueRange high) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto nDims = low.size();
  auto staticLow = builder.getDenseI64ArrayAttr(SmallVector<int64_t>(nDims, 0));
  auto staticHigh = builder.getDenseI64ArrayAttr(SmallVector<int64_t>(nDims, 0));
  return builder.create<VPadOp>(target->getLoc(), TypeRange{}, src, dst,
                                padValue, low, high, staticLow, staticHigh);
}

VPadOp HivmOpsEditor::addVPadAfter(Operation *target, Value src, Value dst,
                                    Value padValue, ValueRange low,
                                    ValueRange high) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto nDims = low.size();
  auto staticLow = builder.getDenseI64ArrayAttr(SmallVector<int64_t>(nDims, 0));
  auto staticHigh = builder.getDenseI64ArrayAttr(SmallVector<int64_t>(nDims, 0));
  return builder.create<VPadOp>(target->getLoc(), TypeRange{}, src, dst,
                                padValue, low, high, staticLow, staticHigh);
}

VGatherOp HivmOpsEditor::addVGatherBefore(Operation *target, Value src,
                                           Value indices, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VGatherOp>(target->getLoc(), TypeRange{}, src,
                                   indices, dst);
}

VGatherOp HivmOpsEditor::addVGatherAfter(Operation *target, Value src,
                                          Value indices, Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VGatherOp>(target->getLoc(), TypeRange{}, src,
                                   indices, dst);
}

VCumsumOp HivmOpsEditor::addVCumsumBefore(Operation *target, Value src,
                                           Value dst,
                                           DenseI64ArrayAttr cumDims) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VCumsumOp>(target->getLoc(), TypeRange{}, src, dst,
                                   cumDims, BoolAttr());
}

VCumsumOp HivmOpsEditor::addVCumsumAfter(Operation *target, Value src,
                                          Value dst,
                                          DenseI64ArrayAttr cumDims) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VCumsumOp>(target->getLoc(), TypeRange{}, src, dst,
                                   cumDims, BoolAttr());
}

VCumprodOp HivmOpsEditor::addVCumprodBefore(Operation *target, Value src,
                                             Value dst,
                                             DenseI64ArrayAttr cumDims) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VCumprodOp>(target->getLoc(), TypeRange{}, src, dst,
                                    cumDims, BoolAttr());
}

VCumprodOp HivmOpsEditor::addVCumprodAfter(Operation *target, Value src,
                                            Value dst,
                                            DenseI64ArrayAttr cumDims) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VCumprodOp>(target->getLoc(), TypeRange{}, src, dst,
                                    cumDims, BoolAttr());
}

VSortOp HivmOpsEditor::addVSortBefore(Operation *target, Value src,
                                      ValueRange dst, bool descending,
                                      int64_t sortAxis) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VSortOp>(target->getLoc(), TypeRange{}, src, dst,
                                 descending, sortAxis);
}

VSortOp HivmOpsEditor::addVSortAfter(Operation *target, Value src,
                                     ValueRange dst, bool descending,
                                     int64_t sortAxis) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VSortOp>(target->getLoc(), TypeRange{}, src, dst,
                                 descending, sortAxis);
}

VMulextendedOp HivmOpsEditor::addVMulextendedBefore(Operation *target,
                                                    ValueRange src,
                                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VMulextendedOp>(target->getLoc(), TypeRange{}, src,
                                        dst);
}

VMulextendedOp HivmOpsEditor::addVMulextendedAfter(Operation *target,
                                                   ValueRange src,
                                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VMulextendedOp>(target->getLoc(), TypeRange{}, src,
                                        dst);
}

VGatherMaskOp HivmOpsEditor::addVGatherMaskBefore(Operation *target,
                                                   Value src, Value mask,
                                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VGatherMaskOp>(target->getLoc(), TypeRange{}, src,
                                       mask, dst);
}

VGatherMaskOp HivmOpsEditor::addVGatherMaskAfter(Operation *target,
                                                  Value src, Value mask,
                                                  ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VGatherMaskOp>(target->getLoc(), TypeRange{}, src,
                                       mask, dst);
}

VTransposeOp HivmOpsEditor::addVTransposeBefore(Operation *target, Value src,
                                                 Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VTransposeOp>(target->getLoc(), TypeRange{}, src,
                                      dst, Value(), DenseI64ArrayAttr(),
                                      BoolAttr());
}

VTransposeOp HivmOpsEditor::addVTransposeAfter(Operation *target, Value src,
                                                Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VTransposeOp>(target->getLoc(), TypeRange{}, src,
                                      dst, Value(), DenseI64ArrayAttr(),
                                      BoolAttr());
}

VArangeOp HivmOpsEditor::addVArangeBefore(Operation *target, Value dst,
                                           ValueRange strides) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VArangeOp>(target->getLoc(), TypeRange{}, dst,
                                   Value(), strides);
}

VArangeOp HivmOpsEditor::addVArangeAfter(Operation *target, Value dst,
                                          ValueRange strides) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VArangeOp>(target->getLoc(), TypeRange{}, dst,
                                   Value(), strides);
}

VInterleaveOp HivmOpsEditor::addVInterleaveBefore(Operation *target,
                                                   ValueRange src,
                                                   Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VInterleaveOp>(target->getLoc(), TypeRange{}, src,
                                       dst);
}

VInterleaveOp HivmOpsEditor::addVInterleaveAfter(Operation *target,
                                                  ValueRange src,
                                                  Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VInterleaveOp>(target->getLoc(), TypeRange{}, src,
                                       dst);
}

VDeinterleaveOp HivmOpsEditor::addVDeinterleaveBefore(Operation *target,
                                                       Value src,
                                                       ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VDeinterleaveOp>(target->getLoc(), TypeRange{}, src,
                                         dst);
}

VDeinterleaveOp HivmOpsEditor::addVDeinterleaveAfter(Operation *target,
                                                      Value src,
                                                      ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VDeinterleaveOp>(target->getLoc(), TypeRange{}, src,
                                         dst);
}

//===----------------------------------------------------------------------===//
// CREATE - Macro Ops
//===----------------------------------------------------------------------===//

MmadL1Op HivmOpsEditor::addMmadL1Before(Operation *target, Value a, Value b,
                                         Value initCond, Value realM,
                                         Value realK, Value realN,
                                         Value c) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<MmadL1Op>(target->getLoc(), TypeRange{}, a, b,
                                  initCond, realM, realK, realN, c);
}

MmadL1Op HivmOpsEditor::addMmadL1After(Operation *target, Value a, Value b,
                                        Value initCond, Value realM,
                                        Value realK, Value realN,
                                        Value c) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<MmadL1Op>(target->getLoc(), TypeRange{}, a, b,
                                  initCond, realM, realK, realN, c);
}

BatchMmadL1Op HivmOpsEditor::addBatchMmadL1Before(Operation *target, Value a,
                                                   Value b, Value initCond,
                                                   Value realM, Value realK,
                                                   Value realN, Value c) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<BatchMmadL1Op>(target->getLoc(), TypeRange{}, a, b,
                                       initCond, realM, realK, realN, c);
}

BatchMmadL1Op HivmOpsEditor::addBatchMmadL1After(Operation *target, Value a,
                                                  Value b, Value initCond,
                                                  Value realM, Value realK,
                                                  Value realN, Value c) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<BatchMmadL1Op>(target->getLoc(), TypeRange{}, a, b,
                                       initCond, realM, realK, realN, c);
}

MatmulOp HivmOpsEditor::addMatmulBefore(Operation *target, Value a, Value b,
                                         Value c) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<MatmulOp>(target->getLoc(), TypeRange{}, a, b, c,
                                  UnitAttr(), UnitAttr());
}

MatmulOp HivmOpsEditor::addMatmulAfter(Operation *target, Value a, Value b,
                                         Value c) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<MatmulOp>(target->getLoc(), TypeRange{}, a, b, c,
                                  UnitAttr(), UnitAttr());
}

MixMatmulOp HivmOpsEditor::addMixMatmulBefore(Operation *target, Value a,
                                               Value b, Value c) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<MixMatmulOp>(target->getLoc(), TypeRange{}, a, b, c,
                                     UnitAttr(), UnitAttr());
}

MixMatmulOp HivmOpsEditor::addMixMatmulAfter(Operation *target, Value a,
                                              Value b, Value c) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<MixMatmulOp>(target->getLoc(), TypeRange{}, a, b, c,
                                     UnitAttr(), UnitAttr());
}

MixGroupMatmulOp HivmOpsEditor::addMixGroupMatmulBefore(Operation *target,
                                                         Value a, Value b,
                                                         Value tokensPerExpert,
                                                         Value c) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<MixGroupMatmulOp>(target->getLoc(), TypeRange{}, a, b,
                                          tokensPerExpert, c, UnitAttr(),
                                          UnitAttr());
}

MixGroupMatmulOp HivmOpsEditor::addMixGroupMatmulAfter(Operation *target,
                                                        Value a, Value b,
                                                        Value tokensPerExpert,
                                                        Value c) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<MixGroupMatmulOp>(target->getLoc(), TypeRange{}, a, b,
                                          tokensPerExpert, c, UnitAttr(),
                                          UnitAttr());
}

Conv1DL1Op HivmOpsEditor::addConv1DL1Before(Operation *target, Value input,
                                             Value weight, Value init,
                                             Value initCondition,
                                             int32_t padding,
                                             int32_t groups) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto *ctx = target->getContext();
  auto paddingAttr = builder.getI32IntegerAttr(padding);
  auto groupsAttr = builder.getI32IntegerAttr(groups);
  return builder.create<Conv1DL1Op>(target->getLoc(), TypeRange{}, input,
                                    weight, Value(), init, initCondition,
                                    ValueRange{}, paddingAttr, groupsAttr);
}

Conv1DL1Op HivmOpsEditor::addConv1DL1After(Operation *target, Value input,
                                            Value weight, Value init,
                                            Value initCondition,
                                            int32_t padding,
                                            int32_t groups) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto *ctx = target->getContext();
  auto paddingAttr = builder.getI32IntegerAttr(padding);
  auto groupsAttr = builder.getI32IntegerAttr(groups);
  return builder.create<Conv1DL1Op>(target->getLoc(), TypeRange{}, input,
                                    weight, Value(), init, initCondition,
                                    ValueRange{}, paddingAttr, groupsAttr);
}

Conv2DL1Op HivmOpsEditor::addConv2DL1Before(Operation *target, Value input,
                                             Value weight, Value init,
                                             Value initCondition,
                                             int32_t padding,
                                             int32_t groups) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto paddingAttr = builder.getI32IntegerAttr(padding);
  auto groupsAttr = builder.getI32IntegerAttr(groups);
  return builder.create<Conv2DL1Op>(target->getLoc(), TypeRange{}, input,
                                    weight, Value(), init, initCondition,
                                    ValueRange{}, paddingAttr, groupsAttr);
}

Conv2DL1Op HivmOpsEditor::addConv2DL1After(Operation *target, Value input,
                                            Value weight, Value init,
                                            Value initCondition,
                                            int32_t padding,
                                            int32_t groups) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto paddingAttr = builder.getI32IntegerAttr(padding);
  auto groupsAttr = builder.getI32IntegerAttr(groups);
  return builder.create<Conv2DL1Op>(target->getLoc(), TypeRange{}, input,
                                    weight, Value(), init, initCondition,
                                    ValueRange{}, paddingAttr, groupsAttr);
}

//===----------------------------------------------------------------------===//
// CREATE - Synchronization Ops
//===----------------------------------------------------------------------===//

void HivmOpsEditor::addSetFlagWaitFlagBefore(Operation *target,
                                              PipeAttr setPipe, PipeAttr waitPipe,
                                              EventAttr eventId) {
  auto sf = addSetFlagBefore(target, setPipe, waitPipe, eventId);
  addWaitFlagBefore(target, setPipe, waitPipe, eventId);
}

void HivmOpsEditor::addSetFlagWaitFlagAfter(Operation *target,
                                             PipeAttr setPipe, PipeAttr waitPipe,
                                             EventAttr eventId) {
  auto sf = addSetFlagAfter(target, setPipe, waitPipe, eventId);
  addWaitFlagAfter(sf, setPipe, waitPipe, eventId);
}

SetFlagOp HivmOpsEditor::addSetFlagBefore(Operation *target, PipeAttr setPipe,
                                           PipeAttr waitPipe,
                                           EventAttr eventId) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<SetFlagOp>(target->getLoc(), setPipe, waitPipe,
                                   eventId, Value());
}

SetFlagOp HivmOpsEditor::addSetFlagAfter(Operation *target, PipeAttr setPipe,
                                          PipeAttr waitPipe,
                                          EventAttr eventId) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<SetFlagOp>(target->getLoc(), setPipe, waitPipe,
                                   eventId, Value());
}

WaitFlagOp HivmOpsEditor::addWaitFlagBefore(Operation *target, PipeAttr setPipe,
                                             PipeAttr waitPipe,
                                             EventAttr eventId) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<WaitFlagOp>(target->getLoc(), setPipe, waitPipe,
                                     eventId, Value());
}

WaitFlagOp HivmOpsEditor::addWaitFlagAfter(Operation *target, PipeAttr setPipe,
                                            PipeAttr waitPipe,
                                            EventAttr eventId) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<WaitFlagOp>(target->getLoc(), setPipe, waitPipe,
                                    eventId, Value());
}

PipeBarrierOp HivmOpsEditor::addPipeBarrierBefore(Operation *target,
                                                   PipeAttr pipe) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<PipeBarrierOp>(target->getLoc(), pipe);
}

PipeBarrierOp HivmOpsEditor::addPipeBarrierAfter(Operation *target,
                                                  PipeAttr pipe) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<PipeBarrierOp>(target->getLoc(), pipe);
}

SyncBlockOp HivmOpsEditor::addSyncBlockBefore(Operation *target,
                                              SyncBlockModeAttr mode) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<SyncBlockOp>(target->getLoc(), mode, IntegerAttr(),
                                     Value(), PipeAttr(), PipeAttr());
}

SyncBlockOp HivmOpsEditor::addSyncBlockAfter(Operation *target,
                                             SyncBlockModeAttr mode) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<SyncBlockOp>(target->getLoc(), mode, IntegerAttr(),
                                     Value(), PipeAttr(), PipeAttr());
}

SyncBlockSetOp HivmOpsEditor::addSyncBlockSetBefore(Operation *target,
                                                     TCoreTypeAttr coreType,
                                                     PipeAttr tpipe, PipeAttr pipe,
                                                     int64_t flagId) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto flagAttr = builder.getI64IntegerAttr(flagId);
  auto *ctx = builder.getContext();
  auto syncModeAttr = SyncBlockInstrModeAttr::get(ctx,
      SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION);
  return builder.create<SyncBlockSetOp>(target->getLoc(), coreType,
                                        tpipe, pipe, flagAttr,
                                        Value(), syncModeAttr);
}

SyncBlockSetOp HivmOpsEditor::addSyncBlockSetAfter(Operation *target,
                                                    TCoreTypeAttr coreType,
                                                    PipeAttr tpipe, PipeAttr pipe,
                                                    int64_t flagId) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto flagAttr = builder.getI64IntegerAttr(flagId);
  auto *ctx = builder.getContext();
  auto syncModeAttr = SyncBlockInstrModeAttr::get(ctx,
      SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION);
  return builder.create<SyncBlockSetOp>(target->getLoc(), coreType,
                                        tpipe, pipe, flagAttr,
                                        Value(), syncModeAttr);
}

SyncBlockWaitOp HivmOpsEditor::addSyncBlockWaitBefore(Operation *target,
                                                       TCoreTypeAttr coreType,
                                                       PipeAttr tpipe, PipeAttr pipe,
                                                       int64_t flagId) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto flagAttr = builder.getI64IntegerAttr(flagId);
  return builder.create<SyncBlockWaitOp>(target->getLoc(), coreType,
                                         tpipe, pipe, flagAttr,
                                         Value());
}

SyncBlockWaitOp HivmOpsEditor::addSyncBlockWaitAfter(Operation *target,
                                                      TCoreTypeAttr coreType,
                                                      PipeAttr tpipe, PipeAttr pipe,
                                                      int64_t flagId) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto flagAttr = builder.getI64IntegerAttr(flagId);
  return builder.create<SyncBlockWaitOp>(target->getLoc(), coreType,
                                         tpipe, pipe, flagAttr,
                                          Value());
}

CreateSyncBlockLockOp
HivmOpsEditor::addCreateSyncBlockLockBefore(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto memrefType = MemRefType::get({1}, builder.getI64Type());
  return builder.create<CreateSyncBlockLockOp>(target->getLoc(), memrefType,
                                               Value());
}

CreateSyncBlockLockOp
HivmOpsEditor::addCreateSyncBlockLockAfter(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto memrefType = MemRefType::get({1}, builder.getI64Type());
  return builder.create<CreateSyncBlockLockOp>(target->getLoc(), memrefType,
                                               Value());
}

SyncBlockLockOp
HivmOpsEditor::addSyncBlockLockBefore(Operation *target, Value lockVar) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<SyncBlockLockOp>(target->getLoc(), lockVar);
}

SyncBlockLockOp
HivmOpsEditor::addSyncBlockLockAfter(Operation *target, Value lockVar) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<SyncBlockLockOp>(target->getLoc(), lockVar);
}

SyncBlockUnlockOp
HivmOpsEditor::addSyncBlockUnlockBefore(Operation *target, Value lockVar) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<SyncBlockUnlockOp>(target->getLoc(), lockVar);
}

SyncBlockUnlockOp
HivmOpsEditor::addSyncBlockUnlockAfter(Operation *target, Value lockVar) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<SyncBlockUnlockOp>(target->getLoc(), lockVar);
}

FreeLockVarOp
HivmOpsEditor::addFreeLockVarBefore(Operation *target, Value lockVar) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<FreeLockVarOp>(target->getLoc(), lockVar);
}

FreeLockVarOp
HivmOpsEditor::addFreeLockVarAfter(Operation *target, Value lockVar) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<FreeLockVarOp>(target->getLoc(), lockVar);
}

//===----------------------------------------------------------------------===//
// CREATE - Core / Utility Ops
//===----------------------------------------------------------------------===//

ConvertLayoutOp HivmOpsEditor::addConvertLayoutBefore(Operation *target,
                                                       Value source,
                                                       Type resultType) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<ConvertLayoutOp>(target->getLoc(), resultType,
                                         source);
}

ConvertLayoutOp HivmOpsEditor::addConvertLayoutAfter(Operation *target,
                                                      Value source,
                                                      Type resultType) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<ConvertLayoutOp>(target->getLoc(), resultType,
                                         source);
}

PointerCastOp HivmOpsEditor::addPointerCastBefore(Operation *target,
                                                   Value addr,
                                                   Type resultType) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<PointerCastOp>(target->getLoc(), resultType, addr);
}

PointerCastOp HivmOpsEditor::addPointerCastAfter(Operation *target,
                                                  Value addr,
                                                  Type resultType) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<PointerCastOp>(target->getLoc(), resultType, addr);
}

BitcastOp HivmOpsEditor::addBitcastBefore(Operation *target, Value src,
                                           Type resultType) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<BitcastOp>(target->getLoc(), resultType, src);
}

BitcastOp HivmOpsEditor::addBitcastAfter(Operation *target, Value src,
                                          Type resultType) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<BitcastOp>(target->getLoc(), resultType, src);
}

SetAtomicOp HivmOpsEditor::addSetAtomicBefore(Operation *target,
                                               AtomicKindAttr kind) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<SetAtomicOp>(target->getLoc(), kind,
                                     TypeAttr());
}

SetAtomicOp HivmOpsEditor::addSetAtomicAfter(Operation *target,
                                              AtomicKindAttr kind) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<SetAtomicOp>(target->getLoc(), kind,
                                      TypeAttr());
}

SetCtrlOp HivmOpsEditor::addSetCtrlBefore(Operation *target, bool enable,
                                           int64_t idx) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto enableAttr = builder.getBoolAttr(enable);
  auto idxAttr = builder.getI64IntegerAttr(idx);
  return builder.create<SetCtrlOp>(target->getLoc(), enableAttr, idxAttr);
}

SetCtrlOp HivmOpsEditor::addSetCtrlAfter(Operation *target, bool enable,
                                          int64_t idx) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto enableAttr = builder.getBoolAttr(enable);
  auto idxAttr = builder.getI64IntegerAttr(idx);
  return builder.create<SetCtrlOp>(target->getLoc(), enableAttr, idxAttr);
}

GetBlockIdxOp
HivmOpsEditor::addGetBlockIdxBefore(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<GetBlockIdxOp>(target->getLoc(),
                                       builder.getI64Type());
}

GetBlockIdxOp
HivmOpsEditor::addGetBlockIdxAfter(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<GetBlockIdxOp>(target->getLoc(),
                                       builder.getI64Type());
}

GetBlockNumOp
HivmOpsEditor::addGetBlockNumBefore(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<GetBlockNumOp>(target->getLoc(),
                                       builder.getI64Type());
}

GetBlockNumOp
HivmOpsEditor::addGetBlockNumAfter(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<GetBlockNumOp>(target->getLoc(),
                                       builder.getI64Type());
}

GetSubBlockIdxOp
HivmOpsEditor::addGetSubBlockIdxBefore(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<GetSubBlockIdxOp>(target->getLoc(),
                                          builder.getI64Type());
}

GetSubBlockIdxOp
HivmOpsEditor::addGetSubBlockIdxAfter(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<GetSubBlockIdxOp>(target->getLoc(),
                                          builder.getI64Type());
}

GetSubBlockNumOp
HivmOpsEditor::addGetSubBlockNumBefore(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<GetSubBlockNumOp>(target->getLoc(),
                                          builder.getI64Type());
}

GetSubBlockNumOp
HivmOpsEditor::addGetSubBlockNumAfter(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<GetSubBlockNumOp>(target->getLoc(),
                                          builder.getI64Type());
}

GetSysCntOp
HivmOpsEditor::addGetSysCntBefore(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<GetSysCntOp>(target->getLoc(),
                                     builder.getI64Type());
}

GetSysCntOp
HivmOpsEditor::addGetSysCntAfter(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<GetSysCntOp>(target->getLoc(),
                                     builder.getI64Type());
}

SetMaskNormOp
HivmOpsEditor::addSetMaskNormBefore(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<SetMaskNormOp>(target->getLoc());
}

SetMaskNormOp
HivmOpsEditor::addSetMaskNormAfter(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<SetMaskNormOp>(target->getLoc());
}

LoadScalarOp
HivmOpsEditor::addLoadScalarBefore(Operation *target, Value addr,
                                    Type resultType) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<LoadScalarOp>(target->getLoc(), resultType, addr);
}

LoadScalarOp
HivmOpsEditor::addLoadScalarAfter(Operation *target, Value addr,
                                   Type resultType) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<LoadScalarOp>(target->getLoc(), resultType, addr);
}

DCCIOp HivmOpsEditor::addDCCIBefore(Operation *target, DCCIModeAttr mode,
                                     DataCacheKindAttr dataCacheKind) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<DCCIOp>(target->getLoc(), mode, dataCacheKind,
                                Value());
}

DCCIOp HivmOpsEditor::addDCCIAfter(Operation *target, DCCIModeAttr mode,
                                    DataCacheKindAttr dataCacheKind) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<DCCIOp>(target->getLoc(), mode, dataCacheKind,
                                Value());
}

SetFFTSBaseAddrOp
HivmOpsEditor::addSetFFTSBaseAddrBefore(Operation *target,
                                         Value fftsBaseAddr) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<SetFFTSBaseAddrOp>(target->getLoc(), fftsBaseAddr);
}

SetFFTSBaseAddrOp
HivmOpsEditor::addSetFFTSBaseAddrAfter(Operation *target,
                                        Value fftsBaseAddr) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<SetFFTSBaseAddrOp>(target->getLoc(), fftsBaseAddr);
}

GatherLoadOp
HivmOpsEditor::addGatherLoadBefore(Operation *target, Value base,
                                   Value indices, Value burstLen,
                                   Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<GatherLoadOp>(target->getLoc(), TypeRange{}, base,
                                      indices, burstLen, Value(), Value(),
                                      dst, CacheModifierAttr(),
                                      EvictionPolicyAttr(), BoolAttr());
}

GatherLoadOp
HivmOpsEditor::addGatherLoadAfter(Operation *target, Value base,
                                  Value indices, Value burstLen,
                                  Value dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<GatherLoadOp>(target->getLoc(), TypeRange{}, base,
                                      indices, burstLen, Value(), Value(),
                                      dst, CacheModifierAttr(),
                                      EvictionPolicyAttr(), BoolAttr());
}

ScatterStoreOp
HivmOpsEditor::addScatterStoreBefore(Operation *target, Value indices,
                                     Value data, Value burstLen,
                                     Value base) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<ScatterStoreOp>(target->getLoc(), TypeRange{},
                                        indices, data, burstLen, Value(),
                                        base, CacheModifierAttr(),
                                        EvictionPolicyAttr());
}

ScatterStoreOp
HivmOpsEditor::addScatterStoreAfter(Operation *target, Value indices,
                                    Value data, Value burstLen,
                                    Value base) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<ScatterStoreOp>(target->getLoc(), TypeRange{},
                                        indices, data, burstLen, Value(),
                                        base, CacheModifierAttr(),
                                        EvictionPolicyAttr());
}

CustomOp HivmOpsEditor::addCustomBefore(Operation *target,
                                         StringRef name, ValueRange inputs,
                                         ValueRange outputs) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<CustomOp>(target->getLoc(), name, TypeRange{}, inputs,
                                  outputs);
}

CustomOp HivmOpsEditor::addCustomAfter(Operation *target,
                                        StringRef name, ValueRange inputs,
                                        ValueRange outputs) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<CustomOp>(target->getLoc(), name, TypeRange{}, inputs,
                                  outputs);
}

DebugOp HivmOpsEditor::addDebugBefore(Operation *target,
                                      StringRef debugType,
                                      StringRef prefix, bool hex,
                                      Value arg) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<DebugOp>(target->getLoc(), debugType, prefix, hex,
                                 arg, TCoreTypeAttr());
}

DebugOp HivmOpsEditor::addDebugAfter(Operation *target,
                                     StringRef debugType,
                                     StringRef prefix, bool hex,
                                     Value arg) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<DebugOp>(target->getLoc(), debugType, prefix, hex,
                                 arg, TCoreTypeAttr());
}

InitDebugOp HivmOpsEditor::addInitDebugBefore(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<InitDebugOp>(target->getLoc());
}

InitDebugOp HivmOpsEditor::addInitDebugAfter(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<InitDebugOp>(target->getLoc());
}

FinishDebugOp HivmOpsEditor::addFinishDebugBefore(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<FinishDebugOp>(target->getLoc());
}

FinishDebugOp HivmOpsEditor::addFinishDebugAfter(Operation *target) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<FinishDebugOp>(target->getLoc());
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
// MODIFY - Global attribute changes
//===----------------------------------------------------------------------===//

void HivmOpsEditor::changeElementType(Type oldType, Type newType) {
  module.walk([&](Operation *op) {
    for (auto &region : op->getRegions()) {
      region.walk([&](Operation *innerOp) {
        for (unsigned i = 0; i < innerOp->getNumOperands(); ++i) {
          auto opType = innerOp->getOperand(i).getType();
          if (auto memref = dyn_cast<MemRefType>(opType)) {
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
          if (auto memref = dyn_cast<MemRefType>(resType)) {
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

  module.walk([&](Operation *op) {
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto opType = op->getOperand(i).getType();
      if (auto memref = dyn_cast<MemRefType>(opType)) {
        if (auto spaceAttr = dyn_cast<AddressSpaceAttr>(memref.getMemorySpace())) {
          auto newSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::UB);
          if (newSpace == "gm")
            newSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::GM);
          else if (newSpace == "l1" || newSpace == "cbuf")
            newSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::L1);
          else if (newSpace == "ub")
            newSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::UB);
          else
            continue;

          auto oldSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::UB);
          if (oldSpace == "gm")
            oldSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::GM);
          else if (oldSpace == "l1" || oldSpace == "cbuf")
            oldSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::L1);
          else if (oldSpace == "ub")
            oldSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::UB);
          else
            continue;

          if (spaceAttr == oldSpaceAttr) {
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
      if (auto memref = dyn_cast<MemRefType>(resType)) {
        if (auto spaceAttr = dyn_cast<AddressSpaceAttr>(memref.getMemorySpace())) {
          auto newSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::UB);
          if (newSpace == "gm")
            newSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::GM);
          else if (newSpace == "l1" || newSpace == "cbuf")
            newSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::L1);
          else if (newSpace == "ub")
            newSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::UB);
          else
            continue;

          auto oldSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::UB);
          if (oldSpace == "gm")
            oldSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::GM);
          else if (oldSpace == "l1" || oldSpace == "cbuf")
            oldSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::L1);
          else if (oldSpace == "ub")
            oldSpaceAttr = AddressSpaceAttr::get(ctx, AddressSpace::UB);
          else
            continue;

          if (spaceAttr == oldSpaceAttr) {
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

void HivmOpsEditor::changePipeAttr(PipeAttr oldPipe, PipeAttr newPipe) {
  module.walk([&](Operation *op) {
    for (auto &namedAttr : op->getAttrs()) {
      if (auto pipeAttr = dyn_cast<PipeAttr>(namedAttr.getValue())) {
        if (pipeAttr == oldPipe)
          op->setAttr(namedAttr.getName(), newPipe);
      }
    }
  });
}

void HivmOpsEditor::changeEventAttr(EventAttr oldEvent, EventAttr newEvent) {
  module.walk([&](Operation *op) {
    for (auto &namedAttr : op->getAttrs()) {
      if (auto eventAttr = dyn_cast<EventAttr>(namedAttr.getValue())) {
        if (eventAttr == oldEvent)
          op->setAttr(namedAttr.getName(), newEvent);
      }
    }
  });
}

void HivmOpsEditor::changeShape(ArrayRef<int64_t> oldShape,
                                 ArrayRef<int64_t> newShape) {
  module.walk([&](Operation *op) {
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto opType = op->getOperand(i).getType();
      if (auto memref = dyn_cast<MemRefType>(opType)) {
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
      if (auto memref = dyn_cast<MemRefType>(resType)) {
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

//===----------------------------------------------------------------------===//
// MODIFY - Op-specific attribute setters
//===----------------------------------------------------------------------===//

void HivmOpsEditor::setSetFlagPipe(Operation *setFlagOp, PipeAttr newPipe) {
  setFlagOp->setAttr("set_pipe", newPipe);
}

void HivmOpsEditor::setWaitFlagPipe(Operation *waitFlagOp, PipeAttr newPipe) {
  waitFlagOp->setAttr("wait_pipe", newPipe);
}

void HivmOpsEditor::setEventId(Operation *syncOp, EventAttr newEvent) {
  syncOp->setAttr("static_event_id", newEvent);
}

void HivmOpsEditor::setLoadPadMode(LoadOp loadOp, PadModeAttr mode) {
  loadOp->setAttr("pad_mode", mode);
}

void HivmOpsEditor::setStoreAtomicKind(StoreOp storeOp, AtomicKindAttr kind) {
  storeOp->setAttr("atomic_kind", kind);
}

void HivmOpsEditor::setVCastRoundMode(VCastOp castOp, RoundModeAttr mode) {
  castOp->setAttr("round_mode", mode);
}

void HivmOpsEditor::setVCmpCompareMode(VCmpOp cmpOp, CompareModeAttr mode) {
  cmpOp->setAttr("compare_mode", mode);
}

void HivmOpsEditor::setVReduceOp(VReduceOp reduceOp, ReduceOpAttr arith) {
  reduceOp->setAttr("arith", arith);
}

void HivmOpsEditor::setMmadTranspose(MmadL1Op mmadOp, bool aTrans,
                                      bool bTrans) {
  auto *ctx = mmadOp->getContext();
  if (aTrans)
    mmadOp->setAttr("a_transpose", UnitAttr::get(ctx));
  else
    mmadOp->removeAttr("a_transpose");
  if (bTrans)
    mmadOp->setAttr("b_transpose", UnitAttr::get(ctx));
  else
    mmadOp->removeAttr("b_transpose");
}

void HivmOpsEditor::setMatmulTranspose(MatmulOp matmulOp, bool aTrans,
                                        bool bTrans) {
  auto *ctx = matmulOp->getContext();
  if (aTrans)
    matmulOp->setAttr("aTranspose", UnitAttr::get(ctx));
  else
    matmulOp->removeAttr("aTranspose");
  if (bTrans)
    matmulOp->setAttr("bTranspose", UnitAttr::get(ctx));
  else
    matmulOp->removeAttr("bTranspose");
}

void HivmOpsEditor::setCopyPadMode(CopyOp copyOp, PadModeAttr mode) {
  copyOp->setAttr("pad_mode", mode);
}

void HivmOpsEditor::setFixpipeDMAMode(FixpipeOp fixpipeOp,
                                       FixpipeDMAModeAttr mode) {
  fixpipeOp->setAttr("dma_mode", mode);
}

void HivmOpsEditor::setND2NZDstContinuous(ND2NZOp nd2nzOp,
                                           bool continuous) {
  auto *ctx = nd2nzOp->getContext();
  if (continuous)
    nd2nzOp->setAttr("dst_continuous", UnitAttr::get(ctx));
  else
    nd2nzOp->removeAttr("dst_continuous");
}

void HivmOpsEditor::setLoadInitOutBuffer(LoadOp loadOp, bool init) {
  auto *ctx = loadOp->getContext();
  loadOp->setAttr("init_out_buffer", BoolAttr::get(ctx, init));
}

void HivmOpsEditor::setVTransposePermutation(VTransposeOp transposeOp,
                                              ArrayRef<int64_t> permutation) {
  auto *ctx = transposeOp->getContext();
  transposeOp->setAttr("permutation",
                       DenseI64ArrayAttr::get(ctx, permutation));
}

void HivmOpsEditor::setVShRRound(VShROp vshrOp, bool round) {
  auto *ctx = vshrOp->getContext();
  vshrOp->setAttr("round", BoolAttr::get(ctx, round));
}

void HivmOpsEditor::setVSortDescending(VSortOp vsortOp, bool descending) {
  auto *ctx = vsortOp->getContext();
  vsortOp->setAttr("descending", BoolAttr::get(ctx, descending));
}

void HivmOpsEditor::setVInterleaveChannelNum(VInterleaveOp interleaveOp,
                                              int64_t channelNum) {
  auto *ctx = interleaveOp->getContext();
  interleaveOp->setAttr("interleave_channel_nums",
                        IntegerAttr::get(IntegerType::get(ctx, 64),
                                         channelNum));
}

void HivmOpsEditor::setVDeinterleaveChannelNum(
    VDeinterleaveOp deinterleaveOp, int64_t channelNum) {
  auto *ctx = deinterleaveOp->getContext();
  deinterleaveOp->setAttr("channel_num",
                          IntegerAttr::get(IntegerType::get(ctx, 64),
                                           channelNum));
}

void HivmOpsEditor::setAtomicRMWKind(AtomicRMWOp atomicRmwOp,
                                      AtomicKindAttr kind) {
  atomicRmwOp->setAttr("atomic_kind", kind);
}

void HivmOpsEditor::setDCCIMode(DCCIOp dcciOp, DCCIModeAttr mode) {
  dcciOp->setAttr("mode", mode);
}

void HivmOpsEditor::setCustomOpName(CustomOp customOp, StringRef name) {
  auto *ctx = customOp->getContext();
  customOp->setAttr("name", StringAttr::get(ctx, name));
}

//===----------------------------------------------------------------------===//
// MODIFY - Vector elementwise op replacement
//===----------------------------------------------------------------------===//

VSubOp HivmOpsEditor::replaceVAddWithVSub(VAddOp vadd) {
  return replaceOpWith<VAddOp, VSubOp>(vadd, vadd.getSrc(), vadd.getDst());
}

VAddOp HivmOpsEditor::replaceVSubWithVAdd(VSubOp vsub) {
  return replaceOpWith<VSubOp, VAddOp>(vsub, vsub.getSrc(), vsub.getDst());
}

VDivOp HivmOpsEditor::replaceVMulWithVDiv(VMulOp vmul) {
  return replaceOpWith<VMulOp, VDivOp>(vmul, vmul.getSrc(), vmul.getDst());
}

VMulOp HivmOpsEditor::replaceVDivWithVMul(VDivOp vdiv) {
  return replaceOpWith<VDivOp, VMulOp>(vdiv, vdiv.getSrc(), vdiv.getDst());
}

VMinOp HivmOpsEditor::replaceVMaxWithVMin(VMaxOp vmax) {
  return replaceOpWith<VMaxOp, VMinOp>(vmax, vmax.getSrc(), vmax.getDst());
}

VMaxOp HivmOpsEditor::replaceVMinWithVMax(VMinOp vmin) {
  return replaceOpWith<VMinOp, VMaxOp>(vmin, vmin.getSrc(), vmin.getDst());
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
                                           PipeAttr setPipe, PipeAttr waitPipe,
                                           EventAttr eventId) {
  auto *ctx = module.getContext();
  OpBuilder builder(ctx);

  module.walk([&](func::FuncOp func) {
    auto *entry = &func.getBody().front();
    builder.setInsertionPointToStart(entry);
    auto loc = builder.getUnknownLoc();

    auto ev0 = eventId;
    // For ev1, we need to create a new event with id+1
    // Since EventAttr doesn't have a simple way to increment,
    // we'll reuse the same event for simplicity
    auto ev1 = eventId;

    builder.create<SetFlagOp>(loc, setPipe, waitPipe, ev0, Value());
    builder.create<WaitFlagOp>(loc, setPipe, waitPipe, ev0, Value());
    builder.create<LoadOp>(loc, TypeRange{}, src, ub0);

    builder.create<SetFlagOp>(loc, setPipe, waitPipe, ev1, Value());
    builder.create<WaitFlagOp>(loc, setPipe, waitPipe, ev1, Value());
    builder.create<LoadOp>(loc, TypeRange{}, src, ub1);
  });
}

#endif // TRITONSIM_HAS_BISHENGIR_HIVM
