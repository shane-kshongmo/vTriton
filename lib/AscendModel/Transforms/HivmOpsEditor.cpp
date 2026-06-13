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
                                               AtomicKind kind) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto kindAttr = AtomicKindAttr::get(target->getContext(), kind);
  return builder.create<AtomicRMWOp>(target->getLoc(), TypeRange{}, src,
                                     dst, kindAttr);
}

AtomicRMWOp HivmOpsEditor::addAtomicRMWAfter(Operation *target, Value src,
                                              Value dst,
                                              AtomicKind kind) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto kindAttr = AtomicKindAttr::get(target->getContext(), kind);
  return builder.create<AtomicRMWOp>(target->getLoc(), TypeRange{}, src,
                                     dst, kindAttr);
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
  return builder.create<VShROp>(target->getLoc(), TypeRange{}, src, dst);
}

VShROp HivmOpsEditor::addVShRAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VShROp>(target->getLoc(), TypeRange{}, src, dst);
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

VMulExtUiOp HivmOpsEditor::addVMulExtUiBefore(Operation *target,
                                               ValueRange src,
                                               ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VMulExtUiOp>(target->getLoc(), TypeRange{}, src,
                                     dst);
}

VMulExtUiOp HivmOpsEditor::addVMulExtUiAfter(Operation *target,
                                              ValueRange src,
                                              ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VMulExtUiOp>(target->getLoc(), TypeRange{}, src,
                                     dst);
}

//===----------------------------------------------------------------------===//
// CREATE - Vector Ternary / Special Ops
//===----------------------------------------------------------------------===//

VSelOp HivmOpsEditor::addVSelBefore(Operation *target, ValueRange src,
                                    ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VSelOp>(target->getLoc(), TypeRange{}, src, dst);
}

VSelOp HivmOpsEditor::addVSelAfter(Operation *target, ValueRange src,
                                   ValueRange dst) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  return builder.create<VSelOp>(target->getLoc(), TypeRange{}, src, dst);
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
                                           ReduceOp arith,
                                           DenseI64ArrayAttr reduceDims) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  return builder.create<VReduceOp>(target->getLoc(), TypeRange{}, src, dst,
                                   arith, reduceDims);
}

VReduceOp HivmOpsEditor::addVReduceAfter(Operation *target, Value src,
                                          ValueRange dst,
                                          ReduceOp arith,
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

//===----------------------------------------------------------------------===//
// CREATE - Synchronization Ops
//===----------------------------------------------------------------------===//

void HivmOpsEditor::addSetFlagWaitFlagBefore(Operation *target,
                                              PIPE setPipe, PIPE waitPipe,
                                              EVENT_ID eventId) {
  auto sf = addSetFlagBefore(target, setPipe, waitPipe, eventId);
  addWaitFlagBefore(target, setPipe, waitPipe, eventId);
}

void HivmOpsEditor::addSetFlagWaitFlagAfter(Operation *target,
                                             PIPE setPipe, PIPE waitPipe,
                                             EVENT_ID eventId) {
  auto sf = addSetFlagAfter(target, setPipe, waitPipe, eventId);
  addWaitFlagAfter(sf, setPipe, waitPipe, eventId);
}

SetFlagOp HivmOpsEditor::addSetFlagBefore(Operation *target, PIPE setPipe,
                                           PIPE waitPipe,
                                           EVENT_ID eventId) {
  OpBuilder builder(target->getContext());
  auto *ctx = target->getContext();
  builder.setInsertionPoint(target);
  auto sp = PipeAttr::get(ctx, setPipe);
  auto wp = PipeAttr::get(ctx, waitPipe);
  auto ev = EventAttr::get(ctx, eventId);
  return builder.create<SetFlagOp>(target->getLoc(), sp, wp, ev, Value());
}

SetFlagOp HivmOpsEditor::addSetFlagAfter(Operation *target, PIPE setPipe,
                                          PIPE waitPipe,
                                          EVENT_ID eventId) {
  OpBuilder builder(target->getContext());
  auto *ctx = target->getContext();
  builder.setInsertionPointAfter(target);
  auto sp = PipeAttr::get(ctx, setPipe);
  auto wp = PipeAttr::get(ctx, waitPipe);
  auto ev = EventAttr::get(ctx, eventId);
  return builder.create<SetFlagOp>(target->getLoc(), sp, wp, ev, Value());
}

WaitFlagOp HivmOpsEditor::addWaitFlagBefore(Operation *target, PIPE setPipe,
                                             PIPE waitPipe,
                                             EVENT_ID eventId) {
  OpBuilder builder(target->getContext());
  auto *ctx = target->getContext();
  builder.setInsertionPoint(target);
  auto sp = PipeAttr::get(ctx, setPipe);
  auto wp = PipeAttr::get(ctx, waitPipe);
  auto ev = EventAttr::get(ctx, eventId);
  return builder.create<WaitFlagOp>(target->getLoc(), sp, wp, ev, Value());
}

WaitFlagOp HivmOpsEditor::addWaitFlagAfter(Operation *target, PIPE setPipe,
                                            PIPE waitPipe,
                                            EVENT_ID eventId) {
  OpBuilder builder(target->getContext());
  auto *ctx = target->getContext();
  builder.setInsertionPointAfter(target);
  auto sp = PipeAttr::get(ctx, setPipe);
  auto wp = PipeAttr::get(ctx, waitPipe);
  auto ev = EventAttr::get(ctx, eventId);
  return builder.create<WaitFlagOp>(target->getLoc(), sp, wp, ev, Value());
}

PipeBarrierOp HivmOpsEditor::addPipeBarrierBefore(Operation *target,
                                                   PIPE pipe) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto pipeAttr = PipeAttr::get(target->getContext(), pipe);
  return builder.create<PipeBarrierOp>(target->getLoc(), pipeAttr);
}

PipeBarrierOp HivmOpsEditor::addPipeBarrierAfter(Operation *target,
                                                  PIPE pipe) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto pipeAttr = PipeAttr::get(target->getContext(), pipe);
  return builder.create<PipeBarrierOp>(target->getLoc(), pipeAttr);
}

SyncBlockOp HivmOpsEditor::addSyncBlockBefore(Operation *target,
                                               SyncBlockMode mode) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto modeAttr = SyncBlockModeAttr::get(target->getContext(), mode);
  return builder.create<SyncBlockOp>(target->getLoc(), modeAttr,
                                     IntegerAttr(), Value(),
                                     PipeAttr(), PipeAttr());
}

SyncBlockOp HivmOpsEditor::addSyncBlockAfter(Operation *target,
                                              SyncBlockMode mode) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto modeAttr = SyncBlockModeAttr::get(target->getContext(), mode);
  return builder.create<SyncBlockOp>(target->getLoc(), modeAttr,
                                     IntegerAttr(), Value(),
                                     PipeAttr(), PipeAttr());
}

SyncBlockSetOp HivmOpsEditor::addSyncBlockSetBefore(Operation *target,
                                                     TCoreType coreType,
                                                     PIPE tpipe, PIPE pipe,
                                                     int64_t flagId) {
  OpBuilder builder(target->getContext());
  auto *ctx = target->getContext();
  builder.setInsertionPoint(target);
  auto coreAttr = TCoreTypeAttr::get(ctx, coreType);
  auto tpipeAttr = PipeAttr::get(ctx, tpipe);
  auto pipeAttr = PipeAttr::get(ctx, pipe);
  auto flagAttr = builder.getI64IntegerAttr(flagId);
  return builder.create<SyncBlockSetOp>(target->getLoc(), coreAttr,
                                        tpipeAttr, pipeAttr, flagAttr,
                                        Value(), SyncBlockInstrModeAttr());
}

SyncBlockSetOp HivmOpsEditor::addSyncBlockSetAfter(Operation *target,
                                                    TCoreType coreType,
                                                    PIPE tpipe, PIPE pipe,
                                                    int64_t flagId) {
  OpBuilder builder(target->getContext());
  auto *ctx = target->getContext();
  builder.setInsertionPointAfter(target);
  auto coreAttr = TCoreTypeAttr::get(ctx, coreType);
  auto tpipeAttr = PipeAttr::get(ctx, tpipe);
  auto pipeAttr = PipeAttr::get(ctx, pipe);
  auto flagAttr = builder.getI64IntegerAttr(flagId);
  return builder.create<SyncBlockSetOp>(target->getLoc(), coreAttr,
                                        tpipeAttr, pipeAttr, flagAttr,
                                        Value(), SyncBlockInstrModeAttr());
}

SyncBlockWaitOp HivmOpsEditor::addSyncBlockWaitBefore(Operation *target,
                                                       TCoreType coreType,
                                                       PIPE tpipe, PIPE pipe,
                                                       int64_t flagId) {
  OpBuilder builder(target->getContext());
  auto *ctx = target->getContext();
  builder.setInsertionPoint(target);
  auto coreAttr = TCoreTypeAttr::get(ctx, coreType);
  auto tpipeAttr = PipeAttr::get(ctx, tpipe);
  auto pipeAttr = PipeAttr::get(ctx, pipe);
  auto flagAttr = builder.getI64IntegerAttr(flagId);
  return builder.create<SyncBlockWaitOp>(target->getLoc(), coreAttr,
                                         tpipeAttr, pipeAttr, flagAttr,
                                         Value());
}

SyncBlockWaitOp HivmOpsEditor::addSyncBlockWaitAfter(Operation *target,
                                                      TCoreType coreType,
                                                      PIPE tpipe, PIPE pipe,
                                                      int64_t flagId) {
  OpBuilder builder(target->getContext());
  auto *ctx = target->getContext();
  builder.setInsertionPointAfter(target);
  auto coreAttr = TCoreTypeAttr::get(ctx, coreType);
  auto tpipeAttr = PipeAttr::get(ctx, tpipe);
  auto pipeAttr = PipeAttr::get(ctx, pipe);
  auto flagAttr = builder.getI64IntegerAttr(flagId);
  return builder.create<SyncBlockWaitOp>(target->getLoc(), coreAttr,
                                         tpipeAttr, pipeAttr, flagAttr,
                                         Value());
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
                                               AtomicKind kind) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPoint(target);
  auto kindAttr = AtomicKindAttr::get(target->getContext(), kind);
  return builder.create<SetAtomicOp>(target->getLoc(), kindAttr,
                                     TypeAttr());
}

SetAtomicOp HivmOpsEditor::addSetAtomicAfter(Operation *target,
                                              AtomicKind kind) {
  OpBuilder builder(target->getContext());
  builder.setInsertionPointAfter(target);
  auto kindAttr = AtomicKindAttr::get(target->getContext(), kind);
  return builder.create<SetAtomicOp>(target->getLoc(), kindAttr,
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

//===----------------------------------------------------------------------===//
// MODIFY - Op-specific attribute setters
//===----------------------------------------------------------------------===//

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

void HivmOpsEditor::setLoadPadMode(LoadOp loadOp, PadMode mode) {
  auto *ctx = loadOp->getContext();
  auto modeAttr = PadModeAttr::get(ctx, mode);
  loadOp->setAttr("pad_mode", modeAttr);
}

void HivmOpsEditor::setStoreAtomicKind(StoreOp storeOp, AtomicKind kind) {
  auto *ctx = storeOp->getContext();
  auto kindAttr = AtomicKindAttr::get(ctx, kind);
  storeOp->setAttr("atomic_kind", kindAttr);
}

void HivmOpsEditor::setVCastRoundMode(VCastOp castOp, RoundMode mode) {
  auto *ctx = castOp->getContext();
  auto modeAttr = RoundModeAttr::get(ctx, mode);
  castOp->setAttr("round_mode", modeAttr);
}

void HivmOpsEditor::setVCmpCompareMode(VCmpOp cmpOp, CompareMode mode) {
  auto *ctx = cmpOp->getContext();
  auto modeAttr = CmpModeAttr::get(ctx, mode);
  cmpOp->setAttr("compare_mode", modeAttr);
}

void HivmOpsEditor::setVReduceOp(VReduceOp reduceOp, ReduceOp arith) {
  auto *ctx = reduceOp->getContext();
  auto arithAttr = ReduceOpAttr::get(ctx, arith);
  reduceOp->setAttr("arith", arithAttr);
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
