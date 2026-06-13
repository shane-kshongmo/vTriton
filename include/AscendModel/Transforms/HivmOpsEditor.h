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

struct HivmOpInfo {
  unsigned index;
  std::string qualifiedName;
  Operation *op;
};

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
  // CREATE - Generic
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

  //===--------------------------------------------------------------------===//
  // CREATE - DMA Ops
  //===--------------------------------------------------------------------===//

  hivm::LoadOp addLoadBefore(Operation *target, Value src, Value dst);
  hivm::LoadOp addLoadAfter(Operation *target, Value src, Value dst);
  hivm::StoreOp addStoreBefore(Operation *target, Value src, Value dst);
  hivm::StoreOp addStoreAfter(Operation *target, Value src, Value dst);
  hivm::CopyOp addCopyBefore(Operation *target, Value src, Value dst);
  hivm::CopyOp addCopyAfter(Operation *target, Value src, Value dst);
  hivm::FixpipeOp addFixpipeBefore(Operation *target, Value src, Value dst);
  hivm::FixpipeOp addFixpipeAfter(Operation *target, Value src, Value dst);
  hivm::ND2NZOp addND2NZBefore(Operation *target, Value src, Value dst);
  hivm::ND2NZOp addND2NZAfter(Operation *target, Value src, Value dst);
  hivm::NZ2NDOp addNZ2NDBefore(Operation *target, Value src, Value dst);
  hivm::NZ2NDOp addNZ2NDAfter(Operation *target, Value src, Value dst);
  hivm::AtomicCasOp addAtomicCasBefore(Operation *target,
                                       ValueRange src, Value dst);
  hivm::AtomicCasOp addAtomicCasAfter(Operation *target,
                                      ValueRange src, Value dst);
  hivm::AtomicXchgOp addAtomicXchgBefore(Operation *target,
                                         Value src, Value dst);
  hivm::AtomicXchgOp addAtomicXchgAfter(Operation *target,
                                        Value src, Value dst);
  hivm::AtomicRMWOp addAtomicRMWBefore(Operation *target, Value src,
                                       Value dst,
                                       hivm::AtomicKind kind);
  hivm::AtomicRMWOp addAtomicRMWAfter(Operation *target, Value src,
                                      Value dst,
                                      hivm::AtomicKind kind);

  //===--------------------------------------------------------------------===//
  // CREATE - Vector Unary Ops
  //===--------------------------------------------------------------------===//

  hivm::VExpOp addVExpBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VExpOp addVExpAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VAbsOp addVAbsBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VAbsOp addVAbsAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VLnOp addVLnBefore(Operation *target, ValueRange src,
                           ValueRange dst);
  hivm::VLnOp addVLnAfter(Operation *target, ValueRange src,
                          ValueRange dst);
  hivm::VReluOp addVReluBefore(Operation *target, ValueRange src,
                               ValueRange dst);
  hivm::VReluOp addVReluAfter(Operation *target, ValueRange src,
                              ValueRange dst);
  hivm::VRsqrtOp addVRsqrtBefore(Operation *target, ValueRange src,
                                 ValueRange dst);
  hivm::VRsqrtOp addVRsqrtAfter(Operation *target, ValueRange src,
                                ValueRange dst);
  hivm::VSqrtOp addVSqrtBefore(Operation *target, ValueRange src,
                               ValueRange dst);
  hivm::VSqrtOp addVSqrtAfter(Operation *target, ValueRange src,
                              ValueRange dst);
  hivm::VTanhOp addVTanhBefore(Operation *target, ValueRange src,
                               ValueRange dst);
  hivm::VTanhOp addVTanhAfter(Operation *target, ValueRange src,
                              ValueRange dst);
  hivm::VSinOp addVSinBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VSinOp addVSinAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VCosOp addVCosBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VCosOp addVCosAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VErfOp addVErfBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VErfOp addVErfAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VRecOp addVRecBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VRecOp addVRecAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VNotOp addVNotBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VNotOp addVNotAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VCastOp addVCastBefore(Operation *target, ValueRange src,
                               ValueRange dst);
  hivm::VCastOp addVCastAfter(Operation *target, ValueRange src,
                              ValueRange dst);

  //===--------------------------------------------------------------------===//
  // CREATE - Vector Binary Ops
  //===--------------------------------------------------------------------===//

  hivm::VAddOp addVAddBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VAddOp addVAddAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VSubOp addVSubBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VSubOp addVSubAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VMulOp addVMulBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VMulOp addVMulAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VDivOp addVDivBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VDivOp addVDivAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VMaxOp addVMaxBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VMaxOp addVMaxAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VMinOp addVMinBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VMinOp addVMinAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VOrOp addVOrBefore(Operation *target, ValueRange src,
                           ValueRange dst);
  hivm::VOrOp addVOrAfter(Operation *target, ValueRange src,
                          ValueRange dst);
  hivm::VAndOp addVAndBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VAndOp addVAndAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VXorOp addVXorBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VXorOp addVXorAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VModOp addVModBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VModOp addVModAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VShLOp addVShLBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VShLOp addVShLAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VShROp addVShRBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VShROp addVShRAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VCmpOp addVCmpBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VCmpOp addVCmpAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VPowOp addVPowBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VPowOp addVPowAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VMulExtOp addVMulExtBefore(Operation *target, ValueRange src,
                                   ValueRange dst);
  hivm::VMulExtOp addVMulExtAfter(Operation *target, ValueRange src,
                                  ValueRange dst);
  hivm::VMulExtUiOp addVMulExtUiBefore(Operation *target, ValueRange src,
                                       ValueRange dst);
  hivm::VMulExtUiOp addVMulExtUiAfter(Operation *target, ValueRange src,
                                      ValueRange dst);

  //===--------------------------------------------------------------------===//
  // CREATE - Vector Ternary / Special Ops
  //===--------------------------------------------------------------------===//

  hivm::VSelOp addVSelBefore(Operation *target, ValueRange src,
                             ValueRange dst);
  hivm::VSelOp addVSelAfter(Operation *target, ValueRange src,
                            ValueRange dst);
  hivm::VBrcOp addVBrcBefore(Operation *target, Value src, Value dst);
  hivm::VBrcOp addVBrcAfter(Operation *target, Value src, Value dst);
  hivm::VReduceOp addVReduceBefore(Operation *target, Value src,
                                   ValueRange dst,
                                   hivm::ReduceOp arith,
                                   DenseI64ArrayAttr reduceDims);
  hivm::VReduceOp addVReduceAfter(Operation *target, Value src,
                                  ValueRange dst,
                                  hivm::ReduceOp arith,
                                  DenseI64ArrayAttr reduceDims);
  hivm::VConcatOp addVConcatBefore(Operation *target, int64_t dim,
                                   ValueRange src, Value dst);
  hivm::VConcatOp addVConcatAfter(Operation *target, int64_t dim,
                                  ValueRange src, Value dst);
  hivm::VFlipOp addVFlipBefore(Operation *target, Value src, Value dst,
                               int64_t flipAxis);
  hivm::VFlipOp addVFlipAfter(Operation *target, Value src, Value dst,
                              int64_t flipAxis);
  hivm::VPadOp addVPadBefore(Operation *target, Value src, Value dst,
                             Value padValue, ValueRange low,
                             ValueRange high);
  hivm::VPadOp addVPadAfter(Operation *target, Value src, Value dst,
                            Value padValue, ValueRange low,
                            ValueRange high);
  hivm::VGatherOp addVGatherBefore(Operation *target, Value src,
                                   Value indices, Value dst);
  hivm::VGatherOp addVGatherAfter(Operation *target, Value src,
                                  Value indices, Value dst);
  hivm::VGatherMaskOp addVGatherMaskBefore(Operation *target, Value src,
                                           Value mask, ValueRange dst);
  hivm::VGatherMaskOp addVGatherMaskAfter(Operation *target, Value src,
                                          Value mask, ValueRange dst);
  hivm::VCumsumOp addVCumsumBefore(Operation *target, Value src,
                                   Value dst,
                                   DenseI64ArrayAttr cumDims);
  hivm::VCumsumOp addVCumsumAfter(Operation *target, Value src,
                                  Value dst,
                                  DenseI64ArrayAttr cumDims);
  hivm::VCumprodOp addVCumprodBefore(Operation *target, Value src,
                                     Value dst,
                                     DenseI64ArrayAttr cumDims);
  hivm::VCumprodOp addVCumprodAfter(Operation *target, Value src,
                                    Value dst,
                                    DenseI64ArrayAttr cumDims);
  hivm::VSortOp addVSortBefore(Operation *target, Value src,
                               ValueRange dst, bool descending,
                               int64_t sortAxis);
  hivm::VSortOp addVSortAfter(Operation *target, Value src,
                              ValueRange dst, bool descending,
                              int64_t sortAxis);
  hivm::VMulextendedOp addVMulextendedBefore(Operation *target,
                                             ValueRange src,
                                             ValueRange dst);
  hivm::VMulextendedOp addVMulextendedAfter(Operation *target,
                                            ValueRange src,
                                            ValueRange dst);
  hivm::VTransposeOp addVTransposeBefore(Operation *target, Value src,
                                         Value dst);
  hivm::VTransposeOp addVTransposeAfter(Operation *target, Value src,
                                        Value dst);
  hivm::VArangeOp addVArangeBefore(Operation *target, Value dst,
                                   ValueRange strides);
  hivm::VArangeOp addVArangeAfter(Operation *target, Value dst,
                                  ValueRange strides);
  hivm::VInterleaveOp addVInterleaveBefore(Operation *target,
                                           ValueRange src, Value dst);
  hivm::VInterleaveOp addVInterleaveAfter(Operation *target,
                                          ValueRange src, Value dst);
  hivm::VDeinterleaveOp addVDeinterleaveBefore(Operation *target,
                                               Value src,
                                               ValueRange dst);
  hivm::VDeinterleaveOp addVDeinterleaveAfter(Operation *target,
                                              Value src,
                                              ValueRange dst);

  //===--------------------------------------------------------------------===//
  // CREATE - Macro Ops
  //===--------------------------------------------------------------------===//

  hivm::MmadL1Op addMmadL1Before(Operation *target, Value a, Value b,
                                  Value initCond, Value realM,
                                  Value realK, Value realN, Value c);
  hivm::MmadL1Op addMmadL1After(Operation *target, Value a, Value b,
                                 Value initCond, Value realM,
                                 Value realK, Value realN, Value c);
  hivm::BatchMmadL1Op addBatchMmadL1Before(Operation *target, Value a,
                                            Value b, Value initCond,
                                            Value realM, Value realK,
                                            Value realN, Value c);
  hivm::BatchMmadL1Op addBatchMmadL1After(Operation *target, Value a,
                                           Value b, Value initCond,
                                           Value realM, Value realK,
                                           Value realN, Value c);
  hivm::MatmulOp addMatmulBefore(Operation *target, Value a, Value b,
                                  Value c);
  hivm::MatmulOp addMatmulAfter(Operation *target, Value a, Value b,
                                 Value c);
  hivm::MixMatmulOp addMixMatmulBefore(Operation *target, Value a,
                                       Value b, Value c);
  hivm::MixMatmulOp addMixMatmulAfter(Operation *target, Value a,
                                      Value b, Value c);
  hivm::MixGroupMatmulOp addMixGroupMatmulBefore(Operation *target,
                                                  Value a, Value b,
                                                  Value tokensPerExpert,
                                                  Value c);
  hivm::MixGroupMatmulOp addMixGroupMatmulAfter(Operation *target,
                                                 Value a, Value b,
                                                 Value tokensPerExpert,
                                                 Value c);
  hivm::Conv1DL1Op addConv1DL1Before(Operation *target, Value input,
                                      Value weight, Value init,
                                      Value initCondition,
                                      int32_t padding, int32_t groups);
  hivm::Conv1DL1Op addConv1DL1After(Operation *target, Value input,
                                     Value weight, Value init,
                                     Value initCondition,
                                     int32_t padding, int32_t groups);
  hivm::Conv2DL1Op addConv2DL1Before(Operation *target, Value input,
                                       Value weight, Value init,
                                       Value initCondition,
                                       int32_t padding, int32_t groups);
  hivm::Conv2DL1Op addConv2DL1After(Operation *target, Value input,
                                      Value weight, Value init,
                                      Value initCondition,
                                      int32_t padding, int32_t groups);
  hivm::Conv3DL1Op addConv3DL1Before(Operation *target, Value input,
                                       Value weight, Value init,
                                       Value initCondition,
                                       int32_t padding, int32_t groups);
  hivm::Conv3DL1Op addConv3DL1After(Operation *target, Value input,
                                      Value weight, Value init,
                                      Value initCondition,
                                      int32_t padding, int32_t groups);

  //===--------------------------------------------------------------------===//
  // CREATE - Synchronization Ops
  //===--------------------------------------------------------------------===//

  void addSetFlagWaitFlagBefore(Operation *target, PIPE setPipe,
                                PIPE waitPipe, EVENT_ID eventId);
  void addSetFlagWaitFlagAfter(Operation *target, PIPE setPipe,
                               PIPE waitPipe, EVENT_ID eventId);
  hivm::SetFlagOp addSetFlagBefore(Operation *target, PIPE setPipe,
                                   PIPE waitPipe, EVENT_ID eventId);
  hivm::SetFlagOp addSetFlagAfter(Operation *target, PIPE setPipe,
                                  PIPE waitPipe, EVENT_ID eventId);
  hivm::WaitFlagOp addWaitFlagBefore(Operation *target, PIPE setPipe,
                                     PIPE waitPipe, EVENT_ID eventId);
  hivm::WaitFlagOp addWaitFlagAfter(Operation *target, PIPE setPipe,
                                    PIPE waitPipe, EVENT_ID eventId);
  hivm::PipeBarrierOp addPipeBarrierBefore(Operation *target, PIPE pipe);
  hivm::PipeBarrierOp addPipeBarrierAfter(Operation *target, PIPE pipe);
  hivm::SyncBlockOp addSyncBlockBefore(Operation *target,
                                       hivm::SyncBlockMode mode);
  hivm::SyncBlockOp addSyncBlockAfter(Operation *target,
                                      hivm::SyncBlockMode mode);
  hivm::SyncBlockSetOp addSyncBlockSetBefore(Operation *target,
                                             hivm::TCoreType coreType,
                                             PIPE tpipe, PIPE pipe,
                                             int64_t flagId);
  hivm::SyncBlockSetOp addSyncBlockSetAfter(Operation *target,
                                            hivm::TCoreType coreType,
                                            PIPE tpipe, PIPE pipe,
                                            int64_t flagId);
  hivm::SyncBlockWaitOp addSyncBlockWaitBefore(Operation *target,
                                               hivm::TCoreType coreType,
                                               PIPE tpipe, PIPE pipe,
                                               int64_t flagId);
  hivm::SyncBlockWaitOp addSyncBlockWaitAfter(Operation *target,
                                              hivm::TCoreType coreType,
                                              PIPE tpipe, PIPE pipe,
                                              int64_t flagId);
  hivm::CreateSyncBlockLockOp addCreateSyncBlockLockBefore(
      Operation *target);
  hivm::CreateSyncBlockLockOp addCreateSyncBlockLockAfter(
      Operation *target);
  hivm::SyncBlockLockOp addSyncBlockLockBefore(Operation *target,
                                               Value lockVar);
  hivm::SyncBlockLockOp addSyncBlockLockAfter(Operation *target,
                                              Value lockVar);
  hivm::SyncBlockUnlockOp addSyncBlockUnlockBefore(Operation *target,
                                                   Value lockVar);
  hivm::SyncBlockUnlockOp addSyncBlockUnlockAfter(Operation *target,
                                                  Value lockVar);
  hivm::FreeLockVarOp addFreeLockVarBefore(Operation *target,
                                           Value lockVar);
  hivm::FreeLockVarOp addFreeLockVarAfter(Operation *target,
                                          Value lockVar);

  //===--------------------------------------------------------------------===//
  // CREATE - Core / Utility Ops
  //===--------------------------------------------------------------------===//

  hivm::ConvertLayoutOp addConvertLayoutBefore(Operation *target,
                                               Value source,
                                               Type resultType);
  hivm::ConvertLayoutOp addConvertLayoutAfter(Operation *target,
                                              Value source,
                                              Type resultType);
  hivm::PointerCastOp addPointerCastBefore(Operation *target, Value addr,
                                           Type resultType);
  hivm::PointerCastOp addPointerCastAfter(Operation *target, Value addr,
                                          Type resultType);
  hivm::BitcastOp addBitcastBefore(Operation *target, Value src,
                                   Type resultType);
  hivm::BitcastOp addBitcastAfter(Operation *target, Value src,
                                  Type resultType);
  hivm::SetAtomicOp addSetAtomicBefore(Operation *target,
                                       hivm::AtomicKind kind);
  hivm::SetAtomicOp addSetAtomicAfter(Operation *target,
                                      hivm::AtomicKind kind);
  hivm::SetCtrlOp addSetCtrlBefore(Operation *target, bool enable,
                                   int64_t idx);
  hivm::SetCtrlOp addSetCtrlAfter(Operation *target, bool enable,
                                  int64_t idx);
  hivm::GetBlockIdxOp addGetBlockIdxBefore(Operation *target);
  hivm::GetBlockIdxOp addGetBlockIdxAfter(Operation *target);
  hivm::GetBlockNumOp addGetBlockNumBefore(Operation *target);
  hivm::GetBlockNumOp addGetBlockNumAfter(Operation *target);
  hivm::GetSubBlockIdxOp addGetSubBlockIdxBefore(Operation *target);
  hivm::GetSubBlockIdxOp addGetSubBlockIdxAfter(Operation *target);
  hivm::GetSubBlockNumOp addGetSubBlockNumBefore(Operation *target);
  hivm::GetSubBlockNumOp addGetSubBlockNumAfter(Operation *target);
  hivm::GetSysCntOp addGetSysCntBefore(Operation *target);
  hivm::GetSysCntOp addGetSysCntAfter(Operation *target);
  hivm::SetMaskNormOp addSetMaskNormBefore(Operation *target);
  hivm::SetMaskNormOp addSetMaskNormAfter(Operation *target);
  hivm::LoadScalarOp addLoadScalarBefore(Operation *target, Value addr,
                                          Type resultType);
  hivm::LoadScalarOp addLoadScalarAfter(Operation *target, Value addr,
                                         Type resultType);
  hivm::DCCIOp addDCCIBefore(Operation *target, hivm::DCCIMode mode,
                             hivm::DataCacheKind dataCacheKind);
  hivm::DCCIOp addDCCIAfter(Operation *target, hivm::DCCIMode mode,
                            hivm::DataCacheKind dataCacheKind);
  hivm::SetFFTSBaseAddrOp addSetFFTSBaseAddrBefore(Operation *target,
                                                   Value fftsBaseAddr);
  hivm::SetFFTSBaseAddrOp addSetFFTSBaseAddrAfter(Operation *target,
                                                  Value fftsBaseAddr);
  hivm::GatherLoadOp addGatherLoadBefore(Operation *target, Value base,
                                         Value indices, Value burstLen,
                                         Value dst);
  hivm::GatherLoadOp addGatherLoadAfter(Operation *target, Value base,
                                        Value indices, Value burstLen,
                                        Value dst);
  hivm::ScatterStoreOp addScatterStoreBefore(Operation *target,
                                             Value indices, Value data,
                                             Value burstLen, Value base);
  hivm::ScatterStoreOp addScatterStoreAfter(Operation *target,
                                            Value indices, Value data,
                                            Value burstLen, Value base);
  hivm::CustomOp addCustomBefore(Operation *target,
                                 llvm::StringRef name,
                                 ValueRange inputs, ValueRange outputs);
  hivm::CustomOp addCustomAfter(Operation *target,
                                llvm::StringRef name,
                                ValueRange inputs, ValueRange outputs);
  hivm::DebugOp addDebugBefore(Operation *target,
                               llvm::StringRef debugType,
                               llvm::StringRef prefix, bool hex,
                               Value arg);
  hivm::DebugOp addDebugAfter(Operation *target,
                              llvm::StringRef debugType,
                              llvm::StringRef prefix, bool hex,
                              Value arg);
  hivm::InitDebugOp addInitDebugBefore(Operation *target);
  hivm::InitDebugOp addInitDebugAfter(Operation *target);
  hivm::FinishDebugOp addFinishDebugBefore(Operation *target);
  hivm::FinishDebugOp addFinishDebugAfter(Operation *target);
  hivm::IndirectStoreOp addIndirectStoreBefore(Operation *target, Value dst,
                                               Value offsets, Value src);
  hivm::IndirectStoreOp addIndirectStoreAfter(Operation *target, Value dst,
                                              Value offsets, Value src);

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
  // MODIFY - Generic op replacement
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

  //===--------------------------------------------------------------------===//
  // MODIFY - Global attribute changes
  //===--------------------------------------------------------------------===//

  void changeElementType(Type oldType, Type newType);
  void changeMemorySpace(llvm::StringRef oldSpace, llvm::StringRef newSpace);
  void changePipeAttr(PIPE oldPipe, PIPE newPipe);
  void changeEventAttr(EVENT_ID oldEvent, EVENT_ID newEvent);
  void changeShape(ArrayRef<int64_t> oldShape, ArrayRef<int64_t> newShape);

  //===--------------------------------------------------------------------===//
  // MODIFY - Op-specific attribute setters
  //===--------------------------------------------------------------------===//

  void setSetFlagPipe(Operation *setFlagOp, PIPE newPipe);
  void setWaitFlagPipe(Operation *waitFlagOp, PIPE newPipe);
  void setEventId(Operation *syncOp, EVENT_ID newEvent);
  void setLoadPadMode(hivm::LoadOp loadOp, hivm::PadMode mode);
  void setStoreAtomicKind(hivm::StoreOp storeOp, hivm::AtomicKind kind);
  void setCopyPadMode(hivm::CopyOp copyOp, hivm::PadMode mode);
  void setFixpipeDMAMode(hivm::FixpipeOp fixpipeOp,
                         hivm::FixpipeDMAMode mode);
  void setND2NZDstContinuous(hivm::ND2NZOp nd2nzOp, bool continuous);
  void setLoadInitOutBuffer(hivm::LoadOp loadOp, bool init);
  void setVCastRoundMode(hivm::VCastOp castOp, hivm::RoundMode mode);
  void setVCmpCompareMode(hivm::VCmpOp cmpOp, hivm::CompareMode mode);
  void setVReduceOp(hivm::VReduceOp reduceOp, hivm::ReduceOp arith);
  void setVTransposePermutation(hivm::VTransposeOp transposeOp,
                                ArrayRef<int64_t> permutation);
  void setVShRRound(hivm::VShROp vshrOp, bool round);
  void setVSortDescending(hivm::VSortOp vsortOp, bool descending);
  void setVInterleaveChannelNum(hivm::VInterleaveOp interleaveOp,
                                int64_t channelNum);
  void setVDeinterleaveChannelNum(hivm::VDeinterleaveOp deinterleaveOp,
                                  int64_t channelNum);
  void setMmadTranspose(hivm::MmadL1Op mmadOp, bool aTrans, bool bTrans);
  void setMatmulTranspose(hivm::MatmulOp matmulOp, bool aTrans,
                          bool bTrans);
  void setAtomicRMWKind(hivm::AtomicRMWOp atomicRmwOp,
                        hivm::AtomicKind kind);
  void setDCCIMode(hivm::DCCIOp dcciOp, hivm::DCCIMode mode);
  void setCustomOpName(hivm::CustomOp customOp, llvm::StringRef name);

  //===--------------------------------------------------------------------===//
  // MODIFY - Vector elementwise op replacement
  //===--------------------------------------------------------------------===//

  hivm::VSubOp replaceVAddWithVSub(hivm::VAddOp vadd);
  hivm::VAddOp replaceVSubWithVAdd(hivm::VSubOp vsub);
  hivm::VDivOp replaceVMulWithVDiv(hivm::VMulOp vmul);
  hivm::VMulOp replaceVDivWithVMul(hivm::VDivOp vdiv);
  hivm::VMinOp replaceVMaxWithVMin(hivm::VMaxOp vmax);
  hivm::VMaxOp replaceVMinWithVMax(hivm::VMinOp vmin);

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
