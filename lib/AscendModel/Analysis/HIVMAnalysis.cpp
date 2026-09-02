//===- HIVMAnalysis.cpp - HIVM performance analysis ----------------------===//
//
// This implementation consumes HIVM through MLIR-native ingestion and
// schedules the resulting execution graph with either a static scheduler or
// a discrete-event simulator.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/HIVMAnalysis.h"

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#endif

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cmath>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <vector>

using namespace mlir::ascend;

namespace {

static std::string jsonEscape(llvm::StringRef value) {
  std::string escaped;
  escaped.reserve(value.size());
  constexpr char kHex[] = "0123456789abcdef";
  for (unsigned char c : value) {
    switch (c) {
    case '\"':
      escaped += "\\\"";
      break;
    case '\\':
      escaped += "\\\\";
      break;
    case '\b':
      escaped += "\\b";
      break;
    case '\f':
      escaped += "\\f";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      if (c < 0x20) {
        escaped += "\\u00";
        escaped += kHex[(c >> 4) & 0xf];
        escaped += kHex[c & 0xf];
      } else {
        escaped += static_cast<char>(c);
      }
      break;
    }
  }
  return escaped;
}

struct ParsedOp {
  HIVMOp op;
  std::string definedValue;
  std::vector<std::string> operandValues;
  std::vector<mlir::Value> mlirResults;
  std::vector<HIVMPipe> barrierPipes;
  std::string syncCoreType;
  std::string senderEvent;
  std::string receiverEvent;
  std::string eventId;
  mlir::Value syncIdValue;
  HIVMPipe senderPipe = HIVMPipe::Unknown;
  HIVMPipe receiverPipe = HIVMPipe::Unknown;
};

static int64_t estimateMmadL1MTE1Cycles(const ParsedOp &parsed,
                                        const HardwareConfig &config);

static int64_t estimateND2NZCycles(const ParsedOp &parsed,
                                   const HardwareConfig &config);
static void finalizeScheduledReport(HIVMAnalysisReport &report,
                                    const HardwareConfig &config);
static void finalizeDiscreteEventReport(HIVMAnalysisReport &report,
                                        const HardwareConfig &config);

static std::vector<HIVMOp> expandMacroOp(const ParsedOp &parsed,
                                         const HardwareConfig &config) {
  std::vector<HIVMOp> ops;
  llvm::StringRef name = parsed.op.opName;
  if (name == "mmadL1") {
    // MTE1 (L1 -> L0A/L0B) and Cube (compute on L0) are on different pipes
    // and overlap in the hardware pipeline.  Each gets its own duration.
    HIVMOp mte = parsed.op;
    mte.opName = "mmadL1.mte1";
    mte.pipe = HIVMPipe::MTE1;
    mte.duration = estimateMmadL1MTE1Cycles(parsed, config);
    mte.isSyncOp = false;
    mte.isBarrier = false;

    HIVMOp cube = parsed.op;
    cube.opName = "mmadL1.cube";
    cube.pipe = HIVMPipe::Cube;
    // Cube gets the full compute duration (startup + tile cycles), NOT the
    // leftover after subtracting MTE1.  The two pipes overlap.
    cube.duration = std::max<int64_t>(1, parsed.op.duration);
    cube.isSyncOp = false;
    cube.isBarrier = false;
    ops.push_back(std::move(mte));
    ops.push_back(std::move(cube));
    return ops;
  }

  if (name == "matmul" || name == "mix_matmul" || name == "mix_group_matmul") {
    int64_t preload = std::max<int64_t>(1, config.getMTE2StartupLatency());
    int64_t drain = std::max<int64_t>(1, config.getMTE3StartupLatency());
    int64_t compute = std::max<int64_t>(1, parsed.op.duration - preload - drain);

    HIVMOp mte2 = parsed.op;
    mte2.opName = name.str() + ".mte2";
    mte2.pipe = HIVMPipe::CubeMTE2;
    mte2.duration = preload;
    mte2.isSyncOp = false;
    mte2.isBarrier = false;

    HIVMOp cube = parsed.op;
    cube.opName = name.str() + ".cube";
    cube.pipe = HIVMPipe::Cube;
    cube.duration = compute;
    cube.isSyncOp = false;
    cube.isBarrier = false;

    HIVMOp mte3 = parsed.op;
    mte3.opName = name.str() + ".mte3";
    mte3.pipe = HIVMPipe::MTE3;
    mte3.duration = drain;
    mte3.isSyncOp = false;
    mte3.isBarrier = false;

    ops.push_back(std::move(mte2));
    ops.push_back(std::move(cube));
    ops.push_back(std::move(mte3));
    return ops;
  }

  ops.push_back(parsed.op);
  return ops;
}

static void attachSyncMetadata(ParsedOp &parsed) {
  parsed.op.senderPipe = parsed.senderPipe;
  parsed.op.receiverPipe = parsed.receiverPipe;
  parsed.op.eventId = parsed.eventId;
}

struct EventKey {
  HIVMPipe sender = HIVMPipe::Unknown;
  HIVMPipe receiver = HIVMPipe::Unknown;
  std::string eventId;

  bool operator<(const EventKey &other) const {
    return std::tie(sender, receiver, eventId) <
           std::tie(other.sender, other.receiver, other.eventId);
  }
};

struct EventInstanceKey {
  EventKey key;
  int64_t generation = 0;

  bool operator<(const EventInstanceKey &other) const {
    return std::tie(key.sender, key.receiver, key.eventId, generation) <
           std::tie(other.key.sender, other.key.receiver, other.key.eventId,
                    other.generation);
  }
};

struct LoopFrame {
  int braceDepth = 0;
  int64_t tripCount = 1;
};

struct TextLoopFrame {
  int braceDepth = 0;
  int64_t multiplier = 1;
};

struct AnalysisState {
  llvm::DenseMap<mlir::Value, int64_t> constants;
  llvm::DenseMap<mlir::Value, int64_t> boundValues;
  llvm::DenseMap<mlir::Value, size_t> valueProducers;
  llvm::DenseMap<mlir::Value, std::string> bufferRoots;
  std::map<std::string, int64_t> bufferSlots;
  std::map<std::string, int64_t> bufferVersions;
  std::map<EventKey, size_t> eventProducers;
  std::map<EventKey, int64_t> eventGenerations;
  std::map<HIVMPipe, size_t> latestPipeProducer;
  std::map<std::string, int64_t> argBindings;
};

static llvm::StringRef trim(llvm::StringRef s) { return s.trim(); }

static int64_t ceilDiv(int64_t num, int64_t den) {
  if (den <= 0)
    return 0;
  return (num + den - 1) / den;
}

/// Resolve an ambiguous core type (empty or "CUBE_OR_VECTOR") by inspecting the
/// enclosing func.func's name for AIC/AIV markers.
static llvm::StringRef resolveCoreTypeFromFunc(mlir::Operation *op,
                                               llvm::StringRef current) {
  if (current == "CUBE" || current == "AIC" || current == "VECTOR" ||
      current == "AIV")
    return current;
  if (auto parentFunc = op->getParentOfType<mlir::func::FuncOp>()) {
    llvm::StringRef funcName = parentFunc.getName();
    if (funcName.contains("aic") || funcName.contains("AIC") ||
        funcName.contains("cube"))
      return "CUBE";
    if (funcName.contains("aiv") || funcName.contains("AIV") ||
        funcName.contains("vector") || funcName.contains("mix"))
      return "VECTOR";
  }
  return current;
}

static bool isVectorDomainPipe(HIVMPipe pipe) {
  return pipe == HIVMPipe::Vector || pipe == HIVMPipe::VectorMTE2 ||
         pipe == HIVMPipe::MTE3;
}

static bool isCubeDomainPipe(HIVMPipe pipe) {
  return pipe == HIVMPipe::Cube || pipe == HIVMPipe::CubeMTE2 ||
         pipe == HIVMPipe::MTE1 || pipe == HIVMPipe::FixPipe;
}

static HIVMPipe disambiguateMTE2Pipe(HIVMPipe pipe, HIVMPipe peerPipe,
                                     llvm::StringRef coreType) {
  if (pipe != HIVMPipe::VectorMTE2)
    return pipe;
  if (isVectorDomainPipe(peerPipe))
    return HIVMPipe::VectorMTE2;
  if (isCubeDomainPipe(peerPipe))
    return HIVMPipe::CubeMTE2;
  if (coreType == "VECTOR" || coreType == "AIV")
    return HIVMPipe::VectorMTE2;
  if (coreType == "CUBE" || coreType == "AIC")
    return HIVMPipe::CubeMTE2;
  return HIVMPipe::VectorMTE2;
}

static HIVMPipe selectMTE2PipeForSpaces(llvm::StringRef srcSpace,
                                        llvm::StringRef dstSpace,
                                        llvm::StringRef coreType) {
  if (dstSpace == "ub" || srcSpace == "ub")
    return HIVMPipe::VectorMTE2;
  if (dstSpace == "l1" || dstSpace == "l0a" || dstSpace == "l0b" ||
      dstSpace == "l0c" || srcSpace == "l1" || srcSpace == "l0a" ||
      srcSpace == "l0b" || srcSpace == "l0c")
    return HIVMPipe::CubeMTE2;
  if (coreType == "CUBE" || coreType == "AIC")
    return HIVMPipe::CubeMTE2;
  return HIVMPipe::VectorMTE2;
}

static int64_t getElementByteWidth(llvm::StringRef typeToken) {
  llvm::StringRef t = trim(typeToken);
  if (t == "f16" || t == "bf16" || t == "i16")
    return 2;
  if (t == "f32" || t == "i32")
    return 4;
  if (t == "f64" || t == "i64")
    return 8;
  if (t == "i8" || t == "ui8" || t == "si8")
    return 1;
  return 0;
}

static int64_t parseMemRefElementCount(llvm::StringRef line) {
  size_t memrefPos = line.find("memref<");
  if (memrefPos == llvm::StringRef::npos)
    return 0;
  size_t addrPos = line.find(", #hivm.address_space<", memrefPos);
  if (addrPos == llvm::StringRef::npos)
    return 0;
  llvm::StringRef shapeAndType = line.slice(memrefPos + 7, addrPos);
  llvm::SmallVector<llvm::StringRef, 8> parts;
  shapeAndType.split(parts, 'x', -1, false);
  if (parts.empty())
    return 0;

  int64_t count = 1;
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    llvm::StringRef dim = trim(parts[i]);
    if (dim == "?" || dim.empty())
      return 0;
    int64_t value = 0;
    if (dim.getAsInteger(10, value))
      return 0;
    count *= value;
  }
  return count;
}

static int64_t parseMemRefBytes(llvm::StringRef line) {
  size_t memrefPos = line.find("memref<");
  if (memrefPos == llvm::StringRef::npos)
    return 0;
  size_t addrPos = line.find(", #hivm.address_space<", memrefPos);
  if (addrPos == llvm::StringRef::npos)
    return 0;
  llvm::StringRef shapeAndType = line.slice(memrefPos + 7, addrPos);
  llvm::SmallVector<llvm::StringRef, 8> parts;
  shapeAndType.split(parts, 'x', -1, false);
  if (parts.empty())
    return 0;

  llvm::StringRef elemType = trim(parts.back());
  int64_t elemBytes = getElementByteWidth(elemType);
  if (elemBytes <= 0)
    return 0;

  int64_t count = 1;
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    llvm::StringRef dim = trim(parts[i]);
    if (dim == "?" || dim.empty())
      return 0;
    int64_t value = 0;
    if (dim.getAsInteger(10, value))
      return 0;
    count *= value;
  }

  return count * elemBytes;
}

static int64_t estimateMmadL1MTE1Cycles(const ParsedOp &parsed,
                                        const HardwareConfig &config) {
  int64_t inputBytes = 0;
  for (const std::string &buffer : parsed.op.readBuffers)
    inputBytes += parseMemRefBytes(buffer);
  if (inputBytes <= 0)
    inputBytes = std::max<int64_t>(parsed.op.bytes, 1);

  double bandwidth = config.getMemoryBandwidthBytesPerCycle("mte1");
  if (bandwidth <= 0.0)
    bandwidth = std::max(1.0, config.getMemoryBandwidthBytesPerCycle("l1"));
  int64_t transferCycles =
      std::max<int64_t>(1, static_cast<int64_t>(std::ceil(inputBytes / bandwidth)));
  int64_t startupCycles = std::max<int64_t>(4, config.getMTE2StartupLatency() / 5);
  return startupCycles + transferCycles;
}

static int64_t estimateND2NZCycles(const ParsedOp &parsed,
                                   const HardwareConfig &config) {
  int64_t bytes = std::max<int64_t>(parsed.op.bytes, 1);
  // nd2nz is not a plain HBM->L1 DMA. The layout conversion runs on the
  // cube-side transfer path but sustains lower throughput than a normal
  // cube_mte2 transport, so keep it on a dedicated calibration path.
  double baseBandwidth = config.getMemoryBandwidthBytesPerCycle("cube_mte2");
  if (baseBandwidth <= 0.0)
    baseBandwidth = std::max(1.0, config.getMemoryBandwidthBytesPerCycle("hbm"));
  double effectiveBandwidth = std::max(1.0, baseBandwidth * 0.5);
  int64_t transferCycles = std::max<int64_t>(
      1, static_cast<int64_t>(std::ceil(bytes / effectiveBandwidth)));
  int64_t startupCycles = std::max<int64_t>(16, config.getMTE2StartupLatency() / 3);
  return startupCycles + transferCycles;
}

static std::string canonicalizeAddressSpace(llvm::StringRef space) {
  llvm::StringRef s = trim(space);
  if (s == "gm")
    return "gm";
  if (s == "ub")
    return "ub";
  if (s == "l1" || s == "cbuf")
    return "l1";
  if (s == "ca" || s == "l0a")
    return "l0a";
  if (s == "cb" || s == "l0b")
    return "l0b";
  if (s == "cc" || s == "l0c")
    return "l0c";
  return s.str();
}

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
static std::string getCanonicalTypeAddressSpace(mlir::Type type) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  if (!memref)
    return "";
  mlir::Attribute memSpace = memref.getMemorySpace();
  auto addrAttr = llvm::dyn_cast_or_null<mlir::hivm::AddressSpaceAttr>(memSpace);
  if (!addrAttr)
    return "";
  switch (addrAttr.getAddressSpace()) {
  case mlir::hivm::AddressSpace::GM:
    return "gm";
  case mlir::hivm::AddressSpace::L1:
    return "l1";
  case mlir::hivm::AddressSpace::L0A:
    return "l0a";
  case mlir::hivm::AddressSpace::L0B:
    return "l0b";
  case mlir::hivm::AddressSpace::L0C:
    return "l0c";
  case mlir::hivm::AddressSpace::UB:
    return "ub";
  default:
    return "";
  }
}
#endif

static int64_t getTypeByteWidth(mlir::Type type) {
  if (!type)
    return 0;
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
    return std::max<int64_t>(1, intType.getWidth() / 8);
  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(type))
    return std::max<int64_t>(1, floatType.getWidth() / 8);
  return 0;
}

/// Get the human-readable element type name from an MLIR type.
/// Returns "" if the type cannot be classified.
static std::string getElementTypeName(mlir::Type type) {
  if (!type)

    return "";
  // Check scalar types first — these use TypeID, never touch ShapedType
  if (auto floatTy = llvm::dyn_cast<mlir::FloatType>(type)) {
    if (floatTy.isBF16())
      return "bf16";
    switch (floatTy.getWidth()) {
    case 16: return "f16";
    case 32: return "f32";
    case 64: return "f64";
    default: return "";
    }
  }
  if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(type)) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "i" << intTy.getWidth();
    return s;
  }
  if (llvm::isa<mlir::IndexType>(type))
    return "index";
  // Unwrap shaped types via concrete MemRefType/TensorType (safe TypeID path)
  // Avoid dyn_cast<ShapedType> which triggers interface dispatch and may
  // crash on types from unregistered dialects (LLVM 19 bug).
  if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type))
    return getElementTypeName(memref.getElementType());
  if (auto tensor = llvm::dyn_cast<mlir::TensorType>(type))
    return getElementTypeName(tensor.getElementType());
  auto shaped = llvm::dyn_cast<mlir::ShapedType>(type);
  if (shaped)
    return getElementTypeName(shaped.getElementType());
  return "";
}

static int64_t getShapedTypeElementCount(mlir::Type type) {

  // Use concrete type casts (TypeID-based) before ShapedType interface
  if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type)) {
    if (!memref.hasStaticShape()) return 0;
    int64_t count = 1;
    for (int64_t dim : memref.getShape()) count *= dim;
    return count;
  }
  if (auto tensor = llvm::dyn_cast<mlir::TensorType>(type)) {
    if (!tensor.hasStaticShape()) return 0;
    int64_t count = 1;
    for (int64_t dim : tensor.getShape()) count *= dim;
    return count;
  }
  auto shaped = llvm::dyn_cast<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  int64_t count = 1;
  for (int64_t dim : shaped.getShape())
    count *= dim;
  return count;
}

static int64_t getShapedTypeBytes(mlir::Type type) {

  // Use concrete type casts (TypeID-based) before ShapedType interface
  if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type)) {
    int64_t count = getShapedTypeElementCount(type);
    if (count <= 0) return 0;
    return count * getTypeByteWidth(memref.getElementType());
  }
  if (auto tensor = llvm::dyn_cast<mlir::TensorType>(type)) {
    int64_t count = getShapedTypeElementCount(type);
    if (count <= 0) return 0;
    return count * getTypeByteWidth(tensor.getElementType());
  }
  auto shaped = llvm::dyn_cast<mlir::ShapedType>(type);
  if (!shaped)
    return 0;
  int64_t count = getShapedTypeElementCount(type);
  if (count <= 0)
    return 0;
  return count * getTypeByteWidth(shaped.getElementType());
}

static int64_t inferValueBytes(mlir::Value value) {
  if (!value)
    return 0;
  return getShapedTypeBytes(value.getType());
}

static int64_t inferValueElements(mlir::Value value) {
  if (!value)
    return 0;
  return getShapedTypeElementCount(value.getType());
}

static int getLineNumberFromLocation(mlir::Location loc) {
  return llvm::TypeSwitch<mlir::Location, int>(loc)
      .Case<mlir::FileLineColLoc>([](auto fileLoc) {
        return static_cast<int>(fileLoc.getLine());
      })
      .Case<mlir::NameLoc>([](auto nameLoc) {
        return getLineNumberFromLocation(nameLoc.getChildLoc());
      })
      .Case<mlir::FusedLoc>([](auto fusedLoc) {
        for (mlir::Location child : fusedLoc.getLocations()) {
          int line = getLineNumberFromLocation(child);
          if (line > 0)
            return line;
        }
        return 0;
      })
      .Default([](mlir::Location) { return 0; });
}

static std::string renderOperation(mlir::Operation *op) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs();
  op->print(os, flags);
  os.flush();
  return storage;
}

static void eraseAttributeAssignment(std::string &line, llvm::StringRef name) {
  std::string needle = name.str();
  for (;;) {
    size_t pos = line.find(needle);
    if (pos == std::string::npos)
      return;

    size_t cursor = pos + needle.size();
    while (cursor < line.size() && line[cursor] == ' ')
      ++cursor;
    if (cursor >= line.size() || line[cursor] != '=')
      return;
    ++cursor;
    while (cursor < line.size() && line[cursor] == ' ')
      ++cursor;
    if (cursor >= line.size())
      return;

    char opener = line[cursor];
    if (opener != '#' && opener != '<')
      return;
    size_t valueEnd = line.find('>', cursor);
    if (valueEnd == std::string::npos)
      return;
    ++valueEnd;

    size_t eraseStart = pos;
    while (eraseStart > 0 && line[eraseStart - 1] == ' ')
      --eraseStart;
    if (eraseStart > 0 && line[eraseStart - 1] == ',') {
      --eraseStart;
      while (eraseStart > 0 && line[eraseStart - 1] == ' ')
        --eraseStart;
    } else {
      while (valueEnd < line.size() && line[valueEnd] == ' ')
        ++valueEnd;
      if (valueEnd < line.size() && line[valueEnd] == ',')
        ++valueEnd;
      while (valueEnd < line.size() && line[valueEnd] == ' ')
        ++valueEnd;
    }

    line.erase(eraseStart, valueEnd - eraseStart);
  }
}

static std::string sanitizeMlirBuffer(llvm::StringRef buffer) {
  // Pre-process to remove custom dialect attributes/types that require
  // registered dialects.  When built without BiShengIR, the parser cannot
  // handle #hivm.address_space<...>, #hacc.arg_type<...>, etc.
  std::string preprocessed;
  {
    llvm::SmallVector<llvm::StringRef, 0> lines;
    buffer.split(lines, '\n');
    llvm::raw_string_ostream os(preprocessed);
    for (llvm::StringRef line : lines) {
      llvm::StringRef trimmed = line.trim();
      if (trimmed.starts_with("warning: ") || trimmed.ends_with("warning generated."))
        break;
      if (trimmed.starts_with("ld.lld:") || trimmed.starts_with("[ERROR]") ||
          trimmed.starts_with("[WARNING]") || trimmed.starts_with("[INFO]"))
        continue;
      std::string l = line.str();
#ifndef TRITONSIM_HAS_BISHENGIR_HIVM
      // Replace #hivm.address_space<xxx> with integer memory space when the
      // typed HIVM dialect is unavailable.
      while (auto pos = l.find("#hivm.address_space<")) {
        auto end = l.find('>', pos);
        if (end == std::string::npos) break;
        auto space = llvm::StringRef(l).slice(pos + 20, end);
        int num = 0;
        if (space == "gm") num = 0;
        else if (space == "ub") num = 1;
        else if (space == "l1") num = 2;
        else if (space == "l0a") num = 3;
        else if (space == "l0b") num = 4;
        else if (space == "l0c") num = 5;
        else if (space == "cbuf") num = 6;
        l.replace(pos, end - pos + 1, std::to_string(num));
      }
#endif
      // Strip whole custom attribute assignments only when the typed dialects
      // are unavailable.  When they are registered, keep hacc.arg_type so
      // argument binding remains consistent with the MLIR pass path.
#ifndef TRITONSIM_HAS_BISHENGIR_HIVM
      eraseAttributeAssignment(l, "hacc.arg_type");
      eraseAttributeAssignment(l, "hivm.func_core_type");
      eraseAttributeAssignment(l, "hacc.function_kind");
#endif
      os << l << "\n";
    }
    os.flush();
  }

  return preprocessed;
}

static std::string wrapBareMlirModule(llvm::StringRef buffer) {
  llvm::StringRef trimmed = buffer.trim();
  if (trimmed.empty() || trimmed.starts_with("module"))
    return buffer.str();

  std::string wrapped;
  llvm::raw_string_ostream os(wrapped);
  os << "module {\n";
  llvm::SmallVector<llvm::StringRef, 0> lines;
  buffer.split(lines, '\n');
  for (llvm::StringRef line : lines)
    os << "  " << line << "\n";
  os << "}\n";
  os.flush();
  return wrapped;
}

static bool startsWithHivmOp(mlir::Operation *op) {
  return op->getName().getStringRef().starts_with("hivm.hir.");
}

static llvm::StringRef getLeafOpName(mlir::Operation *op) {
  llvm::StringRef fullName = op->getName().getStringRef();
  size_t dot = fullName.rfind('.');
  if (dot == llvm::StringRef::npos)
    return fullName;
  return fullName.drop_front(dot + 1);
}

#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
static HIVMPipe convertTypedPipe(mlir::hivm::PIPE pipe) {
  switch (pipe) {
  case mlir::hivm::PIPE::PIPE_V:
    return HIVMPipe::Vector;
  case mlir::hivm::PIPE::PIPE_MTE2:
  case mlir::hivm::PIPE::VIRTUAL_PIPE_MTE2_L1A:
  case mlir::hivm::PIPE::VIRTUAL_PIPE_MTE2_L1B:
    return HIVMPipe::VectorMTE2;
  case mlir::hivm::PIPE::PIPE_MTE3:
    return HIVMPipe::MTE3;
  case mlir::hivm::PIPE::PIPE_S:
    return HIVMPipe::Scalar;
  case mlir::hivm::PIPE::PIPE_FIX:
    return HIVMPipe::FixPipe;
  case mlir::hivm::PIPE::PIPE_M:
    return HIVMPipe::Cube;
  case mlir::hivm::PIPE::PIPE_MTE1:
    return HIVMPipe::MTE1;
  case mlir::hivm::PIPE::PIPE_ALL:
    return HIVMPipe::All;
  default:
    return HIVMPipe::Unknown;
  }
}

static std::string stringifyTypedPipe(mlir::hivm::PIPE pipe) {
  return mlir::hivm::stringifyPIPE(pipe).str();
}

static std::string stringifyTypedCore(std::optional<mlir::hivm::TCoreType> core) {
  if (!core)
    return "";
  return mlir::hivm::stringifyTCoreType(*core).str();
}


static std::string stringifyTypedEvent(std::optional<mlir::hivm::EventAttr> staticEvent) {
  if (staticEvent)
    return ("event_" + mlir::hivm::stringifyEVENT(staticEvent->getEvent()).str());
  return "";
}

static std::string stringifyTypedFlag(std::optional<mlir::IntegerAttr> staticFlag) {
  if (staticFlag)
    return ("flag_" + std::to_string(staticFlag->getInt()));
  return "";
}

static std::string canonicalizeStaticEventToken(llvm::StringRef token) {
  if (token.consume_front("EVENT_ID"))
    return token.str();
  return token.str();
}

static bool populateTypedHivmOp(mlir::Operation *op, ParsedOp &parsed) {
  auto isTypedCubeOpName = [&](llvm::StringRef opName) {
    return opName == "matmul" || opName == "mix_matmul" ||
           opName == "mix_group_matmul" || opName == "mmadL1";
  };
  parsed.op.opName = getLeafOpName(op).str();

  if (auto pipeIface = llvm::dyn_cast<mlir::hivm::OpPipeInterface>(op)) {
    // VBrcOp::getPipe() internally calls getHIVMAddressSpace() which
    // dereferences a memory-space attribute that may be absent on
    // non-standard memref types in some MLIR variants.
    if (parsed.op.opName != "vbrc") {
      if (pipeIface.isSinglePipeOp()) {
        parsed.op.pipe = convertTypedPipe(pipeIface.getPipe());
      } else if (!isTypedCubeOpName(parsed.op.opName)) {
        parsed.op.pipe = convertTypedPipe(pipeIface.getOutPipe());
      }
    }
  }
  if (auto coreIface = llvm::dyn_cast<mlir::hivm::CoreTypeInterface>(op)) {
    auto ct = coreIface.getCoreType();
    if (ct)
      parsed.op.coreType = stringifyTypedCore(ct);
  }
  if (parsed.op.coreType.empty()) {
    if (auto inferIface = llvm::dyn_cast<mlir::hivm::InferCoreTypeInterface>(op)) {
      auto inferred = inferIface.inferCoreType();
      if (inferred)
        parsed.op.coreType = stringifyTypedCore(inferred);
    }
  }
  // Resolve "CUBE_OR_VECTOR" to a concrete core type using the enclosing
  // function name.  This is critical for disambiguating PIPE_MTE2 into
  // PIPE_MTE2_C vs PIPE_MTE2_V for pipe_barrier ops.
  parsed.op.coreType =
      resolveCoreTypeFromFunc(op, parsed.op.coreType).str();

  if (llvm::isa<mlir::hivm::LoadOp>(op) && op->getNumOperands() > 0) {
    parsed.op.pipe = selectMTE2PipeForSpaces(
        getCanonicalTypeAddressSpace(op->getOperand(0).getType()),
        op->getNumOperands() > 1
            ? getCanonicalTypeAddressSpace(op->getOperand(1).getType())
            : llvm::StringRef(),
        parsed.op.coreType);
  } else if (llvm::isa<mlir::hivm::StoreOp>(op))
    parsed.op.pipe = HIVMPipe::MTE3;
  else if (llvm::isa<mlir::hivm::FixpipeOp>(op))
    parsed.op.pipe = HIVMPipe::FixPipe;
  else if (parsed.op.opName == "nd2nz")
    parsed.op.pipe = HIVMPipe::CubeMTE2;
  else if (parsed.op.opName == "nz2nd")
    parsed.op.pipe = HIVMPipe::MTE3;
  else if (auto copyOp = llvm::dyn_cast<mlir::hivm::CopyOp>(op)) {
    std::string srcSpace = getCanonicalTypeAddressSpace(copyOp.getSrc().getType());
    std::string dstSpace = getCanonicalTypeAddressSpace(copyOp.getDst().getType());
    if (srcSpace == "ub" && dstSpace == "l1")
      parsed.op.pipe = HIVMPipe::MTE3;
    else if (srcSpace == "gm" && dstSpace == "l1")
      parsed.op.pipe = HIVMPipe::CubeMTE2;
    else if (srcSpace == "l0c" && dstSpace == "gm")
      parsed.op.pipe = HIVMPipe::FixPipe;
    else if (srcSpace == "ub" && dstSpace == "ub")
      parsed.op.pipe = HIVMPipe::Vector;
  }
  else if (parsed.op.opName == "convert_layout" ||
           parsed.op.opName == "pointer_cast")
    parsed.op.pipe = HIVMPipe::Unknown;
  else if (parsed.op.opName == "vbrc")
    parsed.op.pipe = HIVMPipe::Vector;
  else if (llvm::isa<mlir::hivm::MmadL1Op, mlir::hivm::MatmulOp,
                     mlir::hivm::MixMatmulOp, mlir::hivm::MixGroupMatmulOp>(op)) {
    parsed.op.pipe = HIVMPipe::Cube;
    for (mlir::Value operand : op->getOperands()) {
      llvm::StringRef elemTy = getElementTypeName(operand.getType());
      if (!elemTy.empty())
        parsed.op.elemType = elemTy.str();
      parsed.op.bytes = std::max(parsed.op.bytes, inferValueBytes(operand));
      parsed.op.elements = std::max(parsed.op.elements, inferValueElements(operand));
    }
  }

  if (auto barrier = llvm::dyn_cast<mlir::hivm::PipeBarrierOp>(op)) {
    parsed.op.opName = "pipe_barrier";
    parsed.op.isSyncOp = true;
    parsed.op.isBarrier = true;
    if (!barrier.getPipe()) return true;
    HIVMPipe rawPipe = convertTypedPipe(barrier.getPipe().getPipe());
    parsed.op.pipe = disambiguateMTE2Pipe(rawPipe, HIVMPipe::Unknown,
                                          parsed.op.coreType);
    parsed.barrierPipes.push_back(parsed.op.pipe);
    return true;
  }
  if (auto setFlag = llvm::dyn_cast<mlir::hivm::SetFlagOp>(op)) {
    parsed.op.opName = "set_flag";
    parsed.op.isSyncOp = true;
    if (!setFlag.getSetPipe() || !setFlag.getWaitPipe()) return true;
    parsed.senderEvent = stringifyTypedPipe(setFlag.getSetPipe().getPipe());
    parsed.receiverEvent = stringifyTypedPipe(setFlag.getWaitPipe().getPipe());
    parsed.eventId =
        stringifyTypedEvent(setFlag.getStaticEventId());
    parsed.syncIdValue = setFlag.getDynamicEventId();
    parsed.senderPipe = disambiguateMTE2Pipe(
        convertTypedPipe(setFlag.getSetPipe().getPipe()),
        convertTypedPipe(setFlag.getWaitPipe().getPipe()), parsed.op.coreType);
    parsed.receiverPipe = disambiguateMTE2Pipe(
        convertTypedPipe(setFlag.getWaitPipe().getPipe()), parsed.senderPipe,
        parsed.op.coreType);
    parsed.op.pipe = parsed.senderPipe;
    return true;
  }
  if (auto waitFlag = llvm::dyn_cast<mlir::hivm::WaitFlagOp>(op)) {
    parsed.op.opName = "wait_flag";
    parsed.op.isSyncOp = true;
    if (!waitFlag.getSetPipe() || !waitFlag.getWaitPipe()) return true;
    parsed.senderEvent = stringifyTypedPipe(waitFlag.getSetPipe().getPipe());
    parsed.receiverEvent = stringifyTypedPipe(waitFlag.getWaitPipe().getPipe());
    parsed.eventId = stringifyTypedEvent(waitFlag.getStaticEventId());
    parsed.syncIdValue = waitFlag.getDynamicEventId();
    parsed.senderPipe = disambiguateMTE2Pipe(
        convertTypedPipe(waitFlag.getSetPipe().getPipe()),
        convertTypedPipe(waitFlag.getWaitPipe().getPipe()), parsed.op.coreType);
    parsed.receiverPipe = disambiguateMTE2Pipe(
        convertTypedPipe(waitFlag.getWaitPipe().getPipe()), parsed.senderPipe,
        parsed.op.coreType);
    parsed.op.pipe = parsed.receiverPipe;
    return true;
  }
  if (auto syncSet = llvm::dyn_cast<mlir::hivm::SyncBlockSetOp>(op)) {
    parsed.op.opName = "sync_block_set";
    parsed.op.isSyncOp = true;
    if (!syncSet.getTcoreType() || !syncSet.getTpipe() || !syncSet.getPipe())
      return true;
    parsed.syncCoreType =
        mlir::hivm::stringifyTCoreType(syncSet.getTcoreType().getTcoretype()).str();
    parsed.op.coreType = parsed.syncCoreType;
    parsed.senderEvent = stringifyTypedPipe(syncSet.getTpipe().getPipe());
    parsed.receiverEvent = stringifyTypedPipe(syncSet.getPipe().getPipe());
    parsed.senderPipe = disambiguateMTE2Pipe(
        convertTypedPipe(syncSet.getTpipe().getPipe()),
        convertTypedPipe(syncSet.getPipe().getPipe()), parsed.op.coreType);
    parsed.receiverPipe = disambiguateMTE2Pipe(
        convertTypedPipe(syncSet.getPipe().getPipe()), parsed.senderPipe,
        parsed.op.coreType);
    parsed.eventId =
        stringifyTypedFlag(syncSet.getStaticFlagId());
    parsed.syncIdValue = syncSet.getDynamicFlagId();
    parsed.op.pipe = parsed.senderPipe;
    return true;
  }
  if (auto syncWait = llvm::dyn_cast<mlir::hivm::SyncBlockWaitOp>(op)) {
    parsed.op.opName = "sync_block_wait";
    parsed.op.isSyncOp = true;
    parsed.op.isBarrier = true;
    if (!syncWait.getTcoreType() || !syncWait.getTpipe() || !syncWait.getPipe())
      return true;
    parsed.syncCoreType =
        mlir::hivm::stringifyTCoreType(syncWait.getTcoreType().getTcoretype()).str();
    parsed.op.coreType = parsed.syncCoreType;
    parsed.senderEvent = stringifyTypedPipe(syncWait.getTpipe().getPipe());
    parsed.receiverEvent = stringifyTypedPipe(syncWait.getPipe().getPipe());
    // The sender pipe lives on the *opposite* core in cross-core sync.
    llvm::StringRef senderCoreType =
        (parsed.op.coreType == "CUBE" || parsed.op.coreType == "AIC")
            ? "VECTOR"
            : "CUBE";
    parsed.senderPipe = disambiguateMTE2Pipe(
        convertTypedPipe(syncWait.getTpipe().getPipe()),
        convertTypedPipe(syncWait.getPipe().getPipe()), senderCoreType);
    parsed.receiverPipe = disambiguateMTE2Pipe(
        convertTypedPipe(syncWait.getPipe().getPipe()), parsed.senderPipe,
        parsed.op.coreType);
    parsed.eventId =
        stringifyTypedFlag(syncWait.getStaticFlagId());
    parsed.syncIdValue = syncWait.getDynamicFlagId();
    parsed.op.pipe = HIVMPipe::All;
    parsed.barrierPipes.push_back(HIVMPipe::All);
    return true;
  }
  if (auto syncBlock = llvm::dyn_cast<mlir::hivm::SyncBlockOp>(op)) {
    parsed.op.opName = "sync_block";
    parsed.op.isSyncOp = true;
    parsed.op.isBarrier = true;
    parsed.op.pipe = HIVMPipe::All;
    if (auto inferredCore = syncBlock.inferCoreType())
      parsed.op.coreType = mlir::hivm::stringifyTCoreType(*inferredCore).str();
    return true;
  }

  return startsWithHivmOp(op);
}
#endif

static std::string stringifyAttribute(mlir::Attribute attr) {
  if (!attr)
    return "";
  std::string storage;
  llvm::raw_string_ostream os(storage);
  attr.print(os);
  os.flush();
  return storage;
}

static std::pair<std::string, std::string> parseLoadStoreSpaces(llvm::StringRef line);

static HIVMPipe parsePipeToken(llvm::StringRef text) {
  if (text.contains("PIPE_ALL"))
    return HIVMPipe::All;
  if (text.contains("PIPE_MTE3"))
    return HIVMPipe::MTE3;
  if (text.contains("PIPE_MTE2"))
    return HIVMPipe::VectorMTE2;
  if (text.contains("PIPE_MTE1"))
    return HIVMPipe::MTE1;
  if (text.contains("PIPE_FIX"))
    return HIVMPipe::FixPipe;
  if (text.contains("PIPE_M"))
    return HIVMPipe::Cube;
  if (text.contains("PIPE_V"))
    return HIVMPipe::Vector;
  if (text.contains("PIPE_S"))
    return HIVMPipe::Scalar;
  return HIVMPipe::Unknown;
}

static std::string parseEventToken(llvm::StringRef text) {
  size_t pos = text.find("EVENT_ID");
  if (pos == llvm::StringRef::npos)
    return "";
  size_t end = pos;
  while (end < text.size() &&
         (std::isalnum(static_cast<unsigned char>(text[end])) ||
          text[end] == '_'))
    ++end;
  return text.slice(pos, end).str();
}

#ifndef TRITONSIM_HAS_BISHENGIR_HIVM
static std::string inferGenericCoreType(mlir::Operation *op) {
  if (auto parentFunc = op->getParentOfType<mlir::func::FuncOp>()) {
    std::string attr = stringifyAttribute(parentFunc->getAttr("hivm.func_core_type"));
    if (llvm::StringRef(attr).contains("AIC") ||
        llvm::StringRef(attr).contains("CUBE"))
      return "CUBE";
    if (llvm::StringRef(attr).contains("AIV") ||
        llvm::StringRef(attr).contains("VECTOR"))
      return "VECTOR";
  }
  return resolveCoreTypeFromFunc(op, "").str();
}

static bool populateGenericHivmOp(mlir::Operation *op, ParsedOp &parsed) {
  parsed.op.opName = getLeafOpName(op).str();
  parsed.op.coreType = inferGenericCoreType(op);
  std::string opText = renderOperation(op);
  auto spaces = parseLoadStoreSpaces(opText);

  if (parsed.op.opName == "load") {
    parsed.op.pipe =
        selectMTE2PipeForSpaces(spaces.first, spaces.second, parsed.op.coreType);
  } else if (parsed.op.opName == "store") {
    parsed.op.pipe = HIVMPipe::MTE3;
  } else if (parsed.op.opName == "fixpipe") {
    parsed.op.pipe = HIVMPipe::FixPipe;
  } else if (parsed.op.opName == "nd2nz") {
    parsed.op.pipe = HIVMPipe::CubeMTE2;
  } else if (parsed.op.opName == "nz2nd") {
    parsed.op.pipe = HIVMPipe::MTE3;
  } else if (parsed.op.opName == "copy") {
    if (spaces.second == "gm")
      parsed.op.pipe = HIVMPipe::MTE3;
    else if (spaces.second == "l1")
      parsed.op.pipe = HIVMPipe::CubeMTE2;
    else
      parsed.op.pipe = HIVMPipe::Vector;
  } else if (parsed.op.opName == "pointer_cast" ||
             parsed.op.opName == "convert_layout") {
    parsed.op.pipe = HIVMPipe::Unknown;
  } else if (parsed.op.opName == "matmul" ||
             parsed.op.opName == "mix_matmul" ||
             parsed.op.opName == "mix_group_matmul" ||
             parsed.op.opName == "mmadL1") {
    parsed.op.pipe = HIVMPipe::Cube;
  } else {
    parsed.op.pipe = HIVMPipe::Vector;
  }

  if (parsed.op.opName == "pipe_barrier") {
    parsed.op.isSyncOp = true;
    parsed.op.isBarrier = true;
    parsed.op.pipe =
        disambiguateMTE2Pipe(parsePipeToken(stringifyAttribute(op->getAttr("pipe"))),
                             HIVMPipe::Unknown, parsed.op.coreType);
    parsed.barrierPipes.push_back(parsed.op.pipe);
    return true;
  }
  if (parsed.op.opName == "set_flag" || parsed.op.opName == "wait_flag") {
    parsed.op.isSyncOp = true;
    HIVMPipe setPipe = parsePipeToken(stringifyAttribute(op->getAttr("set_pipe")));
    HIVMPipe waitPipe = parsePipeToken(stringifyAttribute(op->getAttr("wait_pipe")));
    parsed.senderPipe =
        disambiguateMTE2Pipe(setPipe, waitPipe, parsed.op.coreType);
    parsed.receiverPipe =
        disambiguateMTE2Pipe(waitPipe, parsed.senderPipe, parsed.op.coreType);
    parsed.eventId = parseEventToken(
        stringifyAttribute(op->getAttr("static_event_id")));
    parsed.op.pipe = parsed.op.opName == "set_flag" ? parsed.senderPipe
                                                     : parsed.receiverPipe;
    attachSyncMetadata(parsed);
    return true;
  }

  return true;
}
#endif

/// Extract CCE repeat and mask state from an MLIR operation.
///
/// Repeat is sourced from:
///   1. An explicit ``repeat`` integer attribute on the op (if present).
///   2. A ``repeat = N`` pattern in the rendered op text (fallback).
///
/// Mask is sourced from:
///   1. A ``mask_count`` integer attribute on the op (if present).
///
/// Defaults (repeat=1, mask=0) are left intact when no data is found.
static void extractRepeatMask(mlir::Operation *op, ParsedOp &parsed) {
  // 1. Explicit integer attribute "repeat"
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("repeat"))
    parsed.op.repeat = std::max<int64_t>(1, attr.getInt());

  // 2. Fallback: parse "repeat = N" from rendered op text.
  if (parsed.op.repeat == 1 && !parsed.op.text.empty()) {
    llvm::StringRef txt = parsed.op.text;
    auto pos = txt.find("repeat = ");
    if (pos != llvm::StringRef::npos) {
      // Anchor: preceding char must NOT be an identifier char — prevents
      // matching "loop_repeat = 4" at the wrong offset.
      bool anchored = (pos == 0 ||
                       (!std::isalnum(static_cast<unsigned char>(txt[pos - 1])) &&
                        txt[pos - 1] != '_'));
      if (anchored) {
        int64_t val = 0;
        llvm::StringRef tail = txt.drop_front(pos + 9); // len("repeat = ")
        // consumeInteger parses a leading integer and tolerates trailing text
        // (e.g. "8 : i64}" from the rendered "repeat = 8 : i64"), whereas
        // getAsInteger fails on any trailing chars.
        if (!tail.consumeInteger(10, val) && val >= 1)
          parsed.op.repeat = val;
      }
    }
  }

  // 3. Explicit integer attribute "mask_count"
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("mask_count"))
    parsed.op.mask = std::max<int64_t>(0, attr.getInt());
}

static std::string renderValueToken(mlir::Value value) {
  if (!value)
    return "";
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << value;
  os.flush();
  return storage;
}

static std::string renderOpaqueValueToken(mlir::Value value) {
  if (!value)
    return "";
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "ssa@" << value.getAsOpaquePointer();
  os.flush();
  return storage;
}

static std::optional<int64_t> evaluateAffineExpr(mlir::AffineExpr expr,
                                                 mlir::AffineMap map,
                                                 llvm::ArrayRef<int64_t> inputs);

static bool resolveMLIRValueImpl(mlir::Value value, const AnalysisState &state,
                                 int64_t &resolved,
                                 llvm::SmallDenseSet<mlir::Value, 8> &visited);

static bool resolveAffineApply(mlir::affine::AffineApplyOp affineApply,
                               const AnalysisState &state, int64_t &resolved,
                               llvm::SmallDenseSet<mlir::Value, 8> &visited) {
  llvm::SmallVector<int64_t, 8> inputs;
  inputs.reserve(affineApply.getOperands().size());
  for (mlir::Value operand : affineApply.getOperands()) {
    int64_t operandValue = 0;
    if (!resolveMLIRValueImpl(operand, state, operandValue, visited))
      return false;
    inputs.push_back(operandValue);
  }
  auto result =
      evaluateAffineExpr(affineApply.getAffineMap().getResult(0),
                         affineApply.getAffineMap(), inputs);
  if (!result)
    return false;
  resolved = *result;
  return true;
}

// Resolve an affine.min / affine.max: evaluate every result expr of the map
// over the resolved operands and fold by min/max.  Needed because masked tile
// sizes lower to affine.min(BLOCK, remaining) and offsets to affine.max(0, …),
// which gate the dynamic transfer extents.
static bool resolveAffineMinMax(mlir::AffineMap map, mlir::ValueRange operands,
                                bool isMin, const AnalysisState &state,
                                int64_t &resolved,
                                llvm::SmallDenseSet<mlir::Value, 8> &visited) {
  llvm::SmallVector<int64_t, 8> inputs;
  inputs.reserve(operands.size());
  for (mlir::Value operand : operands) {
    int64_t v = 0;
    if (!resolveMLIRValueImpl(operand, state, v, visited))
      return false;
    inputs.push_back(v);
  }
  std::optional<int64_t> acc;
  for (mlir::AffineExpr expr : map.getResults()) {
    auto r = evaluateAffineExpr(expr, map, inputs);
    if (!r)
      return false;
    acc = acc ? (isMin ? std::min(*acc, *r) : std::max(*acc, *r)) : *r;
  }
  if (!acc)
    return false;
  resolved = *acc;
  return true;
}

static bool resolveMLIRValueImpl(mlir::Value value, const AnalysisState &state,
                                 int64_t &resolved,
                                 llvm::SmallDenseSet<mlir::Value, 8> &visited) {
  if (!visited.insert(value).second)
    return false;
  auto finish = [&](bool ok) {
    visited.erase(value);
    return ok;
  };

  auto cstIt = state.constants.find(value);
  if (cstIt != state.constants.end()) {
    resolved = cstIt->second;
    return finish(true);
  }
  auto boundIt = state.boundValues.find(value);
  if (boundIt != state.boundValues.end()) {
    resolved = boundIt->second;
    return finish(true);
  }
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp || defOp->getNumResults() != 1)
    return finish(false);

  if (auto constantOp = llvm::dyn_cast<mlir::arith::ConstantOp>(defOp)) {
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constantOp.getValue())) {
      resolved = intAttr.getInt();
      return finish(true);
    }
  }
  if (auto castOp = llvm::dyn_cast<mlir::arith::IndexCastOp>(defOp))
    return finish(resolveMLIRValueImpl(castOp.getIn(), state, resolved, visited));
  if (auto castOp = llvm::dyn_cast<mlir::arith::IndexCastUIOp>(defOp))
    return finish(resolveMLIRValueImpl(castOp.getIn(), state, resolved, visited));
  if (auto truncOp = llvm::dyn_cast<mlir::arith::TruncIOp>(defOp))
    return finish(resolveMLIRValueImpl(truncOp.getIn(), state, resolved, visited));
  if (auto extOp = llvm::dyn_cast<mlir::arith::ExtSIOp>(defOp))
    return finish(resolveMLIRValueImpl(extOp.getIn(), state, resolved, visited));
  if (auto addOp = llvm::dyn_cast<mlir::arith::AddIOp>(defOp)) {
    int64_t lhs = 0, rhs = 0;
    if (!resolveMLIRValueImpl(addOp.getLhs(), state, lhs, visited) ||
        !resolveMLIRValueImpl(addOp.getRhs(), state, rhs, visited))
      return finish(false);
    resolved = lhs + rhs;
    return finish(true);
  }
  if (auto subOp = llvm::dyn_cast<mlir::arith::SubIOp>(defOp)) {
    int64_t lhs = 0, rhs = 0;
    if (!resolveMLIRValueImpl(subOp.getLhs(), state, lhs, visited) ||
        !resolveMLIRValueImpl(subOp.getRhs(), state, rhs, visited))
      return finish(false);
    resolved = lhs - rhs;
    return finish(true);
  }
  if (auto mulOp = llvm::dyn_cast<mlir::arith::MulIOp>(defOp)) {
    int64_t lhs = 0, rhs = 0;
    bool lhsResolved = resolveMLIRValueImpl(mulOp.getLhs(), state, lhs, visited);
    if (lhsResolved && lhs == 0) {
      resolved = 0;
      return finish(true);
    }
    bool rhsResolved = resolveMLIRValueImpl(mulOp.getRhs(), state, rhs, visited);
    if (rhsResolved && rhs == 0) {
      resolved = 0;
      return finish(true);
    }
    if (!lhsResolved || !rhsResolved)
      return finish(false);
    resolved = lhs * rhs;
    return finish(true);
  }
  if (auto divOp = llvm::dyn_cast<mlir::arith::DivSIOp>(defOp)) {
    int64_t lhs = 0;
    if (!resolveMLIRValueImpl(divOp.getLhs(), state, lhs, visited))
      return finish(false);
    if (lhs == 0) {
      resolved = 0;
      return finish(true);
    }
    int64_t rhs = 0;
    if (!resolveMLIRValueImpl(divOp.getRhs(), state, rhs, visited) || rhs == 0)
      return finish(false);
    resolved = lhs / rhs;
    return finish(true);
  }
  if (auto remOp = llvm::dyn_cast<mlir::arith::RemSIOp>(defOp)) {
    int64_t lhs = 0;
    if (!resolveMLIRValueImpl(remOp.getLhs(), state, lhs, visited))
      return finish(false);
    if (lhs == 0) {
      resolved = 0;
      return finish(true);
    }
    int64_t rhs = 0;
    if (!resolveMLIRValueImpl(remOp.getRhs(), state, rhs, visited) || rhs == 0)
      return finish(false);
    resolved = lhs % rhs;
    return finish(true);
  }
  if (auto minOp = llvm::dyn_cast<mlir::arith::MinSIOp>(defOp)) {
    int64_t lhs = 0, rhs = 0;
    if (!resolveMLIRValueImpl(minOp.getLhs(), state, lhs, visited) ||
        !resolveMLIRValueImpl(minOp.getRhs(), state, rhs, visited))
      return finish(false);
    resolved = std::min(lhs, rhs);
    return finish(true);
  }
  if (auto minOp = llvm::dyn_cast<mlir::arith::MinUIOp>(defOp)) {
    int64_t lhs = 0, rhs = 0;
    if (!resolveMLIRValueImpl(minOp.getLhs(), state, lhs, visited) ||
        !resolveMLIRValueImpl(minOp.getRhs(), state, rhs, visited))
      return finish(false);
    resolved = static_cast<int64_t>(
        std::min(static_cast<uint64_t>(lhs), static_cast<uint64_t>(rhs)));
    return finish(true);
  }
  if (auto cmpOp = llvm::dyn_cast<mlir::arith::CmpIOp>(defOp)) {
    int64_t lhs = 0, rhs = 0;
    if (!resolveMLIRValueImpl(cmpOp.getLhs(), state, lhs, visited) ||
        !resolveMLIRValueImpl(cmpOp.getRhs(), state, rhs, visited))
      return finish(false);
    switch (cmpOp.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
      resolved = lhs == rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::ne:
      resolved = lhs != rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::slt:
      resolved = lhs < rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::sle:
      resolved = lhs <= rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::sgt:
      resolved = lhs > rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::sge:
      resolved = lhs >= rhs;
      return finish(true);
    case mlir::arith::CmpIPredicate::ult:
      resolved = static_cast<uint64_t>(lhs) < static_cast<uint64_t>(rhs);
      return finish(true);
    case mlir::arith::CmpIPredicate::ule:
      resolved = static_cast<uint64_t>(lhs) <= static_cast<uint64_t>(rhs);
      return finish(true);
    case mlir::arith::CmpIPredicate::ugt:
      resolved = static_cast<uint64_t>(lhs) > static_cast<uint64_t>(rhs);
      return finish(true);
    case mlir::arith::CmpIPredicate::uge:
      resolved = static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs);
      return finish(true);
    }
  }
  if (auto selectOp = llvm::dyn_cast<mlir::arith::SelectOp>(defOp)) {
    int64_t cond = 0, trueValue = 0, falseValue = 0;
    if (!resolveMLIRValueImpl(selectOp.getCondition(), state, cond, visited) ||
        !resolveMLIRValueImpl(selectOp.getTrueValue(), state, trueValue,
                              visited) ||
        !resolveMLIRValueImpl(selectOp.getFalseValue(), state, falseValue,
                              visited))
      return finish(false);
    resolved = cond != 0 ? trueValue : falseValue;
    return finish(true);
  }
  if (auto affineApply = llvm::dyn_cast<mlir::affine::AffineApplyOp>(defOp))
    return finish(resolveAffineApply(affineApply, state, resolved, visited));
  if (auto affineMin = llvm::dyn_cast<mlir::affine::AffineMinOp>(defOp))
    return finish(resolveAffineMinMax(affineMin.getMap(),
                                      affineMin.getOperands(), /*isMin=*/true,
                                      state, resolved, visited));
  if (auto affineMax = llvm::dyn_cast<mlir::affine::AffineMaxOp>(defOp))
    return finish(resolveAffineMinMax(affineMax.getMap(),
                                      affineMax.getOperands(), /*isMin=*/false,
                                      state, resolved, visited));

  return finish(false);
}

static bool resolveMLIRValue(mlir::Value value, const AnalysisState &state,
                             int64_t &resolved) {
  llvm::SmallDenseSet<mlir::Value, 8> visited;
  return resolveMLIRValueImpl(value, state, resolved, visited);
}

// Resolve an OpFoldResult (static attribute or dynamic SSA value) to a concrete
// integer, using arg bindings for the dynamic case.
static bool resolveOpFoldResult(mlir::OpFoldResult ofr,
                                const AnalysisState &state, int64_t &out) {
  if (auto val = llvm::dyn_cast_if_present<mlir::Value>(ofr))
    return resolveMLIRValue(val, state, out);
  if (auto attr =
          llvm::dyn_cast_if_present<mlir::IntegerAttr>(
              llvm::dyn_cast_if_present<mlir::Attribute>(ofr))) {
    out = attr.getInt();
    return true;
  }
  return false;
}

// Product of a static-dims array (with kDynamic entries) where each kDynamic is
// filled, in order, by the next dynamic operand resolved through arg bindings.
// Uses raw ODS accessors only (getStaticSizes/getSizes), deliberately avoiding
// the OffsetSizeAndStrideOpInterface mixed-helpers — those pull MemRef's tiling
// interface impl (→ linalg::makeTiledShapes) into the link.
static int64_t resolveDimsProduct(llvm::ArrayRef<int64_t> staticDims,
                                  mlir::OperandRange dynamicDims,
                                  const AnalysisState &state) {
  int64_t count = 1;
  unsigned dynIdx = 0;
  for (int64_t d : staticDims) {
    int64_t v = d;
    if (d == mlir::ShapedType::kDynamic) {
      if (dynIdx >= dynamicDims.size() ||
          !resolveMLIRValue(dynamicDims[dynIdx++], state, v))
        return 0;
    }
    if (v <= 0)
      return 0;
    count *= v;
  }
  return count;
}

// Element count of a (possibly dynamic-shaped) value: resolves dynamic
// subview / reinterpret_cast sizes from arg bindings.  Falls back to the
// static element count when no view op produces the value.
static int64_t inferValueElementsWithBindings(mlir::Value value,
                                              const AnalysisState &state) {
  if (!value)
    return 0;
  int64_t staticCount = inferValueElements(value);
  if (staticCount > 0)
    return staticCount;

  mlir::Operation *defOp = value.getDefiningOp();
  if (auto sv = llvm::dyn_cast_or_null<mlir::memref::SubViewOp>(defOp))
    return resolveDimsProduct(sv.getStaticSizes(), sv.getSizes(), state);
  if (auto rc = llvm::dyn_cast_or_null<mlir::memref::ReinterpretCastOp>(defOp))
    return resolveDimsProduct(rc.getStaticSizes(), rc.getSizes(), state);
  return 0;
}

static int64_t inferValueBytesWithBindings(mlir::Value value,
                                           const AnalysisState &state) {
  int64_t staticBytes = inferValueBytes(value);
  if (staticBytes > 0)
    return staticBytes;
  auto shaped = llvm::dyn_cast<mlir::ShapedType>(value.getType());
  if (!shaped)
    return 0;
  int64_t width = getTypeByteWidth(shaped.getElementType());
  if (width <= 0)
    return 0;
  int64_t count = inferValueElementsWithBindings(value, state);
  return count > 0 ? count * width : 0;
}

// Resolve the innermost ABSOLUTE stride (in elements) of a memref value.
// 1 => fully contiguous innermost dimension.  Resolves dynamic strides from
// the producing reinterpret_cast / subview using arg bindings.
static bool resolveInnermostStride(mlir::Value value,
                                   const AnalysisState &state, int64_t &stride) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(value.getType());
  if (!memref)
    return false;
  // Read the innermost stride straight from the strided layout attribute
  // (pure MLIRIR — avoids pulling the Linalg tiling interface).  An identity /
  // absent layout is contiguous (stride 1).
  auto strided =
      llvm::dyn_cast_or_null<mlir::StridedLayoutAttr>(memref.getLayout());
  if (!strided) {
    stride = 1;  // identity layout → contiguous
    return true;
  }
  llvm::ArrayRef<int64_t> strides = strided.getStrides();
  if (strides.empty())
    return false;
  int64_t inner = strides.back();
  if (inner != mlir::ShapedType::kDynamic) {
    stride = inner;
    return true;
  }
  // Dynamic innermost stride: resolve from the producing view op using raw
  // accessors (getStaticStrides/getStrides) — see resolveDimsProduct on why the
  // mixed-helpers are avoided.  The innermost dim is the last; if its static
  // stride is dynamic it is filled by the last dynamic stride operand.
  auto resolveLast = [&](llvm::ArrayRef<int64_t> staticArr,
                         mlir::OperandRange dynVals, int64_t &out) -> bool {
    if (staticArr.empty())
      return false;
    if (staticArr.back() != mlir::ShapedType::kDynamic) {
      out = staticArr.back();
      return true;
    }
    return !dynVals.empty() && resolveMLIRValue(dynVals.back(), state, out);
  };
  mlir::Operation *defOp = value.getDefiningOp();
  if (auto rc =
          llvm::dyn_cast_or_null<mlir::memref::ReinterpretCastOp>(defOp))
    return resolveLast(rc.getStaticStrides(), rc.getStrides(), stride);
  if (auto sv = llvm::dyn_cast_or_null<mlir::memref::SubViewOp>(defOp)) {
    int64_t srcStride = 1, svStride = 1;
    if (resolveLast(sv.getStaticStrides(), sv.getStrides(), svStride) &&
        resolveInnermostStride(sv.getSource(), state, srcStride)) {
      stride = srcStride * svStride;
      return true;
    }
  }
  return false;
}

// Contiguous packet size (bytes) the hardware moves in one shot for an MTE
// transfer of `value`.  Contiguous innermost dim => the whole run; strided =>
// a single element.  Returns `totalBytes` when contiguity can't be determined
// (conservative: assume coalesced, no Gap-2 penalty invented).
static int64_t inferTransferPacketBytes(mlir::Value value,
                                        const AnalysisState &state,
                                        int64_t totalBytes) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(value.getType());
  if (!memref)
    return totalBytes;
  int64_t width = getTypeByteWidth(memref.getElementType());
  if (width <= 0)
    return totalBytes;
  int64_t innermost = 1;
  if (!resolveInnermostStride(value, state, innermost))
    return totalBytes;  // unknown stride → assume contiguous
  if (innermost == 1)
    return totalBytes;  // contiguous run
  return width;         // strided/gather → one element per packet
}

static std::string canonicalizeSyncId(mlir::Value value,
                                      const AnalysisState &state) {
  if (!value)
    return "";

  int64_t resolved = 0;
  if (resolveMLIRValue(value, state, resolved))
    return std::to_string(resolved);

  auto producerIt = state.valueProducers.find(value);
  if (producerIt != state.valueProducers.end())
    return ("ssa_producer_" + std::to_string(producerIt->second));

  return renderOpaqueValueToken(value);
}

static bool parseForTripCount(mlir::scf::ForOp forOp, const AnalysisState &state,
                              int64_t &tripCount) {
  int64_t lb = 0;
  int64_t ub = 0;
  int64_t step = 0;
  if (!resolveMLIRValue(forOp.getLowerBound(), state, lb) ||
      !resolveMLIRValue(forOp.getUpperBound(), state, ub) ||
      !resolveMLIRValue(forOp.getStep(), state, step) || step <= 0 || ub < lb) {
    tripCount = 1;
    return false;
  }
  tripCount = std::max<int64_t>(1, ceilDiv(ub - lb, step));
  return true;
}

// Derive a sound STRUCTURAL upper bound on an scf.for's trip count even when
// the exact trip count is unresolvable (e.g. a program-id-dependent bound).
// Recognizes the common "clamped by a constant" shape: an upper-bound
// expression built from arith.minsi(...) with at least one branch of the
// form `arith.addi(lowerBound, CONST)` (in either operand order), where
// `lowerBound` is the SAME SSA value as the loop's own lower bound and
// CONST resolves to a compile-time constant. That branch alone proves
// `trip_count <= CONST`, regardless of whether the other minsi operand (or
// the lower bound's own numeric value) ever resolves — unlike
// resolveMLIRValueImpl's arith.MinSIOp handling, which requires BOTH
// operands to fully resolve to reason about anything.
//
// This is NOT a sound lower bound (a real program can have a smaller trip
// count than this) and must never feed the primary op.loopMultiplier used
// for T_bound — it is diagnostic-only, consumed to compute a clearly-labeled
// companion worst-case estimate.
static bool estimateForTripCountUpperBoundImpl(
    mlir::Value upperValue, mlir::Value lowerBoundValue,
    const AnalysisState &state, int64_t &bound,
    llvm::SmallDenseSet<mlir::Value, 8> &visited) {
  if (!visited.insert(upperValue).second)
    return false;
  auto finish = [&](bool ok) {
    visited.erase(upperValue);
    return ok;
  };

  mlir::Operation *defOp = upperValue.getDefiningOp();
  if (!defOp || defOp->getNumResults() != 1)
    return finish(false);

  if (auto minOp = llvm::dyn_cast<mlir::arith::MinSIOp>(defOp)) {
    int64_t lhsBound = 0, rhsBound = 0;
    bool lhsOk = estimateForTripCountUpperBoundImpl(
        minOp.getLhs(), lowerBoundValue, state, lhsBound, visited);
    bool rhsOk = estimateForTripCountUpperBoundImpl(
        minOp.getRhs(), lowerBoundValue, state, rhsBound, visited);
    if (!lhsOk && !rhsOk)
      return finish(false);
    bound = (lhsOk && rhsOk) ? std::min(lhsBound, rhsBound)
                             : (lhsOk ? lhsBound : rhsBound);
    return finish(true);
  }

  if (auto addOp = llvm::dyn_cast<mlir::arith::AddIOp>(defOp)) {
    mlir::Value constOperand;
    if (addOp.getLhs() == lowerBoundValue)
      constOperand = addOp.getRhs();
    else if (addOp.getRhs() == lowerBoundValue)
      constOperand = addOp.getLhs();
    else
      return finish(false);
    int64_t constValue = 0;
    if (!resolveMLIRValue(constOperand, state, constValue) || constValue < 0)
      return finish(false);
    bound = constValue;
    return finish(true);
  }

  // Not a recognized "clamped by constant" shape from this branch — no
  // constraint derivable here (the min-combination above still lets a
  // sibling branch supply the bound).
  return finish(false);
}

static bool estimateForTripCountUpperBound(mlir::scf::ForOp forOp,
                                            const AnalysisState &state,
                                            int64_t &upperBoundTripCount) {
  int64_t step = 0;
  if (!resolveMLIRValue(forOp.getStep(), state, step) || step <= 0)
    return false;
  int64_t bound = 0;
  llvm::SmallDenseSet<mlir::Value, 8> visited;
  if (!estimateForTripCountUpperBoundImpl(forOp.getUpperBound(),
                                          forOp.getLowerBound(), state, bound,
                                          visited))
    return false;
  upperBoundTripCount = std::max<int64_t>(1, ceilDiv(bound, step));
  return true;
}

static bool captureConstant(mlir::Operation *op, AnalysisState &state) {
  auto constantOp = llvm::dyn_cast<mlir::arith::ConstantOp>(op);
  if (!constantOp || op->getNumResults() != 1)
    return false;

  mlir::Attribute valueAttr = constantOp.getValue();
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(valueAttr)) {
    state.constants[op->getResult(0)] = intAttr.getInt();
    return true;
  }
  return false;
}

static std::optional<int64_t> evaluateAffineExpr(mlir::AffineExpr expr,
                                                 mlir::AffineMap map,
                                                 llvm::ArrayRef<int64_t> inputs);

static bool resolveMLIRValue(mlir::Value value, const AnalysisState &state,
                             int64_t &resolved);

static std::optional<int64_t> evaluateAffineExpr(mlir::AffineExpr expr,
                                                 mlir::AffineMap map,
                                                 llvm::ArrayRef<int64_t> inputs) {
  if (auto constant = llvm::dyn_cast<mlir::AffineConstantExpr>(expr))
    return constant.getValue();
  if (auto dim = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
    unsigned pos = dim.getPosition();
    if (pos < inputs.size())
      return inputs[pos];
    return std::nullopt;
  }
  if (auto symbol = llvm::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
    unsigned pos = map.getNumDims() + symbol.getPosition();
    if (pos < inputs.size())
      return inputs[pos];
    return std::nullopt;
  }
  if (auto binary = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs = evaluateAffineExpr(binary.getLHS(), map, inputs);
    auto rhs = evaluateAffineExpr(binary.getRHS(), map, inputs);
    if (!lhs || !rhs)
      return std::nullopt;
    switch (binary.getKind()) {
    case mlir::AffineExprKind::Add:
      return *lhs + *rhs;
    case mlir::AffineExprKind::Mul:
      return *lhs * *rhs;
    case mlir::AffineExprKind::Mod:
      if (*rhs == 0)
        return std::nullopt;
      return *lhs % *rhs;
    case mlir::AffineExprKind::FloorDiv:
      if (*rhs == 0)
        return std::nullopt;
      return llvm::divideFloorSigned(*lhs, *rhs);
    case mlir::AffineExprKind::CeilDiv:
      if (*rhs == 0)
        return std::nullopt;
      return llvm::divideCeilSigned(*lhs, *rhs);
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

static bool captureDerivedScalarValue(mlir::Operation *op, AnalysisState &state) {
  if (op->getNumResults() != 1)
    return false;

  auto recordValue = [&](int64_t value) {
    state.boundValues[op->getResult(0)] = value;
    return true;
  };

  if (op->getName().getStringRef() == "hivm.hir.get_block_idx" ||
      op->getName().getStringRef() == "get_block_idx") {
    auto it = state.argBindings.find("pid_x");
    if (it == state.argBindings.end())
      it = state.argBindings.find("program_id_x");
    if (it != state.argBindings.end())
      return recordValue(it->second);
    return false;
  }

  if (auto castOp = llvm::dyn_cast<mlir::arith::IndexCastOp>(op)) {
    int64_t resolved = 0;
    if (resolveMLIRValue(castOp.getIn(), state, resolved))
      return recordValue(resolved);
    return false;
  }

  if (auto castOp = llvm::dyn_cast<mlir::arith::IndexCastUIOp>(op)) {
    int64_t resolved = 0;
    if (resolveMLIRValue(castOp.getIn(), state, resolved))
      return recordValue(resolved);
    return false;
  }

  if (auto truncOp = llvm::dyn_cast<mlir::arith::TruncIOp>(op)) {
    int64_t resolved = 0;
    if (resolveMLIRValue(truncOp.getIn(), state, resolved))
      return recordValue(resolved);
    return false;
  }

  if (auto extOp = llvm::dyn_cast<mlir::arith::ExtSIOp>(op)) {
    int64_t resolved = 0;
    if (resolveMLIRValue(extOp.getIn(), state, resolved))
      return recordValue(resolved);
    return false;
  }

  if (auto addOp = llvm::dyn_cast<mlir::arith::AddIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(addOp.getLhs(), state, lhs) &&
        resolveMLIRValue(addOp.getRhs(), state, rhs))
      return recordValue(lhs + rhs);
    return false;
  }

  if (auto subOp = llvm::dyn_cast<mlir::arith::SubIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(subOp.getLhs(), state, lhs) &&
        resolveMLIRValue(subOp.getRhs(), state, rhs))
      return recordValue(lhs - rhs);
    return false;
  }

  if (auto mulOp = llvm::dyn_cast<mlir::arith::MulIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(mulOp.getLhs(), state, lhs) &&
        resolveMLIRValue(mulOp.getRhs(), state, rhs))
      return recordValue(lhs * rhs);
    return false;
  }

  if (auto divOp = llvm::dyn_cast<mlir::arith::DivSIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(divOp.getLhs(), state, lhs) &&
        resolveMLIRValue(divOp.getRhs(), state, rhs) && rhs != 0)
      return recordValue(lhs / rhs);
    return false;
  }

  if (auto remOp = llvm::dyn_cast<mlir::arith::RemSIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(remOp.getLhs(), state, lhs) &&
        resolveMLIRValue(remOp.getRhs(), state, rhs) && rhs != 0)
      return recordValue(lhs % rhs);
    return false;
  }

  if (auto minOp = llvm::dyn_cast<mlir::arith::MinSIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(minOp.getLhs(), state, lhs) &&
        resolveMLIRValue(minOp.getRhs(), state, rhs))
      return recordValue(std::min(lhs, rhs));
    return false;
  }

  if (auto maxOp = llvm::dyn_cast<mlir::arith::MaxSIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(maxOp.getLhs(), state, lhs) &&
        resolveMLIRValue(maxOp.getRhs(), state, rhs))
      return recordValue(std::max(lhs, rhs));
    return false;
  }

  if (auto orOp = llvm::dyn_cast<mlir::arith::OrIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (resolveMLIRValue(orOp.getLhs(), state, lhs) &&
        resolveMLIRValue(orOp.getRhs(), state, rhs))
      return recordValue(lhs | rhs);
    return false;
  }

  if (auto cmpOp = llvm::dyn_cast<mlir::arith::CmpIOp>(op)) {
    int64_t lhs = 0, rhs = 0;
    if (!resolveMLIRValue(cmpOp.getLhs(), state, lhs) ||
        !resolveMLIRValue(cmpOp.getRhs(), state, rhs))
      return false;
    bool result = false;
    switch (cmpOp.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
      result = lhs == rhs;
      break;
    case mlir::arith::CmpIPredicate::ne:
      result = lhs != rhs;
      break;
    case mlir::arith::CmpIPredicate::slt:
      result = lhs < rhs;
      break;
    case mlir::arith::CmpIPredicate::sle:
      result = lhs <= rhs;
      break;
    case mlir::arith::CmpIPredicate::sgt:
      result = lhs > rhs;
      break;
    case mlir::arith::CmpIPredicate::sge:
      result = lhs >= rhs;
      break;
    case mlir::arith::CmpIPredicate::ult:
      result = static_cast<uint64_t>(lhs) < static_cast<uint64_t>(rhs);
      break;
    case mlir::arith::CmpIPredicate::ule:
      result = static_cast<uint64_t>(lhs) <= static_cast<uint64_t>(rhs);
      break;
    case mlir::arith::CmpIPredicate::ugt:
      result = static_cast<uint64_t>(lhs) > static_cast<uint64_t>(rhs);
      break;
    case mlir::arith::CmpIPredicate::uge:
      result = static_cast<uint64_t>(lhs) >= static_cast<uint64_t>(rhs);
      break;
    }
    return recordValue(result ? 1 : 0);
  }

  if (auto selectOp = llvm::dyn_cast<mlir::arith::SelectOp>(op)) {
    int64_t cond = 0, trueValue = 0, falseValue = 0;
    if (resolveMLIRValue(selectOp.getCondition(), state, cond) &&
        resolveMLIRValue(selectOp.getTrueValue(), state, trueValue) &&
        resolveMLIRValue(selectOp.getFalseValue(), state, falseValue))
      return recordValue(cond != 0 ? trueValue : falseValue);
    return false;
  }

  if (auto affineApply = llvm::dyn_cast<mlir::affine::AffineApplyOp>(op)) {
    llvm::SmallVector<int64_t, 8> inputs;
    inputs.reserve(affineApply.getOperands().size());
    for (mlir::Value operand : affineApply.getOperands()) {
      int64_t resolved = 0;
      if (!resolveMLIRValue(operand, state, resolved))
        return false;
      inputs.push_back(resolved);
    }
    auto result =
        evaluateAffineExpr(affineApply.getAffineMap().getResult(0),
                           affineApply.getAffineMap(), inputs);
    if (result)
      return recordValue(*result);
    return false;
  }

  return false;
}

static std::string getOrCreateBufferRoot(mlir::Value value, AnalysisState &state) {
  auto it = state.bufferRoots.find(value);
  if (it != state.bufferRoots.end())
    return it->second;
  std::string root = renderValueToken(value);
  state.bufferRoots[value] = root;
  return root;
}

static void captureBufferMetadata(mlir::Operation *op, AnalysisState &state) {
#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
  if (auto markOp = llvm::dyn_cast<mlir::annotation::MarkOp>(op)) {
    mlir::Value src = markOp.getSrc();
    if (markOp.isAnnotatedByStaticAttr("hivm.multi_buffer")) {
      if (auto intAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
              markOp.getStaticAttrValue("hivm.multi_buffer"))) {
        std::string root = getOrCreateBufferRoot(src, state);
        state.bufferSlots[root] = std::max<int64_t>(1, intAttr.getInt());
      }
    }
    return;
  }
#endif

  if (auto subviewOp = llvm::dyn_cast<mlir::memref::SubViewOp>(op)) {
    auto it = state.bufferRoots.find(subviewOp.getSource());
    if (it != state.bufferRoots.end())
      state.bufferRoots[subviewOp.getResult()] = it->second;
    return;
  }
  if (auto castOp = llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(op)) {
    auto it = state.bufferRoots.find(castOp.getSource());
    if (it != state.bufferRoots.end())
      state.bufferRoots[castOp.getResult()] = it->second;
    return;
  }
  if (auto castOp = llvm::dyn_cast<mlir::memref::CastOp>(op)) {
    auto it = state.bufferRoots.find(castOp.getSource());
    if (it != state.bufferRoots.end())
      state.bufferRoots[castOp.getResult()] = it->second;
    return;
  }
}

static void dedupeBufferList(std::vector<std::string> &buffers) {
  std::sort(buffers.begin(), buffers.end());
  buffers.erase(std::unique(buffers.begin(), buffers.end()), buffers.end());
}

static bool isLikelyWritingOp(llvm::StringRef opName) {
  return opName == "load" || opName == "copy" || opName == "vadd" ||
         opName == "vmul" || opName == "vcast" || opName == "vbrc" ||
         opName == "vreduce" || opName == "fixpipe" || opName == "nd2nz" ||
         opName == "nz2nd" || opName == "mmadL1" || opName == "matmul" ||
         opName == "mix_matmul" || opName == "mix_group_matmul";
}

static void attachBufferAccessMetadata(mlir::Operation *op, ParsedOp &parsed,
                                       AnalysisState &state) {
  if (parsed.op.opName == "pointer_cast" && op->getNumResults() == 1 &&
      mlir::isa<mlir::MemRefType>(op->getResult(0).getType())) {
    getOrCreateBufferRoot(op->getResult(0), state);
  }

  llvm::SmallVector<std::string, 4> rootedOperands;
  for (mlir::Value operand : op->getOperands()) {
    auto it = state.bufferRoots.find(operand);
    if (it != state.bufferRoots.end())
      rootedOperands.push_back(it->second);
  }

  if (rootedOperands.empty())
    return;

  if (isLikelyWritingOp(parsed.op.opName)) {
    parsed.op.writeBuffers.push_back(rootedOperands.back());
    parsed.op.multiBufferSlots =
        std::max<int64_t>(parsed.op.multiBufferSlots,
                          state.bufferSlots[rootedOperands.back()]);
    rootedOperands.pop_back();
  }

  for (const std::string &root : rootedOperands)
    parsed.op.readBuffers.push_back(root);
  dedupeBufferList(parsed.op.readBuffers);
  dedupeBufferList(parsed.op.writeBuffers);
}

static std::pair<std::string, std::string> parseLoadStoreSpaces(llvm::StringRef line) {
  llvm::SmallVector<std::string, 2> spaces;
  size_t pos = 0;
  while (spaces.size() < 2) {
    pos = line.find("#hivm.address_space<", pos);
    if (pos == llvm::StringRef::npos)
      break;
    pos += strlen("#hivm.address_space<");
    size_t end = line.find('>', pos);
    if (end == llvm::StringRef::npos)
      break;
    spaces.push_back(canonicalizeAddressSpace(line.slice(pos, end)));
    pos = end + 1;
  }
  if (spaces.size() < 2)
    return {"", ""};
  return {spaces[0], spaces[1]};
}

static std::map<std::string, int64_t> parseArgBindings(llvm::StringRef bindings) {
  std::map<std::string, int64_t> result;
  llvm::SmallVector<llvm::StringRef, 8> entries;
  bindings.split(entries, ',', -1, false);
  for (llvm::StringRef entry : entries) {
    auto kv = trim(entry).split('=');
    if (kv.first.empty() || kv.second.empty())
      continue;
    int64_t value = 0;
    if (trim(kv.second).getAsInteger(10, value))
      continue;
    result[trim(kv.first).str()] = value;
  }
  return result;
}

static std::map<std::string, std::string>
parseStringArgBindings(llvm::StringRef bindings) {
  std::map<std::string, std::string> result;
  llvm::SmallVector<llvm::StringRef, 8> entries;
  bindings.split(entries, ',', -1, false);
  for (llvm::StringRef entry : entries) {
    auto kv = trim(entry).split('=');
    if (kv.first.empty() || kv.second.empty())
      continue;
    result[trim(kv.first).str()] = trim(kv.second).str();
  }
  return result;
}

static int64_t getBindingOr(const std::map<std::string, int64_t> &bindings,
                            std::initializer_list<llvm::StringRef> keys,
                            int64_t fallback = -1) {
  for (llvm::StringRef key : keys) {
    auto it = bindings.find(key.str());
    if (it != bindings.end())
      return it->second;
  }
  return fallback;
}

static KernelMode parseKernelModeForLaunch(llvm::StringRef mode) {
  if (mode.equals_insensitive("aiv") || mode.equals_insensitive("vector"))
    return KernelMode::AIV;
  if (mode.equals_insensitive("simd"))
    return KernelMode::SIMD;
  if (mode.equals_insensitive("aic") || mode.equals_insensitive("cube"))
    return KernelMode::AIC;
  if (mode.equals_insensitive("mix"))
    return KernelMode::Mix;
  if (mode.equals_insensitive("simt"))
    return KernelMode::SIMT;
  if (mode.equals_insensitive("simd_simt_mix"))
    return KernelMode::SIMDSIMTMix;
  return KernelMode::Unknown;
}

static KernelMode inferKernelModeForLaunch(const HIVMAnalysisReport &report) {
  bool hasAIC = false;
  bool hasAIV = false;
  for (const HIVMOp &op : report.operations) {
    llvm::StringRef coreType(op.coreType);
    if (coreType.contains_insensitive("AIC") ||
        op.pipe == HIVMPipe::Cube || op.pipe == HIVMPipe::CubeMTE2 ||
        op.pipe == HIVMPipe::FixPipe || op.pipe == HIVMPipe::MTE1)
      hasAIC = true;
    if (coreType.contains_insensitive("AIV") ||
        op.pipe == HIVMPipe::Vector || op.pipe == HIVMPipe::VectorMTE2 ||
        op.pipe == HIVMPipe::MTE3)
      hasAIV = true;
  }
  if (hasAIC && hasAIV)
    return KernelMode::Mix;
  if (hasAIC)
    return KernelMode::AIC;
  if (hasAIV)
    return KernelMode::AIV;
  return KernelMode::Unknown;
}

static void applyKernelLaunchOverhead(
    HIVMAnalysisReport &report, const HardwareConfig &config,
    const std::map<std::string, int64_t> &bindings,
    llvm::StringRef rawBindings) {
  report.bodyCycles = report.weightedCycles;
  KernelLaunchContext ctx;
  ctx.bodyCycles = report.bodyCycles;
  ctx.opCount = report.opCount;
  ctx.blockDim = getBindingOr(
      bindings, {"block_dim", "blockDim", "block_num", "blockNum"});
  ctx.usingPrograms = getBindingOr(
      bindings, {"using_programs", "usingPrograms", "num_programs",
                 "numPrograms"});
  ctx.numWaves = getBindingOr(bindings, {"num_waves", "numWaves"});
  ctx.mode = inferKernelModeForLaunch(report);
  auto stringBindings = parseStringArgBindings(rawBindings);
  auto modeIt = stringBindings.find("kernel_mode");
  if (modeIt != stringBindings.end()) {
    KernelMode explicitMode = parseKernelModeForLaunch(modeIt->second);
    if (explicitMode != KernelMode::Unknown)
      ctx.mode = explicitMode;
  }
  for (const HIVMOp &op : report.operations) {
    ctx.hasVector |= op.pipe == HIVMPipe::Vector ||
                     op.pipe == HIVMPipe::VectorMTE2;
    ctx.hasCube |= op.pipe == HIVMPipe::Cube ||
                   op.pipe == HIVMPipe::CubeMTE2 ||
                   op.pipe == HIVMPipe::FixPipe || op.pipe == HIVMPipe::MTE1;
    ctx.hasMTE |= op.pipe == HIVMPipe::VectorMTE2 ||
                  op.pipe == HIVMPipe::CubeMTE2 ||
                  op.pipe == HIVMPipe::MTE3 ||
                  op.pipe == HIVMPipe::FixPipe || op.pipe == HIVMPipe::MTE1;
  }

  KernelLaunchEstimate launch = config.estimateKernelLaunchOverhead(ctx);
  report.kernelLaunchOverheadCycles = launch.totalCycles;
  report.predictedTotalCycles = report.bodyCycles + launch.totalCycles;
  report.kernelLaunchBlockDim = launch.blockDim;
  report.kernelLaunchNumWaves = launch.numWaves;
  report.kernelLaunchModel = launch.model;
}

static bool isZeroCostOp(llvm::StringRef opName) {
  return opName == "pointer_cast" || opName == "convert_layout";
}

static bool isCubeOpName(llvm::StringRef opName) {
  return opName == "matmul" || opName == "mix_matmul" ||
         opName == "mix_group_matmul" || opName == "mmadL1";
}

static int64_t estimateDuration(const ParsedOp &parsed, const HardwareConfig &config) {
  llvm::StringRef opName = parsed.op.opName;
  llvm::StringRef line = parsed.op.text;

  auto startupForPipe = [&](HIVMPipe pipe) -> int64_t {
    switch (pipe) {
    case HIVMPipe::Vector:
      return config.getVectorStartupLatency();
    case HIVMPipe::VectorMTE2:
    case HIVMPipe::CubeMTE2:
      return config.getMTE2StartupLatency();
    case HIVMPipe::MTE3:
      return config.getMTE3StartupLatency();
    case HIVMPipe::FixPipe:
      return config.getFixPipeStartupLatency();
    case HIVMPipe::Cube:
      return config.getCubeStartupLatency();
    case HIVMPipe::MTE1:
      return std::max<int64_t>(1, config.getMTE2StartupLatency() / 2);
    case HIVMPipe::Scalar:
      return 1;
    case HIVMPipe::All:
      return config.getPipeBarrierCyclesPerIter();
    case HIVMPipe::Unknown:
      return 1;
    }
    return 1;
  };

  auto estimateSpaceTransferCycles =
      [&](llvm::StringRef srcSpace, llvm::StringRef dstSpace, int64_t bytes,
          int64_t startup) -> int64_t {
    int64_t clampedBytes = std::max<int64_t>(bytes, 1);
    auto chooseBandwidth = [&](llvm::StringRef name) {
      double bw = config.getMemoryBandwidthBytesPerCycle(name);
      return bw > 0.0 ? bw : 0.0;
    };
    double srcBw = chooseBandwidth(srcSpace);
    double dstBw = chooseBandwidth(dstSpace);
    double bw = 0.0;
    if (srcBw > 0.0 && dstBw > 0.0)
      bw = std::min(srcBw, dstBw);
    else
      bw = std::max(srcBw, dstBw);
    if (bw <= 0.0)
      bw = static_cast<double>(config.getVectorWidthBytes());
    int latency = std::max(config.getMemoryLatencyCycles(srcSpace),
                           config.getMemoryLatencyCycles(dstSpace));
    int64_t transfer = std::max<int64_t>(
        1, static_cast<int64_t>(std::ceil(clampedBytes / bw)));
    return startup + latency + transfer;
  };

  if (isZeroCostOp(opName))
    return 0;

  if (opName == "set_mask_norm" || opName == "get_block_idx")
    return 1;

  if (opName == "set_flag")
    return config.getSyncOpCycles("set_flag", 1);
  if (opName == "wait_flag")
    return config.getSyncOpCycles("wait_flag", 2);
  if (opName == "sync_block_set") {
    HIVMPipe sender = parsed.senderPipe;
    HIVMPipe receiver = parsed.receiverPipe;
    int64_t crossPipe = sender != receiver ? 1 : 0;
    int64_t crossCore = parsed.op.coreType == "CUBE" ? 2 : 1;
    return 1 + crossPipe + crossCore;
  }
  if (opName == "sync_block_wait") {
    HIVMPipe receiver = parsed.receiverPipe;
    int64_t base = std::max<int64_t>(2, startupForPipe(receiver) / 8);
    if (parsed.op.coreType == "CUBE")
      base += 2;
    return base;
  }
  if (opName == "sync_block")
    return std::max<int64_t>(config.getPipeBarrierCyclesPerIter(),
                             config.getVectorStartupLatency());

  // pipe_barrier drains the in-flight instructions on the target pipe.
  // Cost is the pipeline depth, NOT the full startup latency.
  if (opName == "pipe_barrier") {
    if (parsed.barrierPipes.empty())
      return 8;
    if (llvm::is_contained(parsed.barrierPipes, HIVMPipe::All))
      return 64;
    HIVMPipe pipe = parsed.barrierPipes.front();
    switch (pipe) {
    case HIVMPipe::Vector:
      return 4;
    case HIVMPipe::VectorMTE2:
    case HIVMPipe::CubeMTE2:
      return 16;
    case HIVMPipe::MTE3:
      return 16;
    case HIVMPipe::FixPipe:
      return 8;
    case HIVMPipe::Cube:
      return 8;
    default:
      return 8;
    }
  }

  if (opName == "load") {
    int64_t bytes = parsed.op.bytes;
    auto spaces = parseLoadStoreSpaces(line);
    if (spaces.first == "gm" && spaces.second == "ub")
      return config.getMTE2StartupLatency() +
             config.estimateMemoryCycles("vector_mte2", std::max<int64_t>(bytes, 1));
    if (spaces.first == "gm" && spaces.second == "l1")
      return config.getMTE2StartupLatency() +
             config.estimateMemoryCycles("cube_mte2", std::max<int64_t>(bytes, 1));
    return config.getMTE2StartupLatency() +
           config.estimateMemoryCyclesWithLatency("hbm", std::max<int64_t>(bytes, 1));
  }

  if (opName == "store") {
    int64_t bytes = parsed.op.bytes;
    auto spaces = parseLoadStoreSpaces(line);
    if (spaces.first == "ub" && spaces.second == "gm")
      return config.getMTE3StartupLatency() +
             config.estimateMemoryCycles("mte3", std::max<int64_t>(bytes, 1));
    if (spaces.first == "l0c" && spaces.second == "gm")
      return config.getFixPipeStartupLatency() +
             config.estimateMemoryCycles("fixpipe", std::max<int64_t>(bytes, 1));
    return config.getMTE3StartupLatency() +
           config.estimateMemoryCyclesWithLatency("hbm", std::max<int64_t>(bytes, 1));
  }

  if (opName == "copy") {
    int64_t bytes = std::max<int64_t>(parsed.op.bytes, config.getVectorWidthBytes());
    auto spaces = parseLoadStoreSpaces(line);
    if (spaces.first == "ub" && spaces.second == "ub")
      return estimateSpaceTransferCycles("ub", "ub", bytes,
                                         config.getVectorStartupLatency());
    if (spaces.first == "gm" && spaces.second == "l1")
      return config.getMTE2StartupLatency() +
             config.estimateMemoryCycles("cube_mte2", bytes);
    if (spaces.first == "ub" && spaces.second == "l1")
      return estimateSpaceTransferCycles("ub", "l1", bytes,
                                         config.getMTE3StartupLatency());
    if (spaces.first == "l0c" && spaces.second == "gm")
      return config.getFixPipeStartupLatency() +
             config.estimateMemoryCycles("fixpipe", bytes);
    return estimateSpaceTransferCycles("ub", "ub", bytes, startupForPipe(parsed.op.pipe));
  }

  auto vectorCycles = [&](int opCost) -> int64_t {
    int64_t elems = std::max<int64_t>(parsed.op.elements, config.getVectorWidthElements());
    return config.getVectorStartupLatency() +
           ceilDiv(elems, config.getVectorWidthElements()) * opCost;
  };

  auto isVectorALUOp = [&](llvm::StringRef name) {
    return name == "vadd" || name == "vmul" || name == "vsub" ||
           name == "vmax" || name == "vmin" ||
           name == "vcast" || name == "vexp" || name == "vdiv" ||
           name == "vlog" || name == "vsqrt" || name == "vrsqrt" ||
           name == "vtanh" || name == "vsigmoid" || name == "vreduce" ||
           name == "vbrc" || name == "vcmp" || name == "vsel" ||
           name == "vand" || name == "vor" || name == "vnot" ||
           name == "varange" || name == "copy";
  };
  if (isVectorALUOp(opName))
    return vectorCycles(config.getVectorOpCyclesPerInstruction(opName));
  if (opName == "vcall")
    return vectorCycles(config.getVectorOpCyclesPerInstruction("vmul"));
  if (opName == "fixpipe") {
    int64_t bytes = std::max<int64_t>(parsed.op.bytes, 1);
    auto spaces = parseLoadStoreSpaces(line);
    if (spaces.second == "ub")
      return estimateSpaceTransferCycles("l0c", "ub", bytes,
                                         config.getFixPipeStartupLatency());
    if (spaces.second == "gm")
      return config.getFixPipeStartupLatency() +
             config.estimateMemoryCycles("fixpipe", bytes);
    return config.getFixPipeStartupLatency() +
           config.estimateMemoryCycles("fixpipe", bytes);
  }

  if (opName == "nd2nz") {
    return estimateND2NZCycles(parsed, config);
  }

  if (opName == "nz2nd") {
    int64_t bytes = std::max<int64_t>(parsed.op.bytes, 1);
    return config.getMTE3StartupLatency() +
           config.estimateMemoryCycles("mte3", bytes);
  }

  if (isCubeOpName(opName)) {
    int64_t totalElements = 0;
    size_t memrefCount = 0;
    size_t searchPos = 0;
    while ((searchPos = line.find("memref<", searchPos)) != llvm::StringRef::npos &&
           memrefCount < 3) {
      totalElements += parseMemRefElementCount(line.drop_front(searchPos));
      ++memrefCount;
      searchPos += 7;
    }
    if (memrefCount >= 3) {
      int64_t lhs = std::max<int64_t>(1, totalElements / 3);
      int64_t rhs = std::max<int64_t>(1, totalElements / 3);
      int64_t out = std::max<int64_t>(1, totalElements - lhs - rhs);
      int64_t M = std::max<int64_t>(1, static_cast<int64_t>(std::sqrt(out)));
      int64_t N = std::max<int64_t>(1, out / M);
      int64_t K = std::max<int64_t>(1, lhs / M);
      return config.getCubeStartupLatency() + config.estimateCubeCycles(M, N, K);
    }
    return config.getCubeStartupLatency() + 16;
  }

  return 1;
}

static void addLatestPipeDependency(HIVMPipe pipe,
                                    const std::map<HIVMPipe, size_t> &latestPipeProducer,
                                    ParsedOp &parsed) {
  auto it = latestPipeProducer.find(pipe);
  if (it != latestPipeProducer.end())
    parsed.op.dependsOn.push_back(it->second);
}

static bool pipeBelongsToCore(HIVMPipe pipe, llvm::StringRef coreType) {
  bool isCubeCore = coreType == "CUBE" || coreType == "AIC";
  bool isVectorCore = coreType == "VECTOR" || coreType == "AIV";
  if (isCubeCore) {
    return pipe == HIVMPipe::Cube || pipe == HIVMPipe::MTE1 ||
           pipe == HIVMPipe::CubeMTE2 || pipe == HIVMPipe::FixPipe ||
           pipe == HIVMPipe::Scalar;
  }
  if (isVectorCore) {
    return pipe == HIVMPipe::Vector || pipe == HIVMPipe::VectorMTE2 ||
           pipe == HIVMPipe::MTE3 || pipe == HIVMPipe::Scalar;
  }
  return false;
}

static llvm::SmallVector<HIVMPipe, 5> getCoreBarrierPipes(llvm::StringRef coreType) {
  if (coreType == "CUBE" || coreType == "AIC")
    return {HIVMPipe::Cube, HIVMPipe::MTE1, HIVMPipe::CubeMTE2,
            HIVMPipe::FixPipe, HIVMPipe::Scalar};
  if (coreType == "VECTOR" || coreType == "AIV")
    return {HIVMPipe::Vector, HIVMPipe::VectorMTE2, HIVMPipe::MTE3,
            HIVMPipe::Scalar};
  return {};
}

static int64_t ceilToI64(double value) {
  return static_cast<int64_t>(std::ceil(value));
}

static std::string defaultCostSubpipeForOp(const HIVMOp &op) {
  if (op.isSyncOp || op.isBarrier)
    return "sync";
  switch (op.pipe) {
  case HIVMPipe::Vector:
    return "vector";
  case HIVMPipe::VectorMTE2:
  case HIVMPipe::CubeMTE2:
  case HIVMPipe::MTE3:
  case HIVMPipe::MTE1:
    return "mte";
  case HIVMPipe::FixPipe:
    return "fixpipe";
  case HIVMPipe::Cube:
    return "cube";
  case HIVMPipe::Scalar:
    return "scalar";
  case HIVMPipe::All:
    return "sync";
  case HIVMPipe::Unknown:
    return "";
  }
  return "";
}

static bool isSyncSetOp(llvm::StringRef opName) {
  return opName == "set_flag" || opName == "sync_block_set";
}

static bool isSyncWaitOp(llvm::StringRef opName) {
  return opName == "wait_flag" || opName == "sync_block_wait";
}

static void applyTimingModel(HIVMOp &op, const HardwareConfig &config) {
  int64_t issue = std::max<int64_t>(0, op.duration);
  int64_t latency = issue;
  op.calibratedCost = false;
  op.costSource = op.costSource.empty() ? "heuristic" : op.costSource;

  llvm::StringRef pipeName = HIVMAnalyzer::stringifyPipe(op.pipe);
  if (auto cost = config.lookupOpcodeCycleCost(pipeName, op.opName)) {
    op.calibratedCost = true;
    if (!cost->source.empty())
      op.costSource = cost->source;
    else
      op.costSource = "opcode_calibration";
    op.costSubpipe = cost->subpipe.empty() ? defaultCostSubpipeForOp(op)
                                           : cost->subpipe;

    if (cost->hasStartupCycles || cost->hasCyclesPerByte) {
      double startup = cost->hasStartupCycles ? cost->startupCycles : 0.0;
      double cpb = cost->hasCyclesPerByte ? cost->cyclesPerByte : 0.0;
      issue = std::max<int64_t>(
          1, ceilToI64(startup + cpb * std::max<int64_t>(op.bytes, 1)));
      latency = issue;
    } else if (cost->hasCycles) {
      if (op.pipe == HIVMPipe::Vector) {
        int64_t elems =
            std::max<int64_t>(op.elements, config.getVectorWidthElements());
        int64_t vectorRepeats =
            ceilDiv(elems, std::max<int>(1, config.getVectorWidthElements()));
        issue = std::max<int64_t>(
            1, ceilToI64(config.getVectorStartupLatency() +
                         vectorRepeats * cost->cycles));
      } else {
        issue = std::max<int64_t>(0, ceilToI64(cost->cycles));
      }
      latency = cost->hasLatency ? std::max<int64_t>(issue, ceilToI64(cost->latency))
                                 : issue;
    }
  }

  if (op.isSyncOp && (isSyncSetOp(op.opName) || isSyncWaitOp(op.opName))) {
    int64_t eventLatency = std::max<int64_t>(1, latency);
    int64_t syncIssue = std::min<int64_t>(eventLatency, 16);
    syncIssue = std::max<int64_t>(1, syncIssue);
    issue = syncIssue;
    latency = isSyncSetOp(op.opName) ? eventLatency : syncIssue;
  }

  op.issueDuration = issue;
  op.dependencyLatency = std::max<int64_t>(latency, issue);
  op.duration = op.issueDuration;
}

static void ingestParsedOp(const ParsedOp &parsed, AnalysisState &state,
                           HIVMAnalysisReport &report, const HardwareConfig &config) {
  ParsedOp mutableParsed = parsed;
  if (mutableParsed.syncIdValue) {
    std::string canonicalSyncId =
        canonicalizeSyncId(mutableParsed.syncIdValue, state);
    if (!canonicalSyncId.empty())
      mutableParsed.op.eventId = canonicalSyncId;
  }
  EventKey opEventKey{mutableParsed.senderPipe, mutableParsed.receiverPipe,
                      mutableParsed.op.eventId};

  // Enforce program order within each pipe: every op depends on the previous
  // op on the same pipe, matching hardware sequential execution semantics.
  if (mutableParsed.op.pipe != HIVMPipe::Unknown &&
      mutableParsed.op.pipe != HIVMPipe::All)
    addLatestPipeDependency(mutableParsed.op.pipe, state.latestPipeProducer,
                            mutableParsed);

  mutableParsed.op.readBufferVersions.reserve(mutableParsed.op.readBuffers.size());
  for (const std::string &root : mutableParsed.op.readBuffers) {
    auto it = state.bufferVersions.find(root);
    mutableParsed.op.readBufferVersions.push_back(
        it != state.bufferVersions.end() ? it->second : 0);
  }
  mutableParsed.op.writeBufferVersions.reserve(
      mutableParsed.op.writeBuffers.size());
  for (const std::string &root : mutableParsed.op.writeBuffers) {
    int64_t nextVersion = state.bufferVersions[root] + 1;
    state.bufferVersions[root] = nextVersion;
    mutableParsed.op.writeBufferVersions.push_back(nextVersion);
  }

  if (mutableParsed.op.isBarrier) {
    if (mutableParsed.barrierPipes.empty() ||
        llvm::is_contained(mutableParsed.barrierPipes, HIVMPipe::All)) {
      for (const auto &entry : state.latestPipeProducer) {
        if (mutableParsed.op.pipe == HIVMPipe::All &&
            !mutableParsed.op.coreType.empty() &&
            !pipeBelongsToCore(entry.first, mutableParsed.op.coreType))
          continue;
        mutableParsed.op.dependsOn.push_back(entry.second);
      }
    } else {
      for (HIVMPipe pipe : mutableParsed.barrierPipes)
        addLatestPipeDependency(pipe, state.latestPipeProducer, mutableParsed);
    }
  }

  if (mutableParsed.op.opName == "wait_flag" ||
      mutableParsed.op.opName == "sync_block_wait") {
    auto genIt = state.eventGenerations.find(opEventKey);
    if (genIt != state.eventGenerations.end())
      mutableParsed.op.eventGeneration = genIt->second;
    auto eventIt = state.eventProducers.find(opEventKey);
    if (eventIt != state.eventProducers.end()) {
      mutableParsed.op.dependsOn.push_back(eventIt->second);
      mutableParsed.op.eventDependsOn.push_back(eventIt->second);
    }
    addLatestPipeDependency(mutableParsed.op.pipe, state.latestPipeProducer,
                            mutableParsed);
  }

  std::vector<HIVMOp> expandedOps = expandMacroOp(mutableParsed, config);
  size_t firstExpandedId = report.operations.size();
  size_t previousExpandedId = std::numeric_limits<size_t>::max();
  for (HIVMOp &expanded : expandedOps) {
    applyTimingModel(expanded, config);
    expanded.id = report.operations.size();
    if (previousExpandedId != std::numeric_limits<size_t>::max())
      expanded.dependsOn.push_back(previousExpandedId);
    // Expanded sub-ops may land on a different pipe than the parsed op;
    // enforce program order on each sub-op's actual pipe.
    if (expanded.pipe != mutableParsed.op.pipe &&
        expanded.pipe != HIVMPipe::Unknown &&
        expanded.pipe != HIVMPipe::All) {
      auto it = state.latestPipeProducer.find(expanded.pipe);
      if (it != state.latestPipeProducer.end())
        expanded.dependsOn.push_back(it->second);
    }
    previousExpandedId = expanded.id;
    report.operations.push_back(std::move(expanded));
  }

  if (mutableParsed.op.opName == "set_flag" ||
      mutableParsed.op.opName == "sync_block_set") {
    int64_t nextGeneration = state.eventGenerations[opEventKey] + 1;
    state.eventGenerations[opEventKey] = nextGeneration;
    mutableParsed.op.eventGeneration = nextGeneration;
    for (size_t idx = firstExpandedId;
         idx <= previousExpandedId && idx < report.operations.size(); ++idx)
      report.operations[idx].eventGeneration = nextGeneration;
    state.eventProducers[opEventKey] = firstExpandedId;
  }

  for (mlir::Value result : mutableParsed.mlirResults)
    state.valueProducers[result] = previousExpandedId;

  for (size_t idx = firstExpandedId;
       idx <= previousExpandedId && idx < report.operations.size(); ++idx) {
    HIVMOp &expanded = report.operations[idx];
    if (expanded.pipe == HIVMPipe::All && expanded.isBarrier) {
      // A PIPE_ALL barrier blocks all pipes on its core.  Register it as the
      // latest producer for every pipe in that core so subsequent ops on any
      // of those pipes depend on the barrier completing.
      for (HIVMPipe p : getCoreBarrierPipes(expanded.coreType))
        state.latestPipeProducer[p] = expanded.id;
    } else if (expanded.pipe != HIVMPipe::All &&
               expanded.pipe != HIVMPipe::Unknown) {
      state.latestPipeProducer[expanded.pipe] = expanded.id;
    }
  }
}

static void analyzeParsedOperation(mlir::Operation *op, int64_t loopMultiplier,
                                   AnalysisState &state,
                                   HIVMAnalysisReport &report,
                                   const HardwareConfig &config,
                                   bool replayIterations);

static void analyzeParsedRegion(mlir::Region &region, int64_t loopMultiplier,
                                AnalysisState &state,
                                HIVMAnalysisReport &report,
                                const HardwareConfig &config,
                                bool replayIterations) {
  for (mlir::Block &block : region) {
    for (mlir::Operation &op : block)
      analyzeParsedOperation(&op, loopMultiplier, state, report, config,
                             replayIterations);
  }
}

static void seedLoopCarriedState(mlir::scf::ForOp forOp,
                                 const AnalysisState &parentState,
                                 AnalysisState &loopState) {
  mlir::Block &body = forOp.getRegion().front();
  mlir::Block::BlockArgListType bodyArgs = body.getArguments();
  unsigned iterArgOffset = 1;
  for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
    if (iterArgOffset + idx >= bodyArgs.size())
      break;
    mlir::BlockArgument regionArg = bodyArgs[iterArgOffset + idx];
    if (auto rootIt = parentState.bufferRoots.find(initArg);
        rootIt != parentState.bufferRoots.end()) {
      loopState.bufferRoots[regionArg] = rootIt->second;
    }
    if (auto producerIt = parentState.valueProducers.find(initArg);
        producerIt != parentState.valueProducers.end()) {
      loopState.valueProducers[regionArg] = producerIt->second;
    }
    if (auto constantIt = parentState.constants.find(initArg);
        constantIt != parentState.constants.end()) {
      loopState.constants[regionArg] = constantIt->second;
    }
    if (auto boundIt = parentState.boundValues.find(initArg);
        boundIt != parentState.boundValues.end()) {
      loopState.boundValues[regionArg] = boundIt->second;
    }
  }
}

static void propagateLoopResults(mlir::scf::ForOp forOp,
                                 const AnalysisState &loopState,
                                 AnalysisState &parentState) {
  auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(
      forOp.getRegion().front().getTerminator());
  if (!yieldOp)
    return;

  for (auto [idx, yielded] : llvm::enumerate(yieldOp.getOperands())) {
    if (idx >= forOp.getNumResults())
      break;
    mlir::Value result = forOp.getResult(idx);
    if (auto producerIt = loopState.valueProducers.find(yielded);
        producerIt != loopState.valueProducers.end()) {
      parentState.valueProducers[result] = producerIt->second;
    }
    if (auto rootIt = loopState.bufferRoots.find(yielded);
        rootIt != loopState.bufferRoots.end()) {
      parentState.bufferRoots[result] = rootIt->second;
    }
    if (auto constantIt = loopState.constants.find(yielded);
        constantIt != loopState.constants.end()) {
      parentState.constants[result] = constantIt->second;
    }
    if (auto boundIt = loopState.boundValues.find(yielded);
        boundIt != loopState.boundValues.end()) {
      parentState.boundValues[result] = boundIt->second;
    }
  }

  parentState.bufferSlots = loopState.bufferSlots;
  parentState.bufferVersions = loopState.bufferVersions;
  parentState.eventProducers = loopState.eventProducers;
  parentState.eventGenerations = loopState.eventGenerations;
  parentState.latestPipeProducer = loopState.latestPipeProducer;
}

static void advanceLoopCarriedState(mlir::scf::ForOp forOp,
                                    AnalysisState &loopState) {
  auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(
      forOp.getRegion().front().getTerminator());
  if (!yieldOp)
    return;

  mlir::Block &body = forOp.getRegion().front();
  mlir::Block::BlockArgListType bodyArgs = body.getArguments();
  unsigned iterArgOffset = 1;
  for (auto [idx, yielded] : llvm::enumerate(yieldOp.getOperands())) {
    if (iterArgOffset + idx >= bodyArgs.size())
      break;
    mlir::BlockArgument regionArg = bodyArgs[iterArgOffset + idx];

    if (auto producerIt = loopState.valueProducers.find(yielded);
        producerIt != loopState.valueProducers.end()) {
      loopState.valueProducers[regionArg] = producerIt->second;
    } else {
      loopState.valueProducers.erase(regionArg);
    }

    if (auto rootIt = loopState.bufferRoots.find(yielded);
        rootIt != loopState.bufferRoots.end()) {
      loopState.bufferRoots[regionArg] = rootIt->second;
    } else {
      loopState.bufferRoots.erase(regionArg);
    }

    if (auto constantIt = loopState.constants.find(yielded);
        constantIt != loopState.constants.end()) {
      loopState.constants[regionArg] = constantIt->second;
    } else {
      loopState.constants.erase(regionArg);
    }

    if (auto boundIt = loopState.boundValues.find(yielded);
        boundIt != loopState.boundValues.end()) {
      loopState.boundValues[regionArg] = boundIt->second;
    } else {
      loopState.boundValues.erase(regionArg);
    }
  }
}

static void analyzeParsedOperation(mlir::Operation *op, int64_t loopMultiplier,
                                   AnalysisState &state,
                                   HIVMAnalysisReport &report,
                                   const HardwareConfig &config,
                                   bool replayIterations) {
  if (captureConstant(op, state))
    return;
  captureDerivedScalarValue(op, state);

  captureBufferMetadata(op, state);

  if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
    // Each function (AIC / AIV) runs on its own core in parallel.  Use a
    // fresh per-function state so that pipe-ordering and event tracking do
    // not bleed across functions.  Only preserve arg-bindings.
    AnalysisState funcState;
    funcState.argBindings = state.argBindings;
    std::optional<unsigned> minBoundArgIndex;
    for (const auto &entry : funcState.argBindings) {
      llvm::StringRef name(entry.first);
      if (!name.consume_front("arg"))
        continue;
      unsigned idxValue = 0;
      if (name.getAsInteger(10, idxValue))
        continue;
      if (!minBoundArgIndex || idxValue < *minBoundArgIndex)
        minBoundArgIndex = idxValue;
    }

    std::optional<unsigned> firstScalarArgIndex;
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (funcOp.getArgAttr(idx, "hacc.arg_type"))
        continue;
      if (llvm::isa<mlir::MemRefType>(arg.getType()))
        continue;
      firstScalarArgIndex = idx;
      break;
    }

    auto isBindableIntegerArg = [&](unsigned idx) {
      if (idx >= funcOp.getNumArguments() ||
          funcOp.getArgAttr(idx, "hacc.arg_type"))
        return false;
      mlir::Type type = funcOp.getArgument(idx).getType();
      return llvm::isa<mlir::IntegerType, mlir::IndexType>(type);
    };

    llvm::SmallVector<unsigned, 8> userArgToActual;
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      (void)arg;
      if (!funcOp.getArgAttr(idx, "hacc.arg_type"))
        userArgToActual.push_back(idx);
    }

    int actualIndexScore = 0;
    int userIndexScore = 0;
    for (const auto &entry : funcState.argBindings) {
      llvm::StringRef name(entry.first);
      if (!name.consume_front("arg"))
        continue;
      unsigned bindingIndex = 0;
      if (name.getAsInteger(10, bindingIndex))
        continue;
      actualIndexScore += isBindableIntegerArg(bindingIndex) ? 1 : -1;
      userIndexScore +=
          bindingIndex < userArgToActual.size() &&
                  isBindableIntegerArg(userArgToActual[bindingIndex])
              ? 1
              : -1;
    }

    bool preferActualArgNames = actualIndexScore > userIndexScore;
    if (actualIndexScore == userIndexScore) {
      preferActualArgNames =
          minBoundArgIndex && firstScalarArgIndex &&
          *minBoundArgIndex >= *firstScalarArgIndex;
    }

    unsigned userArgIndex = 0;
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (preferActualArgNames) {
        auto actualBindIt =
            funcState.argBindings.find("arg" + std::to_string(idx));
        if (actualBindIt != funcState.argBindings.end())
          funcState.boundValues[arg] = actualBindIt->second;
      }

      if (funcOp.getArgAttr(idx, "hacc.arg_type"))
        continue;
      auto userBindIt =
          funcState.argBindings.find("arg" + std::to_string(userArgIndex++));
      if (!preferActualArgNames && userBindIt != funcState.argBindings.end())
        funcState.boundValues[arg] = userBindIt->second;
    }
    analyzeParsedRegion(funcOp.getBody(), loopMultiplier, funcState, report,
                        config, replayIterations);
    return;
  }

  if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
    int64_t tripCount = 1;
    bool hasConcreteTripCount = parseForTripCount(forOp, state, tripCount);
    int64_t lowerBound = 0;
    int64_t upperBound = 0;
    int64_t step = 1;
    bool hasLower = resolveMLIRValue(forOp.getLowerBound(), state, lowerBound);
    bool hasUpper = resolveMLIRValue(forOp.getUpperBound(), state, upperBound);
    bool hasStep = resolveMLIRValue(forOp.getStep(), state, step);
    bool hasConcreteInductionValue =
        hasLower && hasStep;
    int64_t nestedMultiplier =
        loopMultiplier * std::max<int64_t>(tripCount, 1);
    report.loopCount++;
    if (hasConcreteTripCount)
      report.resolvedLoopCount++;
    else
      report.unresolvedLoopCount++;
    report.maxLoopTripCount =
        std::max<int64_t>(report.maxLoopTripCount, tripCount);
    report.maxLoopMultiplier =
        std::max<int64_t>(report.maxLoopMultiplier, nestedMultiplier);

    // Diagnostic-only structural upper bound for unresolved loops (e.g.
    // program-id-dependent bounds clamped by a resolvable constant like a
    // sub-chunk size). Never feeds nestedMultiplier/loopMultiplier above.
    int64_t upperBoundTripCountEstimate = -1;
    if (!hasConcreteTripCount)
      estimateForTripCountUpperBound(forOp, state, upperBoundTripCountEstimate);

    // Source line range spanning the loop body, so downstream (Python)
    // consumers can attribute individual ops to this loop by line number
    // without re-parsing the IR.
    int bodyFirstLine = 0;
    int bodyLastLine = 0;
    forOp.getOperation()->walk([&](mlir::Operation *innerOp) {
      if (innerOp == forOp.getOperation())
        return;
      int line = getLineNumberFromLocation(innerOp->getLoc());
      if (line <= 0)
        return;
      if (bodyFirstLine == 0 || line < bodyFirstLine)
        bodyFirstLine = line;
      if (line > bodyLastLine)
        bodyLastLine = line;
    });

    HIVMLoopDiagnostic diag;
    diag.lineNumber = getLineNumberFromLocation(forOp.getLoc());
    diag.lowerBound = hasLower ? lowerBound : 0;
    diag.upperBound = hasUpper ? upperBound : 0;
    diag.step = hasStep ? step : 0;
    diag.tripCount = tripCount;
    diag.multiplier = nestedMultiplier;
    diag.resolved = hasConcreteTripCount;
    diag.upperBoundTripCountEstimate = upperBoundTripCountEstimate;
    diag.bodyFirstLine = bodyFirstLine;
    diag.bodyLastLine = bodyLastLine;
    report.loopDiagnostics.push_back(diag);
    AnalysisState loopState = state;
    seedLoopCarriedState(forOp, state, loopState);
    if (replayIterations && hasConcreteTripCount && tripCount > 1) {
      for (int64_t iter = 0; iter < tripCount; ++iter) {
        if (hasConcreteInductionValue)
          loopState.boundValues[forOp.getInductionVar()] =
              lowerBound + iter * step;
        analyzeParsedRegion(op->getRegion(0), loopMultiplier, loopState, report,
                            config, replayIterations);
        if (iter + 1 < tripCount)
          advanceLoopCarriedState(forOp, loopState);
      }
    } else {
      if (hasConcreteInductionValue)
        loopState.boundValues[forOp.getInductionVar()] = lowerBound;
      int64_t bodyMultiplier =
          (replayIterations && hasConcreteTripCount) ? loopMultiplier
                                                     : nestedMultiplier;
      analyzeParsedRegion(op->getRegion(0), bodyMultiplier, loopState, report,
                          config, replayIterations);
    }
    propagateLoopResults(forOp, loopState, state);
    return;
  }

  if (auto callOp = llvm::dyn_cast<mlir::func::CallOp>(op)) {
    if (!op->hasAttr("hivm.vector_function"))
      return;

    report.outlinedCallCount++;
    ParsedOp parsed;
    parsed.op.opName = "vcall";
    parsed.op.pipe = HIVMPipe::Vector;
    parsed.op.loopMultiplier = loopMultiplier;
    parsed.op.lineNumber = getLineNumberFromLocation(op->getLoc());
    parsed.op.text = renderOperation(op);
    parsed.op.costSource = "outlined_call_lower_bound";
    parsed.op.costSubpipe = "vector";
    parsed.mlirResults.assign(op->result_begin(), op->result_end());

    if (auto parentFunc = op->getParentOfType<mlir::func::FuncOp>()) {
      llvm::StringRef funcName = parentFunc.getName();
      parsed.op.coreType =
          funcName.contains("aic") || funcName.contains("AIC") ? "CUBE"
                                                               : "VECTOR";
    }

    for (mlir::Value operand : callOp.getOperands()) {
      parsed.op.bytes = std::max(
          parsed.op.bytes, inferValueBytesWithBindings(operand, state));
      parsed.op.elements = std::max(
          parsed.op.elements, inferValueElementsWithBindings(operand, state));
      auto producer = state.valueProducers.find(operand);
      if (producer != state.valueProducers.end())
        parsed.op.dependsOn.push_back(producer->second);
    }
    if (parsed.op.elements > 0)
      report.summarizedOutlinedCallCount++;
    parsed.op.elemType = "f32";
    parsed.op.duration = estimateDuration(parsed, config);
    attachBufferAccessMetadata(op, parsed, state);
    ingestParsedOp(parsed, state, report, config);
    return;
  }

  if (startsWithHivmOp(op)) {
    std::string opText = renderOperation(op);
    ParsedOp parsed;
#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
    if (!populateTypedHivmOp(op, parsed))
      return;
    parsed.op.lineNumber = getLineNumberFromLocation(op->getLoc());
#else
    if (!populateGenericHivmOp(op, parsed))
      return;
    parsed.op.lineNumber = getLineNumberFromLocation(op->getLoc());
#endif
    parsed.op.loopMultiplier = loopMultiplier;
    parsed.op.text = opText;
    if (parsed.op.opName.empty())
      parsed.op.opName = getLeafOpName(op).str();
    // Extract CCE repeat/mask from MLIR attributes or op text (Gap 4).
    extractRepeatMask(op, parsed);
    parsed.mlirResults.assign(op->result_begin(), op->result_end());

    for (mlir::Value operand : op->getOperands()) {
      auto it = state.valueProducers.find(operand);
      if (it != state.valueProducers.end())
        parsed.op.dependsOn.push_back(it->second);
    }

    if (op->getNumResults() > 0) {
      parsed.op.bytes = inferValueBytesWithBindings(op->getResult(0), state);
      parsed.op.elements =
          inferValueElementsWithBindings(op->getResult(0), state);
    }
    if (parsed.op.bytes == 0 || parsed.op.elements == 0) {
      for (mlir::Value operand : op->getOperands()) {
        parsed.op.bytes =
            std::max(parsed.op.bytes, inferValueBytesWithBindings(operand, state));
        parsed.op.elements = std::max(
            parsed.op.elements, inferValueElementsWithBindings(operand, state));
      }
    }
    if (parsed.op.bytes == 0)
      parsed.op.bytes = parseMemRefBytes(parsed.op.text);
    if (parsed.op.elements == 0)
      parsed.op.elements = parseMemRefElementCount(parsed.op.text);

    // Gap-2 packet size: for MTE transfers, the contiguous run the hardware
    // moves per shot.  Strided/gather transfers keep a small packet even when
    // the total volume is large — that gap is what the coalescing model reads.
    if (parsed.op.bytes > 0 &&
        (parsed.op.opName == "load" || parsed.op.opName == "store")) {
      int64_t packet = parsed.op.bytes;  // default: fully contiguous
      for (mlir::Value operand : op->getOperands()) {
        if (!llvm::isa<mlir::MemRefType>(operand.getType()))
          continue;
        int64_t opTotal = inferValueBytesWithBindings(operand, state);
        if (opTotal <= 0)
          opTotal = parsed.op.bytes;
        packet = std::min(packet,
                          inferTransferPacketBytes(operand, state, opTotal));
      }
      parsed.op.packetBytes = packet;
    }
    // Gap 4: derive the CCE repeat count (SIMD iteration count) analytically
    // from the op's element count and per-element width.  Per-op repeat/mask
    // are NOT present in the hivm.hir IR — they only materialize in later CCE
    // codegen — so we compute the canonical iteration count instead: a
    // 256-byte (2048-bit) vector register processes (2048/bitsPerElem)
    // elements per iteration (matches AscendModelOps.cpp's vectorWidth math).
    // Only fill when extractRepeatMask found no explicit attribute/text
    // (it leaves repeat==1), so a real CCE repeat field would still win.
    if (parsed.op.repeat == 1 && parsed.op.elements > 0 && parsed.op.bytes > 0) {
      int64_t bitsPerElem = (parsed.op.bytes * 8) / parsed.op.elements;
      if (bitsPerElem > 0) {
        int64_t laneCount = 2048 / bitsPerElem; // elements per 256B register
        if (laneCount < 1)
          laneCount = 1;
        parsed.op.repeat = (parsed.op.elements + laneCount - 1) / laneCount;
      }
    }
#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
    // Extract src/dst memory spaces from MLIR operand/result types.
    {
      llvm::StringRef name = parsed.op.opName;
      if (name == "load" && op->getNumOperands() >= 1) {
        parsed.op.srcSpace = getCanonicalTypeAddressSpace(op->getOperand(0).getType());
        if (op->getNumOperands() >= 2)
          parsed.op.dstSpace = getCanonicalTypeAddressSpace(op->getOperand(1).getType());
        else if (op->getNumResults() > 0)
          parsed.op.dstSpace = getCanonicalTypeAddressSpace(op->getResult(0).getType());
      } else if (name == "store" && op->getNumOperands() >= 2) {
        parsed.op.srcSpace = getCanonicalTypeAddressSpace(op->getOperand(0).getType());
        parsed.op.dstSpace = getCanonicalTypeAddressSpace(op->getOperand(1).getType());
      } else if (name == "copy" && op->getNumOperands() >= 2) {
        parsed.op.srcSpace = getCanonicalTypeAddressSpace(op->getOperand(0).getType());
        parsed.op.dstSpace = getCanonicalTypeAddressSpace(op->getOperand(1).getType());
      } else if (name == "fixpipe" && op->getNumOperands() >= 1) {
        parsed.op.srcSpace = getCanonicalTypeAddressSpace(op->getOperand(0).getType());
        if (op->getNumOperands() >= 2)
          parsed.op.dstSpace = getCanonicalTypeAddressSpace(op->getOperand(1).getType());
      } else if ((name == "nd2nz" || name == "nz2nd") && op->getNumOperands() >= 2) {
        parsed.op.srcSpace = getCanonicalTypeAddressSpace(op->getOperand(0).getType());
        parsed.op.dstSpace = getCanonicalTypeAddressSpace(op->getOperand(1).getType());
      } else if (isCubeOpName(name)) {
        if (op->getNumOperands() >= 1)
          parsed.op.srcSpace = getCanonicalTypeAddressSpace(op->getOperand(0).getType());
        if (op->getNumResults() > 0)
          parsed.op.dstSpace = getCanonicalTypeAddressSpace(op->getResult(0).getType());
      }
    }
#endif
    // Extract element type from MLIR result or operand type.
    {
      mlir::Type ty;
      if (op->getNumResults() > 0)
        ty = op->getResult(0).getType();
      else if (op->getNumOperands() > 0)
        ty = op->getOperand(0).getType();
      parsed.op.elemType = getElementTypeName(ty);
    }
    attachSyncMetadata(parsed);
    attachBufferAccessMetadata(op, parsed, state);
    parsed.op.duration = estimateDuration(parsed, config);
    if ((parsed.op.opName == "load" || parsed.op.opName == "store" ||
         parsed.op.opName == "copy" || parsed.op.opName == "fixpipe") &&
        parsed.op.bytes <= 0)
      report.zeroByteTransferCount++;
    ingestParsedOp(parsed, state, report, config);
  } else if (!llvm::isa<mlir::scf::YieldOp>(op) &&
             !llvm::isa<mlir::func::ReturnOp>(op) &&
             op->getNumResults() > 0) {
    // Non-hivm ops with results are scalar operations (arith, memref, etc.)
    // that execute on PIPE_S.  They produce SSA values consumed by hivm.hir
    // ops, creating cross-pipe dependencies via def-use chains.
    ParsedOp parsed;
    parsed.op.opName = getLeafOpName(op).str();
    parsed.op.pipe = HIVMPipe::Scalar;
    parsed.op.loopMultiplier = loopMultiplier;
    parsed.op.lineNumber = getLineNumberFromLocation(op->getLoc());
    parsed.op.text = renderOperation(op);
    parsed.op.duration = 1;  // scalar ops take 1 cycle
    parsed.mlirResults.assign(op->result_begin(), op->result_end());
    report.zeroWorkScalarOpCount++;

    // Determine core type from function context
    if (auto parentFunc = op->getParentOfType<mlir::func::FuncOp>()) {
      llvm::StringRef funcName = parentFunc.getName();
      if (funcName.contains("aic") || funcName.contains("AIC") ||
          funcName.contains("cube"))
        parsed.op.coreType = "CUBE";
      else if (funcName.contains("aiv") || funcName.contains("AIV") ||
               funcName.contains("vector") || funcName.contains("mix"))
        parsed.op.coreType = "VECTOR";
    }

    // SSA dependencies: if this op uses a value produced by a previous op
    for (mlir::Value operand : op->getOperands()) {
      auto it = state.valueProducers.find(operand);
      if (it != state.valueProducers.end())
        parsed.op.dependsOn.push_back(it->second);
    }

    ingestParsedOp(parsed, state, report, config);
  }

  for (mlir::Region &region : op->getRegions())
    analyzeParsedRegion(region, loopMultiplier, state, report, config,
                        replayIterations);
}

static std::string textLeafOpName(llvm::StringRef record) {
  size_t pos = record.find("hivm.hir.");
  if (pos == llvm::StringRef::npos)
    return "";
  pos += strlen("hivm.hir.");
  size_t end = pos;
  while (end < record.size()) {
    char c = record[end];
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
      break;
    ++end;
  }
  return record.slice(pos, end).str();
}

static std::string textScalarLikeOpName(llvm::StringRef record) {
  llvm::StringRef trimmed = record.trim();
  if (!trimmed.starts_with("%"))
    return "";
  size_t eq = trimmed.find('=');
  if (eq == llvm::StringRef::npos)
    return "";

  auto leafAfterPrefix = [&](llvm::StringRef prefix) -> std::string {
    size_t pos = trimmed.find(prefix, eq + 1);
    if (pos == llvm::StringRef::npos)
      return "";
    pos += prefix.size();
    size_t end = pos;
    while (end < trimmed.size()) {
      char c = trimmed[end];
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
        break;
      ++end;
    }
    return trimmed.slice(pos, end).str();
  };

  std::string arithName = leafAfterPrefix("arith.");
  if (!arithName.empty() && arithName != "constant")
    return arithName;

  std::string affineName = leafAfterPrefix("affine.");
  if (affineName == "apply" || affineName == "min" || affineName == "max")
    return affineName;

  std::string memrefName = leafAfterPrefix("memref.");
  if (memrefName == "reinterpret_cast" || memrefName == "cast" ||
      memrefName == "subview" || memrefName == "collapse_shape" ||
      memrefName == "expand_shape")
    return memrefName;

  return "";
}

static std::string textCurrentCoreType(llvm::StringRef funcName) {
  if (funcName.contains("aic") || funcName.contains("AIC") ||
      funcName.contains("cube"))
    return "CUBE";
  if (funcName.contains("aiv") || funcName.contains("AIV") ||
      funcName.contains("vector") || funcName.contains("mix"))
    return "VECTOR";
  return "";
}

static std::string textFuncName(llvm::StringRef line) {
  size_t at = line.find('@');
  if (at == llvm::StringRef::npos)
    return "";
  size_t end = at + 1;
  while (end < line.size()) {
    char c = line[end];
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '.')
      break;
    ++end;
  }
  return line.slice(at + 1, end).str();
}

static llvm::SmallVector<llvm::StringRef, 4>
textStaticOperands(llvm::StringRef record) {
  llvm::SmallVector<llvm::StringRef, 4> tokens;
  size_t open = record.find('[');
  if (open == llvm::StringRef::npos)
    return tokens;
  size_t close = record.find(']', open + 1);
  if (close == llvm::StringRef::npos)
    return tokens;
  llvm::StringRef body = record.slice(open + 1, close);
  body.split(tokens, ',', -1, false);
  for (llvm::StringRef &token : tokens)
    token = token.trim();
  return tokens;
}

static std::string textDynamicEventToken(llvm::StringRef record) {
  size_t bracketEnd = record.find(']');
  if (bracketEnd == llvm::StringRef::npos)
    return "";
  size_t pos = record.find('%', bracketEnd);
  if (pos == llvm::StringRef::npos)
    return "";
  size_t end = pos + 1;
  while (end < record.size()) {
    char c = record[end];
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
      break;
    ++end;
  }
  return ("dyn_" + record.slice(pos + 1, end).str());
}

static std::string textEventToken(llvm::StringRef record,
                                  llvm::ArrayRef<llvm::StringRef> operands,
                                  size_t staticIndex) {
  if (staticIndex < operands.size()) {
    llvm::StringRef token = operands[staticIndex];
    if (token.contains("EVENT_ID"))
      return parseEventToken(token);
    if (token.contains("FLAG_ID") || token.contains("BLOCK_ID"))
      return token.trim("<>").str();
  }
  return textDynamicEventToken(record);
}

static int64_t textCountChar(llvm::StringRef line, char needle) {
  int64_t n = 0;
  for (char c : line)
    if (c == needle)
      ++n;
  return n;
}

static bool parseTextConstant(llvm::StringRef line, std::string &name,
                              int64_t &value) {
  llvm::StringRef trimmed = line.trim();
  if (!trimmed.starts_with("%") || !trimmed.contains("arith.constant"))
    return false;
  size_t eq = trimmed.find('=');
  if (eq == llvm::StringRef::npos)
    return false;
  name = trimmed.slice(0, eq).trim().str();
  size_t pos = trimmed.find("arith.constant");
  if (pos == llvm::StringRef::npos)
    return false;
  llvm::StringRef tail = trimmed.drop_front(pos + strlen("arith.constant")).trim();
  int64_t parsed = 0;
  if (tail.consumeInteger(10, parsed))
    return false;
  value = parsed;
  return true;
}

static bool resolveTextLoopBound(llvm::StringRef token,
                                 const std::map<std::string, int64_t> &constants,
                                 int64_t &value) {
  token = token.trim();
  if (token.empty())
    return false;
  if (token.starts_with("%")) {
    auto it = constants.find(token.str());
    if (it == constants.end())
      return false;
    value = it->second;
    return true;
  }
  return !token.consumeInteger(10, value);
}

static bool parseTextScfForTripCount(
    llvm::StringRef line, const std::map<std::string, int64_t> &constants,
    int64_t &lower, int64_t &upper, int64_t &step, int64_t &tripCount) {
  llvm::StringRef trimmed = line.trim();
  if (!trimmed.starts_with("scf.for "))
    return false;
  size_t eq = trimmed.find('=');
  size_t toPos = trimmed.find(" to ", eq == llvm::StringRef::npos ? 0 : eq);
  size_t stepPos = trimmed.find(" step ", toPos == llvm::StringRef::npos ? 0 : toPos);
  if (eq == llvm::StringRef::npos || toPos == llvm::StringRef::npos ||
      stepPos == llvm::StringRef::npos)
    return false;
  llvm::StringRef lowerTok = trimmed.slice(eq + 1, toPos).trim();
  llvm::StringRef upperTok = trimmed.slice(toPos + 4, stepPos).trim();
  llvm::StringRef stepTail = trimmed.drop_front(stepPos + 6).trim();
  size_t stepEnd = 0;
  while (stepEnd < stepTail.size()) {
    char c = stepTail[stepEnd];
    if (std::isspace(static_cast<unsigned char>(c)) || c == '{')
      break;
    ++stepEnd;
  }
  llvm::StringRef stepTok = stepTail.take_front(stepEnd).trim();
  if (!resolveTextLoopBound(lowerTok, constants, lower) ||
      !resolveTextLoopBound(upperTok, constants, upper) ||
      !resolveTextLoopBound(stepTok, constants, step) || step <= 0 ||
      upper < lower) {
    tripCount = 1;
    return false;
  }
  tripCount = std::max<int64_t>(1, ceilDiv(upper - lower, step));
  return true;
}

static ParsedOp parseSemanticHivmRecord(llvm::StringRef record,
                                        llvm::StringRef currentFunc,
                                        int lineNumber,
                                        const HardwareConfig &config) {
  ParsedOp parsed;
  parsed.op.opName = textLeafOpName(record);
  parsed.op.text = record.str();
  parsed.op.lineNumber = lineNumber;
  parsed.op.loopMultiplier = 1;
  parsed.op.coreType = textCurrentCoreType(currentFunc);
  parsed.op.bytes = parseMemRefBytes(record);
  parsed.op.elements = parseMemRefElementCount(record);
  parsed.op.packetBytes = parsed.op.bytes;

  auto spaces = parseLoadStoreSpaces(record);
  parsed.op.srcSpace = spaces.first;
  parsed.op.dstSpace = spaces.second;
  {
    size_t memrefPos = record.find("memref<");
    if (memrefPos != llvm::StringRef::npos) {
      size_t addrPos = record.find(", #hivm.address_space<", memrefPos);
      if (addrPos != llvm::StringRef::npos) {
        llvm::StringRef shapeAndType = record.slice(memrefPos + 7, addrPos);
        llvm::SmallVector<llvm::StringRef, 8> parts;
        shapeAndType.split(parts, 'x', -1, false);
        if (!parts.empty())
          parsed.op.elemType = trim(parts.back()).str();
      }
    }
  }

  llvm::SmallVector<llvm::StringRef, 4> operands =
      textStaticOperands(record);
  llvm::StringRef name = parsed.op.opName;
  if (name == "pipe_barrier") {
    parsed.op.isSyncOp = true;
    parsed.op.isBarrier = true;
    HIVMPipe rawPipe = operands.empty() ? HIVMPipe::All : parsePipeToken(operands[0]);
    parsed.op.pipe =
        disambiguateMTE2Pipe(rawPipe, HIVMPipe::Unknown, parsed.op.coreType);
    parsed.barrierPipes.push_back(parsed.op.pipe);
  } else if (name == "set_flag" || name == "wait_flag") {
    parsed.op.isSyncOp = true;
    HIVMPipe setPipe = operands.size() > 0 ? parsePipeToken(operands[0])
                                           : HIVMPipe::Unknown;
    HIVMPipe waitPipe = operands.size() > 1 ? parsePipeToken(operands[1])
                                            : HIVMPipe::Unknown;
    parsed.senderPipe =
        disambiguateMTE2Pipe(setPipe, waitPipe, parsed.op.coreType);
    parsed.receiverPipe =
        disambiguateMTE2Pipe(waitPipe, parsed.senderPipe, parsed.op.coreType);
    parsed.eventId = textEventToken(record, operands, 2);
    parsed.op.pipe = name == "set_flag" ? parsed.senderPipe : parsed.receiverPipe;
  } else if (name == "sync_block_set" || name == "sync_block_wait") {
    parsed.op.isSyncOp = true;
    parsed.op.isBarrier = name == "sync_block_wait";
    parsed.senderPipe =
        operands.size() > 1 ? parsePipeToken(operands[1]) : HIVMPipe::Unknown;
    parsed.receiverPipe =
        operands.size() > 2 ? parsePipeToken(operands[2]) : HIVMPipe::Unknown;
    parsed.eventId = textEventToken(record, operands, 3);
    parsed.op.pipe =
        name == "sync_block_wait" ? HIVMPipe::All : parsed.senderPipe;
    if (parsed.op.isBarrier)
      parsed.barrierPipes.push_back(HIVMPipe::All);
  } else if (name == "sync_block") {
    parsed.op.isSyncOp = true;
    parsed.op.isBarrier = true;
    parsed.op.pipe = HIVMPipe::All;
    parsed.barrierPipes.push_back(HIVMPipe::All);
  } else if (name == "load") {
    parsed.op.pipe = selectMTE2PipeForSpaces(parsed.op.srcSpace,
                                             parsed.op.dstSpace,
                                             parsed.op.coreType);
  } else if (name == "store") {
    parsed.op.pipe = HIVMPipe::MTE3;
  } else if (name == "fixpipe") {
    parsed.op.pipe = HIVMPipe::FixPipe;
  } else if (name == "copy") {
    if (parsed.op.dstSpace == "gm")
      parsed.op.pipe = HIVMPipe::MTE3;
    else if (parsed.op.dstSpace == "l1")
      parsed.op.pipe = HIVMPipe::CubeMTE2;
    else
      parsed.op.pipe = HIVMPipe::Vector;
  } else if (name == "nd2nz") {
    parsed.op.pipe = HIVMPipe::CubeMTE2;
  } else if (name == "nz2nd") {
    parsed.op.pipe = HIVMPipe::MTE3;
  } else if (name == "pointer_cast" || name == "convert_layout") {
    parsed.op.pipe = HIVMPipe::Unknown;
  } else if (isCubeOpName(name)) {
    parsed.op.pipe = HIVMPipe::Cube;
  } else {
    parsed.op.pipe = HIVMPipe::Vector;
  }

  attachSyncMetadata(parsed);
  if (parsed.op.repeat == 1 && parsed.op.elements > 0 && parsed.op.bytes > 0) {
    int64_t bitsPerElem = (parsed.op.bytes * 8) / parsed.op.elements;
    if (bitsPerElem > 0) {
      int64_t laneCount = std::max<int64_t>(1, 2048 / bitsPerElem);
      parsed.op.repeat = (parsed.op.elements + laneCount - 1) / laneCount;
    }
  }
  parsed.op.duration = estimateDuration(parsed, config);
  return parsed;
}

static ParsedOp parseSemanticScalarRecord(llvm::StringRef record,
                                          llvm::StringRef currentFunc,
                                          int lineNumber,
                                          const HardwareConfig &config) {
  ParsedOp parsed;
  parsed.op.opName = textScalarLikeOpName(record);
  parsed.op.text = record.str();
  parsed.op.lineNumber = lineNumber;
  parsed.op.loopMultiplier = 1;
  parsed.op.coreType = textCurrentCoreType(currentFunc);
  parsed.op.pipe = HIVMPipe::Scalar;
  parsed.op.bytes = parseMemRefBytes(record);
  parsed.op.elements = parseMemRefElementCount(record);
  parsed.op.packetBytes = parsed.op.bytes;
  parsed.op.duration = estimateDuration(parsed, config);
  return parsed;
}

static void finalizeSemanticReport(HIVMAnalysisReport &report,
                                   const HardwareConfig &config,
                                   HIVMSchedulerMode schedulerMode) {
  if (schedulerMode == HIVMSchedulerMode::DES)
    finalizeDiscreteEventReport(report, config);
  else
    finalizeScheduledReport(report, config);
}

static bool isReportableScalarFallback(const HIVMOp &op) {
  return op.pipe == HIVMPipe::Scalar &&
         op.opName != "set_flag" && op.opName != "wait_flag" &&
         op.opName != "sync_block_set" && op.opName != "sync_block_wait" &&
         op.opName != "sync_block" && op.opName != "pipe_barrier" &&
         op.opName != "get_block_idx" && op.opName != "get_block_num" &&
         op.opName != "get_sub_block_idx" &&
         op.opName != "get_sub_block_num" &&
         op.opName != "set_mask_norm" && op.opName != "pointer_cast" &&
         op.opName != "convert_layout";
}

static void addCostStat(std::map<std::string, HIVMCostStat> &stats,
                        llvm::StringRef key, const HIVMOp &op) {
  if (key.empty())
    key = "unclassified";
  HIVMCostStat &stat = stats[key.str()];
  stat.ops++;
  stat.cycles += op.duration;
  stat.weightedCycles += op.duration * op.loopMultiplier;
}

static void recordCalibrationSummary(HIVMAnalysisReport &report,
                                     const HIVMOp &op) {
  int64_t weighted = op.duration * op.loopMultiplier;
  if (op.calibratedCost) {
    report.calibratedOpCount++;
    report.calibratedCycles += op.duration;
    report.calibratedWeightedCycles += weighted;
  } else {
    report.heuristicOpCount++;
    report.heuristicCycles += op.duration;
    report.heuristicWeightedCycles += weighted;
  }
  addCostStat(report.costSourceStats, op.costSource, op);
  addCostStat(report.costSubpipeStats, op.costSubpipe, op);
  if (op.costSubpipe.empty()) {
    std::string key = op.opName + "|" + HIVMAnalyzer::stringifyPipe(op.pipe).str() +
                      "|" + (op.costSource.empty() ? "unknown" : op.costSource);
    addCostStat(report.unclassifiedCostStats, key, op);
  }
}

static void resetScheduleSummary(HIVMAnalysisReport &report) {
  report.oneIterationCycles = 0;
  report.weightedCycles = 0;
  report.bodyCycles = 0;
  report.predictedTotalCycles = 0;
  report.totalBusyCycles = 0;
  report.syncCycles = 0;
  report.syncIssueCycles = 0;
  report.syncEventWaitCycles = 0;
  report.criticalPathCycles = 0;
  report.criticalPathIssueCycles = 0;
  report.criticalPathEventWaitCycles = 0;
  report.barrierCycles = 0;
  report.opCount = 0;
  report.syncOpCount = 0;
  report.barrierCount = 0;
  report.unknownOpCount = 0;
  report.calibratedOpCount = 0;
  report.heuristicOpCount = 0;
  report.calibratedCycles = 0;
  report.heuristicCycles = 0;
  report.calibratedWeightedCycles = 0;
  report.heuristicWeightedCycles = 0;
  report.scheduleTruncated = false;
  report.pipeBusyCycles.clear();
  report.weightedPipeCycles.clear();
  report.costSourceStats.clear();
  report.costSubpipeStats.clear();
  report.unclassifiedCostStats.clear();
  report.criticalPathOps.clear();

  for (HIVMOp &op : report.operations) {
    op.startCycle = 0;
    op.resourceReleaseCycle = 0;
    op.valueReadyCycle = 0;
    op.endCycle = 0;
    op.eventWaitCycles = 0;
  }
}

static bool analyzeSemanticHivmBuffer(llvm::StringRef buffer,
                                      llvm::StringRef path,
                                      HIVMAnalysisReport &report,
                                      const HardwareConfig &config,
                                      llvm::StringRef argBindings,
                                      HIVMSchedulerMode schedulerMode) {
  if (!buffer.contains("hivm.hir."))
    return false;

  report = HIVMAnalysisReport();
  report.sourcePath = path.str();
  report.sourceMode = "direct-hivm-semantic";
  report.schedulerMode = schedulerMode;

  AnalysisState state;
  state.argBindings = parseArgBindings(argBindings);
  std::string currentFunc;
  std::map<std::string, int64_t> textConstants;
  std::vector<TextLoopFrame> textLoopStack;
  int braceDepth = 0;
  llvm::SmallVector<llvm::StringRef, 0> lines;
  buffer.split(lines, '\n');

  for (size_t i = 0; i < lines.size(); ++i) {
    llvm::StringRef line = lines[i];
    llvm::StringRef trimmed = line.trim();
    std::string constantName;
    int64_t constantValue = 0;
    if (parseTextConstant(trimmed, constantName, constantValue))
      textConstants[constantName] = constantValue;

    if (trimmed.starts_with("func.func"))
      currentFunc = textFuncName(trimmed);

    if (trimmed.starts_with("scf.for ")) {
      int64_t lower = 0;
      int64_t upper = 0;
      int64_t step = 0;
      int64_t tripCount = 1;
      bool resolved = parseTextScfForTripCount(trimmed, textConstants, lower,
                                               upper, step, tripCount);
      int64_t parentMultiplier =
          textLoopStack.empty() ? 1 : textLoopStack.back().multiplier;
      int64_t nestedMultiplier =
          parentMultiplier * std::max<int64_t>(tripCount, 1);
      report.loopCount++;
      if (resolved)
        report.resolvedLoopCount++;
      else
        report.unresolvedLoopCount++;
      report.maxLoopTripCount =
          std::max<int64_t>(report.maxLoopTripCount, tripCount);
      report.maxLoopMultiplier =
          std::max<int64_t>(report.maxLoopMultiplier, nestedMultiplier);
      report.loopDiagnostics.push_back(HIVMLoopDiagnostic{
          static_cast<int>(i + 1), lower, upper, step, tripCount,
          nestedMultiplier, resolved});
      int openBraces = static_cast<int>(textCountChar(line, '{'));
      int closeBraces = static_cast<int>(textCountChar(line, '}'));
      int bodyDepth = braceDepth + std::max(1, openBraces);
      textLoopStack.push_back(TextLoopFrame{bodyDepth, nestedMultiplier});
      braceDepth += openBraces - closeBraces;
      while (!textLoopStack.empty() &&
             braceDepth < textLoopStack.back().braceDepth)
        textLoopStack.pop_back();
      continue;
    }

    bool isHivmRecord = line.find("hivm.hir.") != llvm::StringRef::npos;
    bool isScalarRecord = !textScalarLikeOpName(line).empty();
    if (!isHivmRecord && !isScalarRecord) {
      braceDepth += static_cast<int>(textCountChar(line, '{')) -
                    static_cast<int>(textCountChar(line, '}'));
      while (!textLoopStack.empty() &&
             braceDepth < textLoopStack.back().braceDepth)
        textLoopStack.pop_back();
      continue;
    }

    std::string record = line.str();
    int64_t parenBalance = textCountChar(line, '(') - textCountChar(line, ')');
    int64_t bracketBalance =
        textCountChar(line, '[') - textCountChar(line, ']');
    int64_t angleBalance = textCountChar(line, '<') - textCountChar(line, '>');
    int startLine = static_cast<int>(i + 1);
    while ((parenBalance > 0 || bracketBalance > 0 || angleBalance > 0) &&
           i + 1 < lines.size()) {
      ++i;
      record += "\n";
      record += lines[i].str();
      parenBalance += textCountChar(lines[i], '(') - textCountChar(lines[i], ')');
      bracketBalance += textCountChar(lines[i], '[') - textCountChar(lines[i], ']');
      angleBalance += textCountChar(lines[i], '<') - textCountChar(lines[i], '>');
    }

    ParsedOp parsed = isHivmRecord
                          ? parseSemanticHivmRecord(record, currentFunc,
                                                    startLine, config)
                          : parseSemanticScalarRecord(record, currentFunc,
                                                      startLine, config);
    if (parsed.op.opName.empty())
      continue;
    int64_t semanticLoopMultiplier =
        textLoopStack.empty() ? 1 : textLoopStack.back().multiplier;
    int64_t replayCount =
        schedulerMode == HIVMSchedulerMode::DES ? semanticLoopMultiplier : 1;
    parsed.op.loopMultiplier =
        schedulerMode == HIVMSchedulerMode::DES ? 1 : semanticLoopMultiplier;
    for (int64_t replay = 0; replay < std::max<int64_t>(1, replayCount);
         ++replay)
      ingestParsedOp(parsed, state, report, config);
    braceDepth += static_cast<int>(textCountChar(line, '{')) -
                  static_cast<int>(textCountChar(line, '}'));
    while (!textLoopStack.empty() &&
           braceDepth < textLoopStack.back().braceDepth)
      textLoopStack.pop_back();
  }

  if (report.operations.empty())
    return false;
  finalizeSemanticReport(report, config, schedulerMode);
  applyKernelLaunchOverhead(report, config, state.argBindings, argBindings);
  return true;
}

struct ScheduleResourceKey {
  HIVMPipe pipe = HIVMPipe::Unknown;
  std::string coreType;
  bool flowControl = false;

  bool operator<(const ScheduleResourceKey &other) const {
    return std::tie(pipe, coreType, flowControl) <
           std::tie(other.pipe, other.coreType, other.flowControl);
  }
};

static std::string normalizeScheduleCore(llvm::StringRef coreType,
                                         HIVMPipe pipe) {
  if (coreType == "CUBE" || coreType == "AIC")
    return "AIC";
  if (coreType == "VECTOR" || coreType == "AIV")
    return "AIV";
  if (pipe == HIVMPipe::Scalar || pipe == HIVMPipe::All ||
      pipe == HIVMPipe::Unknown)
    return "";
  if (pipeBelongsToCore(pipe, "AIC"))
    return "AIC";
  if (pipeBelongsToCore(pipe, "AIV"))
    return "AIV";
  return "";
}

static bool usesFlowControlResource(const HIVMOp &op) {
  return op.opName == "sync_block_wait";
}

static ScheduleResourceKey getScheduleResource(const HIVMOp &op) {
  return {op.pipe, normalizeScheduleCore(op.coreType, op.pipe),
          usesFlowControlResource(op)};
}

static ScheduleResourceKey getScheduleResource(HIVMPipe pipe,
                                               llvm::StringRef coreType) {
  return {pipe, normalizeScheduleCore(coreType, pipe), false};
}

static bool scheduleResourceBelongsToCore(const ScheduleResourceKey &resource,
                                          llvm::StringRef coreType) {
  std::string normalized = normalizeScheduleCore(coreType, resource.pipe);
  return normalized.empty() || resource.coreType.empty() ||
         resource.coreType == normalized;
}

static void finalizeScheduledReport(HIVMAnalysisReport &report,
                                    const HardwareConfig &config) {
  std::map<ScheduleResourceKey, int64_t> pipeAvailableAt;
  for (HIVMOp &op : report.operations) {
    int64_t earliest = 0;
    for (size_t depId : op.dependsOn) {
      if (depId < report.operations.size())
        earliest = std::max(earliest, report.operations[depId].endCycle);
    }

    if (usesFlowControlResource(op)) {
      ScheduleResourceKey resource = getScheduleResource(op);
      int64_t start = std::max(earliest, pipeAvailableAt[resource]);
      op.startCycle = start;
      op.issueDuration = op.issueDuration ? op.issueDuration : op.duration;
      op.dependencyLatency =
          op.dependencyLatency ? op.dependencyLatency : op.duration;
      op.resourceReleaseCycle = start + op.issueDuration;
      op.valueReadyCycle = start + op.dependencyLatency;
      op.endCycle = op.valueReadyCycle;
      pipeAvailableAt[resource] = op.resourceReleaseCycle;
    } else if (op.isBarrier) {
      int64_t start = earliest;
      if (op.pipe == HIVMPipe::All) {
        for (const auto &entry : pipeAvailableAt) {
          if (scheduleResourceBelongsToCore(entry.first, op.coreType))
            start = std::max(start, entry.second);
        }
        op.startCycle = start;
        op.issueDuration = op.issueDuration ? op.issueDuration : op.duration;
        op.dependencyLatency =
            op.dependencyLatency ? op.dependencyLatency : op.duration;
        op.resourceReleaseCycle = start + op.issueDuration;
        op.valueReadyCycle = start + op.dependencyLatency;
        op.endCycle = op.valueReadyCycle;
        for (auto &entry : pipeAvailableAt) {
          if (scheduleResourceBelongsToCore(entry.first, op.coreType))
            entry.second = op.resourceReleaseCycle;
        }
        for (HIVMPipe pipe : getCoreBarrierPipes(op.coreType))
          pipeAvailableAt[getScheduleResource(pipe, op.coreType)] =
              op.resourceReleaseCycle;
      } else {
        ScheduleResourceKey resource = getScheduleResource(op);
        start = std::max(start, pipeAvailableAt[resource]);
        op.startCycle = start;
        op.issueDuration = op.issueDuration ? op.issueDuration : op.duration;
        op.dependencyLatency =
            op.dependencyLatency ? op.dependencyLatency : op.duration;
        op.resourceReleaseCycle = start + op.issueDuration;
        op.valueReadyCycle = start + op.dependencyLatency;
        op.endCycle = op.valueReadyCycle;
        pipeAvailableAt[resource] = op.resourceReleaseCycle;
      }
    } else if (op.pipe == HIVMPipe::Unknown &&
               !usesFlowControlResource(op)) {
      op.startCycle = earliest;
      op.issueDuration = op.issueDuration ? op.issueDuration : op.duration;
      op.dependencyLatency =
          op.dependencyLatency ? op.dependencyLatency : op.duration;
      op.resourceReleaseCycle = earliest + op.issueDuration;
      op.valueReadyCycle = earliest + op.dependencyLatency;
      op.endCycle = op.valueReadyCycle;
    } else {
      ScheduleResourceKey resource = getScheduleResource(op);
      int64_t start = std::max(earliest, pipeAvailableAt[resource]);
      op.startCycle = start;
      op.issueDuration = op.issueDuration ? op.issueDuration : op.duration;
      op.dependencyLatency =
          op.dependencyLatency ? op.dependencyLatency : op.duration;
      op.resourceReleaseCycle = start + op.issueDuration;
      op.valueReadyCycle = start + op.dependencyLatency;
      op.endCycle = op.valueReadyCycle;
      pipeAvailableAt[resource] = op.resourceReleaseCycle;
    }

    report.oneIterationCycles = std::max(report.oneIterationCycles, op.endCycle);
    report.totalBusyCycles += op.duration;
    report.opCount++;
    recordCalibrationSummary(report, op);
    if (isReportableScalarFallback(op))
      report.unknownOpCount++;
    if (op.isSyncOp) {
      report.syncCycles += op.duration;
      report.syncIssueCycles += op.issueDuration;
      report.syncEventWaitCycles += op.eventWaitCycles;
      report.syncOpCount++;
    }
    if (op.isBarrier) {
      report.barrierCycles += op.duration;
      report.barrierCount++;
    }
    if (op.pipe != HIVMPipe::All && op.pipe != HIVMPipe::Unknown) {
      report.pipeBusyCycles[op.pipe] += op.duration;
      report.weightedPipeCycles[op.pipe] += op.duration * op.loopMultiplier;
    }
  }

  int64_t globalBarrierWeightedCycles = 0;
  for (const HIVMOp &op : report.operations) {
    if (op.isBarrier && op.pipe == HIVMPipe::All)
      globalBarrierWeightedCycles += op.duration * op.loopMultiplier;
  }
  for (const auto &entry : report.weightedPipeCycles)
    report.weightedCycles = std::max(report.weightedCycles, entry.second);
  report.weightedCycles += globalBarrierWeightedCycles;
  if (report.weightedCycles == 0)
    report.weightedCycles = report.oneIterationCycles;
}

struct CompletionEvent {
  int64_t time = 0;
  size_t opId = 0;

  bool operator>(const CompletionEvent &other) const {
    return std::tie(time, opId) > std::tie(other.time, other.opId);
  }
};

struct BufferSlotState {
  int64_t writableAt = 0;
  int64_t readableAt = 0;
  int64_t version = 0;
};

struct BufferRootState {
  std::vector<BufferSlotState> slots;
  int64_t latestReadableAt = 0;
  int64_t latestVersion = 0;
  std::map<int64_t, int64_t> versionReadableAt;
  std::map<int64_t, size_t> versionToSlot;
};

static llvm::StringRef getSyncBlockSourceCore(const HIVMOp &op) {
  bool isCubeCore = op.coreType == "CUBE" || op.coreType == "AIC";
  if (op.opName == "sync_block_set")
    return isCubeCore ? "AIC" : "AIV";
  if (op.opName == "sync_block_wait")
    return isCubeCore ? "AIV" : "AIC";
  return "";
}

static void normalizeSyncBlockGenerations(HIVMAnalysisReport &report) {
  std::map<std::pair<std::string, std::string>, int64_t> setGeneration;
  std::map<std::pair<std::string, std::string>, int64_t> waitGeneration;
  for (HIVMOp &op : report.operations) {
    if ((op.opName != "sync_block_set" && op.opName != "sync_block_wait") ||
        op.eventId.empty())
      continue;
    llvm::StringRef sourceCore = getSyncBlockSourceCore(op);
    if (sourceCore.empty())
      continue;
    auto key = std::make_pair(op.eventId, sourceCore.str());
    if (op.opName == "sync_block_set")
      op.eventGeneration = ++setGeneration[key];
    else
      op.eventGeneration = ++waitGeneration[key];
  }
}

/// After generation normalization, wire explicit dependency edges from each
/// sync_block_set to its matching sync_block_wait so the DES respects
/// cross-core ordering.  Without this, the wait may be scheduled before the
/// set completes (they live in different func::FuncOps with independent state).
static void wireCrossCoreSyncDependencies(HIVMAnalysisReport &report) {
  // Key: (eventId, sourceCore, generation) → set-op id
  using SyncKey = std::tuple<std::string, std::string, int64_t>;
  std::map<SyncKey, size_t> setOpById;
  for (HIVMOp &op : report.operations) {
    if (op.opName != "sync_block_set" || op.eventId.empty())
      continue;
    llvm::StringRef sourceCore = getSyncBlockSourceCore(op);
    if (sourceCore.empty())
      continue;
    SyncKey key{op.eventId, sourceCore.str(), op.eventGeneration};
    setOpById[key] = op.id;
  }
  for (HIVMOp &op : report.operations) {
    if (op.opName != "sync_block_wait" || op.eventId.empty())
      continue;
    llvm::StringRef sourceCore = getSyncBlockSourceCore(op);
    if (sourceCore.empty())
      continue;
    // sync_block_wait's sourceCore returns the core that *set* the flag
    // (the opposite core), which matches the set-op's sourceCore.
    SyncKey key{op.eventId, sourceCore.str(), op.eventGeneration};
    auto it = setOpById.find(key);
    if (it != setOpById.end()) {
      if (!llvm::is_contained(op.dependsOn, it->second))
        op.dependsOn.push_back(it->second);
      if (!llvm::is_contained(op.eventDependsOn, it->second))
        op.eventDependsOn.push_back(it->second);
    }
  }
}

static void computeCriticalPathSummary(HIVMAnalysisReport &report) {
  report.criticalPathCycles = report.oneIterationCycles;
  report.criticalPathIssueCycles = 0;
  report.criticalPathEventWaitCycles = 0;
  report.criticalPathOps.clear();
  if (report.operations.empty())
    return;

  size_t current = std::numeric_limits<size_t>::max();
  int64_t maxEnd = std::numeric_limits<int64_t>::min();
  for (const HIVMOp &op : report.operations) {
    if (op.endCycle > maxEnd) {
      maxEnd = op.endCycle;
      current = op.id;
    }
  }
  if (current >= report.operations.size())
    return;

  std::vector<bool> visited(report.operations.size(), false);
  while (current < report.operations.size() && !visited[current]) {
    visited[current] = true;
    const HIVMOp &op = report.operations[current];
    report.criticalPathOps.push_back(current);
    report.criticalPathIssueCycles += op.issueDuration;
    report.criticalPathEventWaitCycles += op.eventWaitCycles;

    size_t next = std::numeric_limits<size_t>::max();
    int64_t nextEnd = std::numeric_limits<int64_t>::min();
    auto considerDep = [&](size_t depId) {
      if (depId >= report.operations.size())
        return;
      const HIVMOp &dep = report.operations[depId];
      if (dep.endCycle > nextEnd) {
        nextEnd = dep.endCycle;
        next = depId;
      }
    };

    if (op.eventWaitCycles > 0) {
      for (size_t depId : op.eventDependsOn)
        considerDep(depId);
      if (next < report.operations.size()) {
        current = next;
        continue;
      }
    }

    for (size_t depId : op.dependsOn)
      considerDep(depId);
    if (next >= report.operations.size())
      break;
    current = next;
  }

  std::reverse(report.criticalPathOps.begin(), report.criticalPathOps.end());
}

static void finalizeDiscreteEventReport(HIVMAnalysisReport &report,
                                        const HardwareConfig &config) {
  normalizeSyncBlockGenerations(report);
  wireCrossCoreSyncDependencies(report);

  const size_t numOps = report.operations.size();
  if (numOps == 0) {
    report.weightedCycles = 0;
    return;
  }

  std::vector<size_t> remainingDeps(numOps, 0);
  std::vector<int64_t> readyAt(numOps, 0);
  std::vector<llvm::SmallVector<size_t, 4>> successors(numOps);
  std::vector<bool> queued(numOps, false);
  std::vector<bool> started(numOps, false);
  std::vector<bool> completed(numOps, false);
  std::deque<size_t> readyOps;
  std::priority_queue<CompletionEvent, std::vector<CompletionEvent>,
                      std::greater<CompletionEvent>>
      completions;
  std::map<ScheduleResourceKey, int64_t> pipeAvailableAt;
  std::map<EventInstanceKey, int64_t> flagEventVisibleAt;
  std::map<EventInstanceKey, int64_t> blockSyncVisibleAt;
  std::map<std::string, BufferRootState> bufferStates;
  std::map<size_t, std::vector<std::pair<std::string, size_t>>> writeSlotAssignments;
  size_t completedCount = 0;

  for (const HIVMOp &op : report.operations) {
    for (const std::string &root : op.writeBuffers) {
      auto &state = bufferStates[root];
      if (state.slots.empty()) {
        int64_t count = std::max<int64_t>(1, op.multiBufferSlots);
        for (int64_t i = 0; i < count; ++i)
          state.slots.push_back(BufferSlotState{});
      }
      state.versionReadableAt.emplace(0, 0);
    }
    for (const std::string &root : op.readBuffers) {
      auto [it, inserted] = bufferStates.try_emplace(root, BufferRootState{});
      it->second.versionReadableAt.emplace(0, 0);
    }
  }

  for (size_t opId = 0; opId < numOps; ++opId) {
    HIVMOp &op = report.operations[opId];
    remainingDeps[opId] = op.dependsOn.size();
    for (size_t depId : op.dependsOn) {
      if (depId < numOps)
        successors[depId].push_back(opId);
    }
    if (remainingDeps[opId] == 0) {
      readyOps.push_back(opId);
      queued[opId] = true;
    }
  }

  auto completeOp = [&](size_t opId, int64_t time) {
    if (completed[opId])
      return;
    HIVMOp &op = report.operations[opId];
    op.endCycle = time;
    op.valueReadyCycle = time;
    completed[opId] = true;
    ++completedCount;
    report.oneIterationCycles = std::max(report.oneIterationCycles, op.endCycle);
    report.totalBusyCycles += op.duration;
    report.opCount++;
    recordCalibrationSummary(report, op);
    if (isReportableScalarFallback(op))
      report.unknownOpCount++;
    if (op.isSyncOp) {
      report.syncCycles += op.duration;
      report.syncIssueCycles += op.issueDuration;
      report.syncEventWaitCycles += op.eventWaitCycles;
      report.syncOpCount++;
      if ((op.opName == "set_flag" || op.opName == "sync_block_set") &&
          !op.eventId.empty()) {
        EventInstanceKey key{{op.senderPipe, op.receiverPipe, op.eventId},
                             op.eventGeneration};
        if (op.opName == "sync_block_set")
          blockSyncVisibleAt[key] = time;
        else
          flagEventVisibleAt[key] = time;
      }
    }
    auto slotIt = writeSlotAssignments.find(opId);
    if (slotIt != writeSlotAssignments.end()) {
      for (const auto &[root, slotIndex] : slotIt->second) {
        auto rootIt = bufferStates.find(root);
        if (rootIt == bufferStates.end() || slotIndex >= rootIt->second.slots.size())
          continue;
        BufferRootState &state = rootIt->second;
        BufferSlotState &slot = state.slots[slotIndex];
        auto rootVersionIt = llvm::find(op.writeBuffers, root);
        if (rootVersionIt == op.writeBuffers.end())
          continue;
        size_t bufferIdx = std::distance(op.writeBuffers.begin(), rootVersionIt);
        if (bufferIdx >= op.writeBufferVersions.size())
          continue;
        int64_t version = op.writeBufferVersions[bufferIdx];
        slot.readableAt = time;
        slot.version = version;
        state.latestVersion = std::max(state.latestVersion, version);
        state.latestReadableAt = std::max(state.latestReadableAt, time);
        state.versionReadableAt[version] = time;
        state.versionToSlot[version] = slotIndex;
      }
    }
    if (op.isBarrier) {
      report.barrierCycles += op.duration;
      report.barrierCount++;
    }
    if (op.pipe != HIVMPipe::All && op.pipe != HIVMPipe::Unknown) {
      report.pipeBusyCycles[op.pipe] += op.duration;
      report.weightedPipeCycles[op.pipe] += op.duration * op.loopMultiplier;
    }
    for (size_t succId : successors[opId]) {
      readyAt[succId] = std::max(readyAt[succId], time);
      if (remainingDeps[succId] > 0)
        --remainingDeps[succId];
      if (remainingDeps[succId] == 0 && !queued[succId]) {
        readyOps.push_back(succId);
        queued[succId] = true;
      }
    }
  };

  auto applyEventGate = [&](HIVMOp &op, int64_t start) -> int64_t {
    op.eventWaitCycles = 0;
    if (!isSyncWaitOp(op.opName))
      return start;
    int64_t eventVisible = 0;
    for (size_t depId : op.eventDependsOn) {
      if (depId < report.operations.size())
        eventVisible = std::max(eventVisible, report.operations[depId].endCycle);
    }
    EventInstanceKey key{{op.senderPipe, op.receiverPipe, op.eventId},
                         op.eventGeneration};
    auto &visibleAt =
        op.opName == "sync_block_wait" ? blockSyncVisibleAt : flagEventVisibleAt;
    auto it = visibleAt.find(key);
    if (it != visibleAt.end())
      eventVisible = std::max(eventVisible, it->second);
    if (eventVisible == 0)
      return start;
    op.eventWaitCycles = std::max<int64_t>(0, eventVisible - start);
    return std::max(start, eventVisible);
  };

  auto computeStartTime = [&](HIVMOp &op) -> int64_t {
    int64_t start = readyAt[op.id];
    if (isSyncWaitOp(op.opName) && !op.eventDependsOn.empty()) {
      start = 0;
      for (size_t depId : op.dependsOn) {
        if (llvm::is_contained(op.eventDependsOn, depId))
          continue;
        if (depId < report.operations.size())
          start = std::max(start, report.operations[depId].endCycle);
      }
    }
    for (size_t idx = 0; idx < op.readBuffers.size(); ++idx) {
      const std::string &root = op.readBuffers[idx];
      auto it = bufferStates.find(root);
      if (it == bufferStates.end())
        continue;
      int64_t version =
          idx < op.readBufferVersions.size() ? op.readBufferVersions[idx] : 0;
      auto readableIt = it->second.versionReadableAt.find(version);
      if (readableIt != it->second.versionReadableAt.end())
        start = std::max(start, readableIt->second);
      else if (it->second.latestVersion >= version)
        start = std::max(start, it->second.latestReadableAt);
    }
    for (const std::string &root : op.writeBuffers) {
      auto it = bufferStates.find(root);
      if (it != bufferStates.end() && !it->second.slots.empty()) {
        int64_t slotReady = std::numeric_limits<int64_t>::max();
        for (const BufferSlotState &slot : it->second.slots)
          slotReady = std::min(slotReady, slot.writableAt);
        start = std::max(start, slotReady);
      }
    }
    if (usesFlowControlResource(op))
      return applyEventGate(
          op, std::max(start, pipeAvailableAt[getScheduleResource(op)]));
    if (op.pipe == HIVMPipe::Unknown)
      if (!usesFlowControlResource(op))
        return applyEventGate(op, start);
    if (op.isBarrier && op.pipe == HIVMPipe::All) {
      if (op.coreType.empty()) {
        for (const auto &entry : pipeAvailableAt)
          start = std::max(start, entry.second);
      } else {
        for (const auto &entry : pipeAvailableAt) {
          if (scheduleResourceBelongsToCore(entry.first, op.coreType))
            start = std::max(start, entry.second);
        }
      }
      return applyEventGate(op, start);
    }
    return applyEventGate(
        op, std::max(start, pipeAvailableAt[getScheduleResource(op)]));
  };

  auto startOp = [&](size_t opId, int64_t startTime) {
    HIVMOp &op = report.operations[opId];
    started[opId] = true;
    op.startCycle = startTime;
    if (op.issueDuration == 0 && op.duration > 0)
      op.issueDuration = op.duration;
    if (op.dependencyLatency == 0 && op.duration > 0)
      op.dependencyLatency = op.duration;
    const int64_t resourceReleaseTime = startTime + op.issueDuration;
    const int64_t valueReadyTime = startTime + op.dependencyLatency;
    op.resourceReleaseCycle = resourceReleaseTime;
    op.valueReadyCycle = valueReadyTime;
    for (size_t idx = 0; idx < op.readBuffers.size(); ++idx) {
      const std::string &root = op.readBuffers[idx];
      auto it = bufferStates.find(root);
      if (it == bufferStates.end())
        continue;
      int64_t version =
          idx < op.readBufferVersions.size() ? op.readBufferVersions[idx] : 0;
      if (version <= 0)
        continue;
      auto slotIt = it->second.versionToSlot.find(version);
      if (slotIt == it->second.versionToSlot.end())
        continue;
      size_t slotIndex = slotIt->second;
      if (slotIndex >= it->second.slots.size())
        continue;
      it->second.slots[slotIndex].writableAt =
          std::max(it->second.slots[slotIndex].writableAt, resourceReleaseTime);
    }
    for (const std::string &root : op.writeBuffers) {
      auto it = bufferStates.find(root);
      if (it == bufferStates.end() || it->second.slots.empty())
        continue;
      BufferRootState &state = it->second;
      size_t bestSlot = 0;
      int64_t bestTime = state.slots.front().writableAt;
      for (size_t i = 1; i < state.slots.size(); ++i) {
        if (state.slots[i].writableAt < bestTime) {
          bestTime = state.slots[i].writableAt;
          bestSlot = i;
        }
      }
      state.slots[bestSlot].writableAt = resourceReleaseTime;
      writeSlotAssignments[opId].push_back({root, bestSlot});
    }
    if (op.pipe != HIVMPipe::Unknown) {
      if (usesFlowControlResource(op)) {
        pipeAvailableAt[getScheduleResource(op)] = resourceReleaseTime;
      } else if (op.isBarrier && op.pipe == HIVMPipe::All) {
        auto barrierPipes = getCoreBarrierPipes(op.coreType);
        if (barrierPipes.empty()) {
          for (auto &entry : pipeAvailableAt)
            entry.second = resourceReleaseTime;
        } else {
          for (HIVMPipe barrierPipe : barrierPipes)
            pipeAvailableAt[getScheduleResource(barrierPipe, op.coreType)] =
                resourceReleaseTime;
        }
      } else {
        pipeAvailableAt[getScheduleResource(op)] = resourceReleaseTime;
      }
    }
    if (op.dependencyLatency == 0)
      completeOp(opId, valueReadyTime);
    else
      completions.push({valueReadyTime, opId});
  };

  int64_t currentTime = 0;
  // Safety: prevent infinite loops on degenerate inputs.
  const size_t maxIterations = numOps * 100 + 10000;
  size_t iterationCount = 0;
  while (completedCount < numOps) {
    if (++iterationCount > maxIterations) {
      llvm::errs() << "DES scheduler: exceeded max iterations ("
                   << maxIterations << "), completed " << completedCount
                   << "/" << numOps << " ops\n";
      report.scheduleTruncated = true;
      break;
    }
    bool startedAny = false;
    size_t readyCount = readyOps.size();
    for (size_t i = 0; i < readyCount; ++i) {
      size_t opId = readyOps.front();
      readyOps.pop_front();
      HIVMOp &op = report.operations[opId];
      if (started[opId] || completed[opId])
        continue;
      int64_t startTime = computeStartTime(op);
      if (startTime <= currentTime) {
        startOp(opId, currentTime);
        startedAny = true;
      } else {
        readyOps.push_back(opId);
      }
    }

    while (!completions.empty() && completions.top().time <= currentTime) {
      size_t opId = completions.top().opId;
      completions.pop();
      if (!completed[opId]) {
        completeOp(opId, currentTime);
      }
    }

    if (startedAny)
      continue;

    int64_t nextTime = std::numeric_limits<int64_t>::max();
    if (!completions.empty())
      nextTime = std::min(nextTime, completions.top().time);
    for (size_t opId : readyOps)
      nextTime = std::min(nextTime, computeStartTime(report.operations[opId]));

    if (nextTime == std::numeric_limits<int64_t>::max()) {
      report.scheduleTruncated = true;
      break;
    }
    currentTime = std::max(currentTime, nextTime);

    while (!completions.empty() && completions.top().time <= currentTime) {
      size_t opId = completions.top().opId;
      completions.pop();
      if (!completed[opId]) {
        completeOp(opId, currentTime);
      }
    }
  }

  int64_t globalBarrierWeightedCycles = 0;
  for (const HIVMOp &op : report.operations) {
    if (op.isBarrier && op.pipe == HIVMPipe::All)
      globalBarrierWeightedCycles += op.duration * op.loopMultiplier;
  }
  for (const auto &entry : report.weightedPipeCycles)
    report.weightedCycles = std::max(report.weightedCycles, entry.second);
  report.weightedCycles += globalBarrierWeightedCycles;
  if (report.weightedCycles == 0)
    report.weightedCycles = report.oneIterationCycles;
  computeCriticalPathSummary(report);
}

enum class SemanticComponent {
  Vector,
  Cube,
  Scalar,
  MTEGM,
  MTEL1,
  MTEUB,
};

struct SemanticWorkKey {
  SemanticComponent component = SemanticComponent::Scalar;
  std::string opName;
  std::string elemType;
  bool isFlops = false;

  bool operator<(const SemanticWorkKey &other) const {
    return std::tie(component, opName, elemType, isFlops) <
           std::tie(other.component, other.opName, other.elemType,
                    other.isFlops);
  }

  bool operator==(const SemanticWorkKey &other) const {
    return component == other.component && opName == other.opName &&
           elemType == other.elemType && isFlops == other.isFlops;
  }
};

struct SemanticSummary {
  std::map<SemanticWorkKey, int64_t> work;
  size_t vectorOps = 0;
  size_t cubeOps = 0;
  size_t scalarOps = 0;
  size_t transferOps = 0;
  size_t unsupportedOps = 0;
  size_t resolvedLoops = 0;
  size_t unresolvedLoops = 0;
  size_t resolvedBranches = 0;
  size_t equivalentBranches = 0;
  size_t unresolvedBranches = 0;

  void add(SemanticWorkKey key, int64_t amount, int64_t multiplier) {
    if (amount <= 0 || multiplier <= 0)
      return;
    if (amount > std::numeric_limits<int64_t>::max() / multiplier)
      amount = std::numeric_limits<int64_t>::max();
    else
      amount *= multiplier;
    work[key] += amount;
    switch (key.component) {
    case SemanticComponent::Vector:
      ++vectorOps;
      break;
    case SemanticComponent::Cube:
      ++cubeOps;
      break;
    case SemanticComponent::Scalar:
      ++scalarOps;
      break;
    case SemanticComponent::MTEGM:
    case SemanticComponent::MTEL1:
    case SemanticComponent::MTEUB:
      ++transferOps;
      break;
    }
  }
};

static int64_t getStaticElementCount(mlir::Type type) {
  if (!type)
    return 0;
  auto shaped = llvm::dyn_cast<mlir::ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  return shaped.getNumElements();
}

static mlir::Type getElementType(mlir::Type type) {
  if (!type)
    return type;
  if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(type))
    return shaped.getElementType();
  return type;
}

static std::string stringifyElementType(mlir::Type type) {
  if (!type)
    return "";
  type = getElementType(type);
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
    return "i" + std::to_string(intType.getWidth());
  return "";
}

static mlir::Type getLargestShapedType(mlir::Operation *op) {
  mlir::Type best;
  int64_t bestElements = 0;
  auto consider = [&](mlir::Type type) {
    int64_t elements = getStaticElementCount(type);
    if (elements > bestElements) {
      best = type;
      bestElements = elements;
    }
  };
  for (mlir::Type type : op->getResultTypes())
    consider(type);
  for (mlir::Value operand : op->getOperands())
    consider(operand.getType());
  return best;
}

static int64_t getLargestStaticElementCount(mlir::Operation *op) {
  return getStaticElementCount(getLargestShapedType(op));
}

static int64_t getStaticByteCount(mlir::Type type) {
  int64_t elements = getStaticElementCount(type);
  if (elements <= 0)
    return 0;
  std::string elemType = stringifyElementType(type);
  int64_t width = getElementByteWidth(elemType);
  return width > 0 ? elements * width : 0;
}

static int64_t getLargestStaticByteCount(mlir::Operation *op) {
  int64_t bytes = 0;
  for (mlir::Type type : op->getResultTypes())
    bytes = std::max(bytes, getStaticByteCount(type));
  for (mlir::Value operand : op->getOperands())
    bytes = std::max(bytes, getStaticByteCount(operand.getType()));
  return bytes;
}

static int64_t getLargestByteCountWithBindings(mlir::Operation *op,
                                               const AnalysisState &state) {
  int64_t bytes = getLargestStaticByteCount(op);
  for (mlir::Value result : op->getResults())
    bytes = std::max(bytes, inferValueBytesWithBindings(result, state));
  for (mlir::Value operand : op->getOperands())
    bytes = std::max(bytes, inferValueBytesWithBindings(operand, state));
  return bytes;
}

static std::optional<std::string> mapSemanticVectorOp(llvm::StringRef name) {
  if (name == "arith.addf" || name == "arith.addi")
    return "vadd";
  if (name == "arith.subf" || name == "arith.subi")
    return "vsub";
  if (name == "arith.mulf" || name == "arith.muli")
    return "vmul";
  if (name == "arith.divf" || name == "arith.divsi" ||
      name == "arith.divui")
    return "vdiv";
  if (name == "arith.maxnumf" || name == "arith.maxsi" ||
      name == "arith.maxui")
    return "vmax";
  if (name == "arith.minnumf" || name == "arith.minsi" ||
      name == "arith.minui")
    return "vmin";
  if (name == "arith.extf" || name == "arith.truncf" ||
      name == "arith.extsi" || name == "arith.extui" ||
      name == "arith.trunci" || name == "arith.uitofp" ||
      name == "arith.sitofp")
    return "vcast";
  if (name == "arith.cmpi" || name == "arith.cmpf")
    return "vcmp";
  if (name == "arith.select")
    return "vsel";
  if (name == "arith.ori")
    return "vor";
  if (name == "math.exp" || name == "math.exp2")
    return "vexp";
  if (name == "math.log" || name == "math.log2")
    return "vlog";
  if (name == "math.absf" || name == "math.absi")
    return "vabs";
  if (name == "math.sqrt")
    return "vsqrt";
  if (name == "math.rsqrt")
    return "vrsqrt";
  if (name == "math.tanh")
    return "vtanh";
  if (name == "linalg.index")
    return "varange";
  return std::nullopt;
}

static bool hasShapedValue(mlir::Operation *op) {
  return llvm::any_of(op->getResultTypes(), [](mlir::Type type) {
           return llvm::isa<mlir::ShapedType>(type);
         }) ||
         llvm::any_of(op->getOperands(), [](mlir::Value value) {
           return llvm::isa<mlir::ShapedType>(value.getType());
         });
}

static bool isGlobalSemanticValue(mlir::Value value,
                                  mlir::func::FuncOp entry) {
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value))
    return blockArg.getOwner() == &entry.getBody().front() &&
           llvm::isa<mlir::MemRefType>(blockArg.getType());
  mlir::Operation *def = value.getDefiningOp();
  if (!def || def->getNumOperands() == 0)
    return false;
  llvm::StringRef name = def->getName().getStringRef();
  if (name == "memref.reinterpret_cast" || name == "memref.subview" ||
      name == "memref.cast")
    return isGlobalSemanticValue(def->getOperand(0), entry);
  return false;
}

static void seedSemanticFunctionBindings(mlir::func::FuncOp func,
                                         AnalysisState &state) {
  llvm::SmallVector<unsigned, 8> userArgToActual;
  bool hasTypedSyntheticArgs = false;
  for (unsigned idx = 0; idx < func.getNumArguments(); ++idx) {
    if (func.getArgAttr(idx, "hacc.arg_type")) {
      hasTypedSyntheticArgs = true;
      continue;
    }
    userArgToActual.push_back(idx);
  }

  if (!hasTypedSyntheticArgs) {
    std::set<unsigned> synthetic;
    for (llvm::StringRef attrName : {"SyncBlockLockArgIdx",
                                     "WorkspaceArgIdx"}) {
      if (auto attr = func->getAttrOfType<mlir::IntegerAttr>(attrName)) {
        int64_t idx = attr.getInt();
        if (idx >= 0 && idx < func.getNumArguments())
          synthetic.insert(static_cast<unsigned>(idx));
      }
    }
    userArgToActual.clear();
    for (unsigned idx = 0; idx < func.getNumArguments(); ++idx) {
      if (synthetic.find(idx) == synthetic.end())
        userArgToActual.push_back(idx);
    }
  }

  auto bindable = [&](unsigned idx) {
    if (idx >= func.getNumArguments())
      return false;
    mlir::Type type = func.getArgument(idx).getType();
    return llvm::isa<mlir::IntegerType, mlir::IndexType>(type);
  };

  for (const auto &entry : state.argBindings) {
    llvm::StringRef name(entry.first);
    if (!name.consume_front("arg"))
      continue;
    unsigned userIndex = 0;
    if (name.getAsInteger(10, userIndex))
      continue;
    std::optional<unsigned> actualIndex;
    if (userIndex < userArgToActual.size() &&
        bindable(userArgToActual[userIndex]))
      actualIndex = userArgToActual[userIndex];
    else if (bindable(userIndex))
      actualIndex = userIndex;
    if (actualIndex)
      state.boundValues[func.getArgument(*actualIndex)] = entry.second;
  }

  if (func.getNumArguments() >= 3) {
    unsigned firstPid = func.getNumArguments() - 3;
    constexpr llvm::StringLiteral pidNames[] = {"pid_x", "pid_y", "pid_z"};
    for (auto [offset, name] : llvm::enumerate(pidNames)) {
      auto binding = state.argBindings.find(name.str());
      if (binding != state.argBindings.end() && bindable(firstPid + offset))
        state.boundValues[func.getArgument(firstPid + offset)] =
            binding->second;
    }
  }
}

static void mergeMinimumBranchWork(SemanticSummary &target,
                                   const SemanticSummary &thenSummary,
                                   const SemanticSummary &elseSummary) {
  for (const auto &entry : thenSummary.work) {
    auto other = elseSummary.work.find(entry.first);
    if (other != elseSummary.work.end())
      target.work[entry.first] += std::min(entry.second, other->second);
  }
  target.unsupportedOps +=
      thenSummary.unsupportedOps + elseSummary.unsupportedOps;
  target.unresolvedLoops +=
      thenSummary.unresolvedLoops + elseSummary.unresolvedLoops;
  target.equivalentBranches += thenSummary.equivalentBranches +
                               elseSummary.equivalentBranches;
  target.unresolvedBranches += thenSummary.unresolvedBranches +
                               elseSummary.unresolvedBranches;
}

static bool haveEquivalentBranchWork(const SemanticSummary &lhs,
                                     const SemanticSummary &rhs) {
  return lhs.work == rhs.work && lhs.unsupportedOps == 0 &&
         rhs.unsupportedOps == 0 && lhs.unresolvedLoops == 0 &&
         rhs.unresolvedLoops == 0 && lhs.unresolvedBranches == 0 &&
         rhs.unresolvedBranches == 0;
}

static void analyzeSemanticRegion(mlir::Region &region, int64_t multiplier,
                                  AnalysisState &state,
                                  SemanticSummary &summary,
                                  mlir::func::FuncOp entry,
                                  std::set<mlir::Operation *> &callStack);

static void addVectorSemanticOp(mlir::Operation *op, llvm::StringRef opName,
                                int64_t elements, int64_t multiplier,
                                SemanticSummary &summary) {
  mlir::Type type = getLargestShapedType(op);
  summary.add({SemanticComponent::Vector, opName.str(),
               stringifyElementType(type), false},
              elements, multiplier);
}

static void analyzeSemanticOperation(mlir::Operation *op, int64_t multiplier,
                                     AnalysisState &state,
                                     SemanticSummary &summary,
                                     mlir::func::FuncOp entry,
                                     std::set<mlir::Operation *> &callStack) {
  if (captureConstant(op, state))
    return;
  captureDerivedScalarValue(op, state);

  if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
    int64_t tripCount = 1;
    bool resolved = parseForTripCount(forOp, state, tripCount);
    if (resolved)
      ++summary.resolvedLoops;
    else
      ++summary.unresolvedLoops;
    AnalysisState loopState = state;
    seedLoopCarriedState(forOp, state, loopState);
    int64_t lower = 0;
    int64_t step = 1;
    bool replay = resolved && tripCount > 0 && tripCount <= 1000000 &&
                  resolveMLIRValue(forOp.getLowerBound(), state, lower) &&
                  resolveMLIRValue(forOp.getStep(), state, step);
    if (replay) {
      for (int64_t iteration = 0; iteration < tripCount; ++iteration) {
        loopState.boundValues[forOp.getInductionVar()] =
            lower + iteration * step;
        analyzeSemanticRegion(forOp.getRegion(), multiplier, loopState,
                              summary, entry, callStack);
        if (iteration + 1 < tripCount)
          advanceLoopCarriedState(forOp, loopState);
      }
    } else {
      int64_t nestedMultiplier =
          multiplier * std::max<int64_t>(tripCount, 1);
      analyzeSemanticRegion(forOp.getRegion(), nestedMultiplier, loopState,
                            summary, entry, callStack);
    }
    propagateLoopResults(forOp, loopState, state);
    return;
  }

  if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(op)) {
    int64_t condition = 0;
    if (resolveMLIRValue(ifOp.getCondition(), state, condition)) {
      ++summary.resolvedBranches;
      mlir::Region &selected =
          condition != 0 ? ifOp.getThenRegion() : ifOp.getElseRegion();
      if (!selected.empty())
        analyzeSemanticRegion(selected, multiplier, state, summary, entry,
                              callStack);
    } else {
      SemanticSummary thenSummary;
      SemanticSummary elseSummary;
      AnalysisState thenState = state;
      AnalysisState elseState = state;
      analyzeSemanticRegion(ifOp.getThenRegion(), multiplier, thenState,
                            thenSummary, entry, callStack);
      if (!ifOp.getElseRegion().empty())
        analyzeSemanticRegion(ifOp.getElseRegion(), multiplier, elseState,
                              elseSummary, entry, callStack);
      if (haveEquivalentBranchWork(thenSummary, elseSummary)) {
        summary.equivalentBranches +=
            1 + thenSummary.equivalentBranches +
            elseSummary.equivalentBranches;
        for (const auto &entry : thenSummary.work)
          summary.work[entry.first] += entry.second;
      } else {
        ++summary.unresolvedBranches;
        mergeMinimumBranchWork(summary, thenSummary, elseSummary);
      }
    }
    return;
  }

  if (auto call = llvm::dyn_cast<mlir::func::CallOp>(op)) {
    auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
        call, call.getCalleeAttr());
    if (!callee || callee.isDeclaration()) {
      if (call.getCallee().contains("cumsum") && call.getNumOperands() >= 1) {
        auto tensorType =
            llvm::dyn_cast<mlir::RankedTensorType>(call.getOperand(0).getType());
        int64_t axis = 0;
        if (call.getNumOperands() >= 2)
          resolveMLIRValue(call.getOperand(1), state, axis);
        if (tensorType && tensorType.hasStaticShape() && axis >= 0 &&
            axis < tensorType.getRank()) {
          int64_t total = tensorType.getNumElements();
          int64_t axisSize = tensorType.getDimSize(axis);
          int64_t adds = axisSize > 0 ? total - total / axisSize : 0;
          summary.add({SemanticComponent::Vector, "vadd",
                       stringifyElementType(tensorType), false},
                      adds, multiplier);
          return;
        }
      }
      ++summary.unsupportedOps;
      return;
    }
    if (!callStack.insert(callee.getOperation()).second) {
      ++summary.unsupportedOps;
      return;
    }
    AnalysisState calleeState;
    calleeState.argBindings = state.argBindings;
    for (auto [idx, operand] : llvm::enumerate(call.getOperands())) {
      if (idx >= callee.getNumArguments())
        break;
      int64_t value = 0;
      if (resolveMLIRValue(operand, state, value))
        calleeState.boundValues[callee.getArgument(idx)] = value;
    }
    analyzeSemanticRegion(callee.getBody(), multiplier, calleeState, summary,
                          entry, callStack);
    callStack.erase(callee.getOperation());
    return;
  }

  llvm::StringRef name = op->getName().getStringRef();
  if (name == "linalg.matmul") {
    if (op->getNumOperands() < 2) {
      ++summary.unsupportedOps;
      return;
    }
    auto lhs = llvm::dyn_cast<mlir::RankedTensorType>(
        op->getOperand(0).getType());
    auto rhs = llvm::dyn_cast<mlir::RankedTensorType>(
        op->getOperand(1).getType());
    if (!lhs || !rhs || !lhs.hasStaticShape() || !rhs.hasStaticShape() ||
        lhs.getRank() != 2 || rhs.getRank() != 2) {
      ++summary.unsupportedOps;
      return;
    }
    int64_t flops = 2 * lhs.getDimSize(0) * lhs.getDimSize(1) *
                    rhs.getDimSize(1);
    summary.add({SemanticComponent::Cube, "matmul",
                 stringifyElementType(lhs), true},
                flops, multiplier);
    return;
  }

  // Fill and broadcast are indexing semantics in TTAdapter. Their consumers
  // can fuse them without issuing a standalone vector instruction, so
  // charging them here would make a theoretical lower bound unsound.
  if (name == "linalg.fill" || name == "linalg.broadcast")
    return;

  if (name == "linalg.transpose") {
    int64_t elements = getLargestStaticElementCount(op);
    if (elements > 0)
      addVectorSemanticOp(op, "vtranspose", elements, multiplier, summary);
    else
      ++summary.unsupportedOps;
    return;
  }

  if (name == "linalg.reduce") {
    int64_t elements = op->getNumOperands() > 0
                           ? getStaticElementCount(op->getOperand(0).getType())
                           : 0;
    std::string reduceName = "vreduce";
    op->walk([&](mlir::Operation *nested) {
      llvm::StringRef nestedName = nested->getName().getStringRef();
      if (nestedName.contains("max"))
        reduceName = "vreduce_max";
      else if (nestedName.contains("min"))
        reduceName = "vreduce_min";
      else if (nestedName.contains("mul"))
        reduceName = "vreduce_prod";
    });
    if (elements > 0)
      addVectorSemanticOp(op, reduceName, elements, multiplier, summary);
    else
      ++summary.unsupportedOps;
    return;
  }

  if (name == "linalg.generic") {
    int64_t elements = getLargestStaticElementCount(op);
    size_t before = summary.vectorOps;
    for (mlir::Region &nestedRegion : op->getRegions()) {
      nestedRegion.walk([&](mlir::Operation *nested) {
        if (nested == op || nested->getName().getStringRef() == "linalg.yield")
          return;
        if (auto mapped =
                mapSemanticVectorOp(nested->getName().getStringRef()))
          addVectorSemanticOp(op, *mapped, elements, multiplier, summary);
      });
    }
    if (summary.vectorOps == before)
      ++summary.unsupportedOps;
    return;
  }

  if (name == "memref.copy" && op->getNumOperands() >= 2) {
    bool srcGlobal = isGlobalSemanticValue(op->getOperand(0), entry);
    bool dstGlobal = isGlobalSemanticValue(op->getOperand(1), entry);
    SemanticComponent component =
        srcGlobal ? SemanticComponent::MTEGM : SemanticComponent::MTEUB;
    if (!srcGlobal && !dstGlobal)
      component = SemanticComponent::MTEUB;
    int64_t bytes = inferValueBytesWithBindings(op->getOperand(0), state);
    if (bytes <= 0)
      bytes = inferValueBytesWithBindings(op->getOperand(1), state);
    summary.add({component, srcGlobal ? "load" : "store",
                 stringifyElementType(op->getOperand(0).getType()), false},
                bytes, multiplier);
    return;
  }

  if (name == "bufferization.materialize_in_destination" ||
      name == "hivm.hir.store") {
    int64_t bytes = getLargestByteCountWithBindings(op, state);
    summary.add({SemanticComponent::MTEUB, "store",
                 stringifyElementType(getLargestShapedType(op)), false},
                bytes, multiplier);
    return;
  }

  if (name == "memref.load") {
    mlir::Type resultType = op->getNumResults() ? op->getResult(0).getType()
                                                : mlir::Type();
    int64_t width = getElementByteWidth(stringifyElementType(resultType));
    if (op->getNumOperands() &&
        isGlobalSemanticValue(op->getOperand(0), entry)) {
      summary.add({SemanticComponent::MTEGM, "load",
                   stringifyElementType(resultType), false},
                  std::max<int64_t>(width, 1), multiplier);
    } else {
      summary.add({SemanticComponent::Scalar, "scalar", "i32", false}, 1,
                  multiplier);
    }
    return;
  }

  if (auto mapped = mapSemanticVectorOp(name)) {
    if (hasShapedValue(op)) {
      int64_t elements = getLargestStaticElementCount(op);
      if (elements > 0)
        addVectorSemanticOp(op, *mapped, elements, multiplier, summary);
      else
        ++summary.unsupportedOps;
    } else {
      summary.add({SemanticComponent::Scalar, "scalar",
                   stringifyElementType(op->getNumResults()
                                            ? op->getResult(0).getType()
                                            : mlir::Type()),
                   false},
                  1, multiplier);
    }
    return;
  }

  if (name == "tensor.extract" || name == "tensor.insert") {
    summary.add({SemanticComponent::Scalar, "scalar", "i32", false}, 1,
                multiplier);
    return;
  }

  if (name == "arith.constant" || name == "func.return" ||
      name == "scf.yield" || name == "linalg.yield" ||
      name == "tensor.empty" || name == "tensor.expand_shape" ||
      name == "tensor.collapse_shape" || name == "tensor.extract_slice" ||
      name == "tensor.insert_slice" || name == "tensor.reshape" ||
      name == "memref.alloc" || name == "memref.reinterpret_cast" ||
      name == "memref.subview" || name == "bufferization.to_tensor" ||
      name == "bufferization.alloc_tensor")
    return;

  if (hasShapedValue(op))
    ++summary.unsupportedOps;
  else if (name.starts_with("arith.") || name.starts_with("math."))
    summary.add({SemanticComponent::Scalar, "scalar", "i32", false}, 1,
                multiplier);
}

static void analyzeSemanticRegion(mlir::Region &region, int64_t multiplier,
                                  AnalysisState &state,
                                  SemanticSummary &summary,
                                  mlir::func::FuncOp entry,
                                  std::set<mlir::Operation *> &callStack) {
  for (mlir::Block &block : region) {
    for (mlir::Operation &op : block)
      analyzeSemanticOperation(&op, multiplier, state, summary, entry,
                               callStack);
  }
}

static std::string semanticGroup(const SemanticWorkKey &key) {
  switch (key.component) {
  case SemanticComponent::Vector:
    return "vector/" + key.opName;
  case SemanticComponent::Cube:
    return "cube";
  case SemanticComponent::Scalar:
    return "scalar";
  case SemanticComponent::MTEGM:
    return "mte_gm";
  case SemanticComponent::MTEL1:
    return "mte_l1";
  case SemanticComponent::MTEUB:
    return "mte_ub";
  }
  return "scalar";
}

static std::optional<std::string> semanticGroup(const HIVMOp &op) {
  if (op.isSyncOp || op.isBarrier)
    return std::nullopt;
  switch (op.pipe) {
  case HIVMPipe::Vector:
    return "vector/" + op.opName;
  case HIVMPipe::Cube:
    return "cube";
  case HIVMPipe::Scalar:
    return "scalar";
  case HIVMPipe::VectorMTE2:
  case HIVMPipe::CubeMTE2:
    return "mte_gm";
  case HIVMPipe::MTE1:
    return "mte_l1";
  case HIVMPipe::MTE3:
  case HIVMPipe::FixPipe:
    return "mte_ub";
  default:
    return std::nullopt;
  }
}

static int64_t semanticAmount(const HIVMOp &op) {
  int64_t amount = 0;
  if (op.pipe == HIVMPipe::VectorMTE2 || op.pipe == HIVMPipe::CubeMTE2 ||
      op.pipe == HIVMPipe::MTE1 || op.pipe == HIVMPipe::MTE3 ||
      op.pipe == HIVMPipe::FixPipe)
    amount = op.bytes;
  else if (op.pipe == HIVMPipe::Cube)
    amount = op.flops > 0 ? op.flops : op.elements;
  else if (op.pipe == HIVMPipe::Scalar)
    amount = op.flops > 0 ? op.flops : op.elements;
  else
    amount = op.flops > 0 ? op.flops : op.elements;
  if (op.pipe == HIVMPipe::Scalar && amount <= 0 && op.duration > 0)
    amount = 1;
  return amount * std::max<int64_t>(op.loopMultiplier, 1);
}

static HIVMPipe semanticPipe(SemanticComponent component) {
  switch (component) {
  case SemanticComponent::Vector:
    return HIVMPipe::Vector;
  case SemanticComponent::Cube:
    return HIVMPipe::Cube;
  case SemanticComponent::Scalar:
    return HIVMPipe::Scalar;
  case SemanticComponent::MTEGM:
    return HIVMPipe::VectorMTE2;
  case SemanticComponent::MTEL1:
    return HIVMPipe::MTE1;
  case SemanticComponent::MTEUB:
    return HIVMPipe::MTE3;
  }
  return HIVMPipe::Scalar;
}

static int64_t estimateSemanticVectorCycles(const SemanticWorkKey &key,
                                            int64_t elements,
                                            const HardwareConfig &config,
                                            bool &calibrated) {
  int64_t instructions =
      ceilDiv(elements, std::max<int>(1, config.getVectorWidthElements()));
  double cyclesPerInstruction =
      config.getVectorOpCyclesPerInstruction(key.opName);
  if (auto dtypeCost = config.lookupVectorOpCyclesPerInstructionByDType(
          key.opName, key.elemType)) {
    cyclesPerInstruction = *dtypeCost;
  } else if (auto cost = config.lookupOpcodeCycleCost("PIPE_V", key.opName)) {
    if (cost->hasCycles && !cost->source.empty()) {
      cyclesPerInstruction = cost->cycles;
    } else {
      calibrated = false;
    }
  } else {
    calibrated = false;
  }
  return std::max<int64_t>(
      1, ceilToI64(static_cast<double>(instructions) * cyclesPerInstruction));
}

static std::string normalizeSemanticElemType(llvm::StringRef elemType) {
  if (elemType == "fp16")
    return "f16";
  if (elemType == "fp32")
    return "f32";
  return elemType.str();
}

static void applySemanticSummary(const SemanticSummary &summary,
                                 HIVMAnalysisReport &report,
                                 const HardwareConfig &config) {
  struct VCallProjection {
    size_t opIndex = 0;
    int64_t elementWeight = 1;
    int64_t loopMultiplier = 1;
  };

  using ExistingSemanticKey = std::pair<std::string, std::string>;
  std::map<ExistingSemanticKey, int64_t> existing;
  std::vector<VCallProjection> vectorCalls;
  int64_t semanticVectorCycles = 0;
  bool semanticVectorCostCalibrated = true;
  report.semanticUnplacedVectorCycles = 0;
  size_t nextId = 0;
  for (auto [opIndex, op] : llvm::enumerate(report.operations)) {
    nextId = std::max(nextId, op.id + 1);
    if (op.opName == "vcall") {
      vectorCalls.push_back(
          {opIndex, std::max<int64_t>(op.elements, 1),
           std::max<int64_t>(op.loopMultiplier, 1)});
      op.elements = 0;
      op.flops = 0;
      op.duration = 0;
      op.issueDuration = 0;
      op.dependencyLatency = 0;
      continue;
    }
    if (auto group = semanticGroup(op)) {
      existing[{*group, normalizeSemanticElemType(op.elemType)}] +=
          semanticAmount(op);
    }
  }

  for (const auto &entry : summary.work) {
    const SemanticWorkKey &key = entry.first;
    std::string group = semanticGroup(key);
    ExistingSemanticKey exactKey{group, normalizeSemanticElemType(key.elemType)};
    int64_t covered = std::min(existing[exactKey], entry.second);
    existing[exactKey] -= covered;
    if (covered < entry.second && !exactKey.second.empty()) {
      ExistingSemanticKey untypedKey{group, ""};
      int64_t untypedCovered =
          std::min(existing[untypedKey], entry.second - covered);
      existing[untypedKey] -= untypedCovered;
      covered += untypedCovered;
    }
    int64_t deficit = entry.second - covered;
    if (deficit <= 0)
      continue;

    if (key.component == SemanticComponent::Vector) {
      int64_t cycles = estimateSemanticVectorCycles(
          key, deficit, config, semanticVectorCostCalibrated);
      if (semanticVectorCycles >
          std::numeric_limits<int64_t>::max() - cycles)
        semanticVectorCycles = std::numeric_limits<int64_t>::max();
      else
        semanticVectorCycles += cycles;
    }

    HIVMOp synthetic;
    synthetic.id = nextId++;
    synthetic.opName = key.opName;
    synthetic.text = "semantic-sidecar aggregate";
    synthetic.pipe = semanticPipe(key.component);
    synthetic.coreType = key.component == SemanticComponent::Cube ? "CUBE"
                                                                   : "VECTOR";
    synthetic.elemType = key.elemType;
    synthetic.loopMultiplier = 1;
    synthetic.costSource = "ttadapter_semantic_overlay";
    synthetic.costSubpipe = group;
    if (key.component == SemanticComponent::MTEGM ||
        key.component == SemanticComponent::MTEL1 ||
        key.component == SemanticComponent::MTEUB) {
      synthetic.bytes = deficit;
      synthetic.packetBytes = deficit;
      if (key.component == SemanticComponent::MTEGM) {
        synthetic.srcSpace = "gm";
        synthetic.dstSpace = "ub";
      } else if (key.component == SemanticComponent::MTEL1) {
        synthetic.srcSpace = "l1";
        synthetic.dstSpace = "l0a";
      } else {
        synthetic.srcSpace = "ub";
        synthetic.dstSpace = "gm";
      }
    } else if (key.isFlops) {
      synthetic.flops = deficit;
    } else {
      synthetic.elements = deficit;
    }
    report.operations.push_back(std::move(synthetic));
    ++report.semanticSyntheticOpCount;
  }

  if (semanticVectorCycles > 0 && !vectorCalls.empty()) {
    long double weightedElements = 0.0;
    for (const VCallProjection &call : vectorCalls)
      weightedElements += static_cast<long double>(call.elementWeight) *
                          call.loopMultiplier;

    std::vector<std::pair<long double, size_t>> remainders;
    int64_t assignedCycles = 0;
    for (const VCallProjection &call : vectorCalls) {
      HIVMOp &op = report.operations[call.opIndex];
      long double share =
          static_cast<long double>(semanticVectorCycles) * call.elementWeight /
          std::max<long double>(weightedElements, 1.0);
      op.duration = static_cast<int64_t>(std::floor(share));
      op.issueDuration = op.duration;
      op.dependencyLatency = op.duration;
      op.calibratedCost = semanticVectorCostCalibrated;
      op.costSource = "ttadapter_semantic_projection";
      op.costSubpipe = "vector/semantic";
      assignedCycles += op.duration * call.loopMultiplier;
      remainders.push_back(
          {(share - std::floor(share)) * call.loopMultiplier, call.opIndex});
    }
    if (assignedCycles < semanticVectorCycles) {
      llvm::sort(remainders, [](const auto &lhs, const auto &rhs) {
        return lhs.first > rhs.first;
      });
      int64_t residual = semanticVectorCycles - assignedCycles;
      for (const auto &remainder : remainders) {
        HIVMOp &op = report.operations[remainder.second];
        int64_t multiplier = std::max<int64_t>(op.loopMultiplier, 1);
        if (residual < multiplier)
          continue;
        ++op.duration;
        ++op.issueDuration;
        ++op.dependencyLatency;
        residual -= multiplier;
        if (residual == 0)
          break;
      }
      report.semanticUnplacedVectorCycles = residual;
    }
  } else if (semanticVectorCycles > 0) {
    report.semanticUnplacedVectorCycles = semanticVectorCycles;
  }
  report.opCount = report.operations.size();
}

} // namespace

llvm::StringRef HIVMAnalyzer::stringifySchedulerMode(HIVMSchedulerMode mode) {
  switch (mode) {
  case HIVMSchedulerMode::Static:
    return "static";
  case HIVMSchedulerMode::DES:
    return "des";
  }
  return "static";
}

llvm::StringRef HIVMAnalyzer::stringifyPipe(HIVMPipe pipe) {
  switch (pipe) {
  case HIVMPipe::Vector:
    return "PIPE_V";
  case HIVMPipe::VectorMTE2:
    return "PIPE_MTE2_V";
  case HIVMPipe::CubeMTE2:
    return "PIPE_MTE2_C";
  case HIVMPipe::MTE3:
    return "PIPE_MTE3";
  case HIVMPipe::Scalar:
    return "PIPE_S";
  case HIVMPipe::FixPipe:
    return "PIPE_FIX";
  case HIVMPipe::Cube:
    return "PIPE_M";
  case HIVMPipe::MTE1:
    return "PIPE_MTE1";
  case HIVMPipe::All:
    return "PIPE_ALL";
  case HIVMPipe::Unknown:
    return "PIPE_UNKNOWN";
  }
  return "PIPE_UNKNOWN";
}

HIVMAnalyzer::HIVMAnalyzer(const HardwareConfig &config,
                           llvm::StringRef argBindings,
                           HIVMSchedulerMode schedulerMode)
    : config(config), argBindingsStr(argBindings.str()),
      schedulerMode(schedulerMode) {}

bool HIVMAnalyzer::analyzeFile(llvm::StringRef path, HIVMAnalysisReport &report,
                               std::string &error) const {
  auto fileOrErr = llvm::MemoryBuffer::getFile(path);
  if (!fileOrErr) {
    error = "failed to read HIVM file: " + path.str();
    return false;
  }

  llvm::StringRef rawBuffer = fileOrErr.get()->getBuffer();
  std::string sanitized = sanitizeMlirBuffer(rawBuffer);

  {
    mlir::DialectRegistry registry;
    registry.insert<mlir::BuiltinDialect, mlir::affine::AffineDialect,
                    mlir::func::FuncDialect, mlir::arith::ArithDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
    registry.insert<mlir::annotation::AnnotationDialect,
                    mlir::hacc::HACCDialect, mlir::hivm::HIVMDialect>();
#endif
    mlir::MLIRContext context(registry);
    context.allowUnregisteredDialects();

    std::string parseDiagnostics;
    mlir::ScopedDiagnosticHandler diagHandler(
        &context, [&](mlir::Diagnostic &diag) {
          llvm::raw_string_ostream os(parseDiagnostics);
          diag.print(os);
          os << "\n";
          return mlir::success();
        });

    std::vector<std::string> parseCandidates;
    // Prefer MLIR-native analysis so scf.for trip counts, arg bindings, and
    // program-id-dependent bounds are reflected in loop_multiplier.  Some
    // dumped NPUIR files are bare func.func bodies with trailing compiler
    // warnings, so sanitize and wrap before falling back to the text scanner.
    parseCandidates.push_back(wrapBareMlirModule(sanitized));
    if (sanitized != parseCandidates.front())
      parseCandidates.push_back(sanitized);
    if (sanitized != rawBuffer)
      parseCandidates.push_back(wrapBareMlirModule(rawBuffer));
    parseCandidates.push_back(rawBuffer.str());

    mlir::ParserConfig parserConfig(&context, /*verifyAfterParse=*/false);
    for (llvm::StringRef buffer : parseCandidates) {
      if (auto module =
              mlir::parseSourceString<mlir::ModuleOp>(buffer, parserConfig)) {
        if (!analyzeModule(*module, report, error))
          return false;
        report.sourcePath = path.str();
        report.sourceMode = "direct-hivm-mlir";
        return true;
      }
    }

    if (analyzeSemanticHivmBuffer(rawBuffer, path, report, config,
                                  argBindingsStr, schedulerMode))
      return true;

    error = "failed to parse HIVM MLIR module";
    if (!parseDiagnostics.empty())
      error += ":\n" + parseDiagnostics;
    return false;
  }
}
bool HIVMAnalyzer::analyzeModule(mlir::ModuleOp module,
                                 HIVMAnalysisReport &report,
                                 std::string &error) const {
  if (!module) {
    error = "null module passed to HIVM analysis";
    return false;
  }

  report = HIVMAnalysisReport();
  report.sourcePath = "<module>";
  report.sourceMode = "mlir-pass";
  report.schedulerMode = schedulerMode;

  AnalysisState state;
  state.argBindings = parseArgBindings(argBindingsStr);
  analyzeParsedRegion(module.getBodyRegion(), 1, state, report, config,
                      schedulerMode == HIVMSchedulerMode::DES);
  if (schedulerMode == HIVMSchedulerMode::DES)
    finalizeDiscreteEventReport(report, config);
  else
    finalizeScheduledReport(report, config);
  applyKernelLaunchOverhead(report, config, state.argBindings, argBindingsStr);
  return true;
}

bool HIVMAnalyzer::overlaySemanticFile(llvm::StringRef path,
                                       HIVMAnalysisReport &report,
                                       std::string &error) const {
  auto fileOrErr = llvm::MemoryBuffer::getFile(path);
  if (!fileOrErr) {
    error = "failed to read semantic IR file: " + path.str();
    return false;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
#ifdef TRITONSIM_HAS_BISHENGIR_HIVM
  registry.insert<mlir::annotation::AnnotationDialect,
                  mlir::hacc::HACCDialect, mlir::hivm::HIVMDialect>();
#endif
  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects();

  std::string parseDiagnostics;
  mlir::ScopedDiagnosticHandler diagHandler(
      &context, [&](mlir::Diagnostic &diag) {
        llvm::raw_string_ostream os(parseDiagnostics);
        diag.print(os);
        os << "\n";
        return mlir::success();
      });
  mlir::ParserConfig parserConfig(&context, /*verifyAfterParse=*/false);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      fileOrErr.get()->getBuffer(), parserConfig);
  if (!module) {
    error = "failed to parse semantic TTAdapter MLIR";
    if (!parseDiagnostics.empty())
      error += ":\n" + parseDiagnostics;
    return false;
  }

  mlir::func::FuncOp entry;
  for (mlir::func::FuncOp func : module->getOps<mlir::func::FuncOp>()) {
    if (func.isDeclaration())
      continue;
    if (func->hasAttr("global_kernel") || func->hasAttr("hacc.entry")) {
      entry = func;
      break;
    }
  }
  if (!entry) {
    for (mlir::func::FuncOp func : module->getOps<mlir::func::FuncOp>()) {
      if (!func.isDeclaration() &&
          !func->hasAttr("hivm.vector_function")) {
        entry = func;
        break;
      }
    }
  }
  if (!entry) {
    error = "semantic TTAdapter MLIR has no analyzable entry function";
    return false;
  }

  AnalysisState state;
  state.argBindings = parseArgBindings(argBindingsStr);
  seedSemanticFunctionBindings(entry, state);
  SemanticSummary summary;
  std::set<mlir::Operation *> callStack = {entry.getOperation()};
  analyzeSemanticRegion(entry.getBody(), 1, state, summary, entry, callStack);
  applySemanticSummary(summary, report, config);

  std::vector<HIVMOp> semanticAggregates;
  std::vector<HIVMOp> schedulableOperations;
  semanticAggregates.reserve(report.semanticSyntheticOpCount);
  schedulableOperations.reserve(report.operations.size());
  for (HIVMOp &op : report.operations) {
    if (op.costSource == "ttadapter_semantic_overlay")
      semanticAggregates.push_back(std::move(op));
    else
      schedulableOperations.push_back(std::move(op));
  }
  report.operations = std::move(schedulableOperations);

  resetScheduleSummary(report);
  if (schedulerMode == HIVMSchedulerMode::DES)
    finalizeDiscreteEventReport(report, config);
  else
    finalizeScheduledReport(report, config);
  applyKernelLaunchOverhead(report, config, state.argBindings, argBindingsStr);
  report.operations.insert(report.operations.end(),
                           std::make_move_iterator(semanticAggregates.begin()),
                           std::make_move_iterator(semanticAggregates.end()));

  report.semanticOverlayApplied = true;
  report.semanticSourcePath = path.str();
  report.semanticVectorOpCount = summary.vectorOps;
  report.semanticCubeOpCount = summary.cubeOps;
  report.semanticScalarOpCount = summary.scalarOps;
  report.semanticTransferOpCount = summary.transferOps;
  report.semanticUnsupportedOpCount = summary.unsupportedOps;
  report.semanticResolvedLoopCount = summary.resolvedLoops;
  report.semanticUnresolvedLoopCount = summary.unresolvedLoops;
  report.semanticResolvedBranchCount = summary.resolvedBranches;
  report.semanticEquivalentBranchCount = summary.equivalentBranches;
  report.semanticUnresolvedBranchCount = summary.unresolvedBranches;
  report.semanticOverlayComplete =
      summary.unsupportedOps == 0 && summary.unresolvedLoops == 0 &&
      summary.unresolvedBranches == 0;
  return true;
}

void HIVMAnalysisReport::print(llvm::raw_ostream &os,
                               const HardwareConfig &config) const {
  os << "=== HIVM Analysis ===\n";
  os << "Source mode: " << sourceMode << "\n";
  os << "Source: " << sourcePath << "\n";
  os << "Scheduler: " << HIVMAnalyzer::stringifySchedulerMode(schedulerMode)
     << "\n";
  os << "Hardware: " << config.getName() << " @ "
     << llvm::format("%.2f", config.getClockFrequencyGHz()) << " GHz\n\n";

  os << "Summary:\n";
  os << "  Operations: " << opCount << "\n";
  os << "  Sync ops: " << syncOpCount << "\n";
  os << "  Barriers: " << barrierCount << "\n";
  os << "  Scalar/unknown fallback ops: " << unknownOpCount << "\n";
  os << "  One-iteration critical path: " << oneIterationCycles << " cycles ("
     << llvm::format("%.3f", config.cyclesToMicroseconds(oneIterationCycles))
     << " us)\n";
  os << "  Weighted pipe max: " << weightedCycles << " cycles ("
     << llvm::format("%.3f", config.cyclesToMicroseconds(weightedCycles))
     << " us)\n";
  os << "  Kernel body: " << bodyCycles << " cycles ("
     << llvm::format("%.3f", config.cyclesToMicroseconds(bodyCycles))
     << " us)\n";
  os << "  Kernel launch overhead: " << kernelLaunchOverheadCycles
     << " cycles ("
     << llvm::format("%.3f",
                     config.cyclesToMicroseconds(kernelLaunchOverheadCycles))
     << " us)";
  if (kernelLaunchBlockDim > 0)
    os << ", block_dim=" << kernelLaunchBlockDim;
  if (kernelLaunchNumWaves > 0)
    os << ", waves=" << kernelLaunchNumWaves;
  if (!kernelLaunchModel.empty())
    os << ", model=" << kernelLaunchModel;
  os << "\n";
  os << "  Predicted total: " << predictedTotalCycles << " cycles ("
     << llvm::format("%.3f", config.cyclesToMicroseconds(predictedTotalCycles))
     << " us)\n";
  os << "  Sync cycles: " << syncCycles << "\n";
  os << "  Sync issue cycles: " << syncIssueCycles << "\n";
  os << "  Sync event wait cycles: " << syncEventWaitCycles << "\n";
  os << "  Critical path issue cycles: " << criticalPathIssueCycles << "\n";
  os << "  Critical path event wait cycles: " << criticalPathEventWaitCycles << "\n";
  os << "  Barrier cycles: " << barrierCycles << "\n";
  os << "  Max loop multiplier: " << maxLoopMultiplier << "\n\n";

  if (semanticOverlayApplied) {
    os << "Semantic overlay:\n";
    os << "  Source: " << semanticSourcePath << "\n";
    os << "  Status: "
       << (semanticOverlayComplete ? "complete" : "partial") << "\n";
    os << "  Ops: vector=" << semanticVectorOpCount
       << ", cube=" << semanticCubeOpCount
       << ", scalar=" << semanticScalarOpCount
       << ", transfer=" << semanticTransferOpCount
       << ", unsupported=" << semanticUnsupportedOpCount << "\n";
    os << "  Synthetic aggregate ops: " << semanticSyntheticOpCount << "\n";
    os << "  Unplaced vector cycles: " << semanticUnplacedVectorCycles << "\n";
    os << "  Loops: resolved=" << semanticResolvedLoopCount
       << ", unresolved=" << semanticUnresolvedLoopCount << "\n";
    os << "  Branches: resolved=" << semanticResolvedBranchCount
       << ", model-equivalent=" << semanticEquivalentBranchCount
       << ", unresolved=" << semanticUnresolvedBranchCount << "\n\n";
  }

  os << "Cost calibration:\n";
  os << "  Calibrated ops: " << calibratedOpCount << ", cycles: "
     << calibratedCycles << ", weighted: " << calibratedWeightedCycles << "\n";
  os << "  Heuristic ops: " << heuristicOpCount << ", cycles: "
     << heuristicCycles << ", weighted: " << heuristicWeightedCycles << "\n";
  if (!costSubpipeStats.empty()) {
    os << "  By subpipe:\n";
    for (const auto &entry : costSubpipeStats) {
      os << "    " << entry.first << ": ops=" << entry.second.ops
         << ", cycles=" << entry.second.cycles
         << ", weighted=" << entry.second.weightedCycles << "\n";
    }
  }
  if (!unclassifiedCostStats.empty()) {
    std::vector<std::pair<std::string, HIVMCostStat>> top(
        unclassifiedCostStats.begin(), unclassifiedCostStats.end());
    std::sort(top.begin(), top.end(), [](const auto &lhs, const auto &rhs) {
      return lhs.second.weightedCycles > rhs.second.weightedCycles;
    });
    os << "  Top unclassified:\n";
    size_t limit = std::min<size_t>(10, top.size());
    for (size_t i = 0; i < limit; ++i) {
      os << "    " << top[i].first << ": ops=" << top[i].second.ops
         << ", cycles=" << top[i].second.cycles
         << ", weighted=" << top[i].second.weightedCycles << "\n";
    }
  }
  os << "\n";

  os << "Loops:\n";
  os << "  Total: " << loopCount << "\n";
  os << "  Resolved: " << resolvedLoopCount << "\n";
  os << "  Unresolved: " << unresolvedLoopCount << "\n";
  os << "  Max trip count: " << maxLoopTripCount << "\n\n";

  os << "Per-pipe utilization (one iteration):\n";
  for (const auto &entry : pipeBusyCycles) {
    double util = oneIterationCycles > 0
                      ? static_cast<double>(entry.second) / oneIterationCycles * 100.0
                      : 0.0;
    os << "  " << HIVMAnalyzer::stringifyPipe(entry.first) << ": "
       << entry.second << " cycles, " << llvm::format("%.1f", util) << "%\n";
  }

  os << "\nPer-pipe weighted cycles:\n";
  for (const auto &entry : weightedPipeCycles) {
    os << "  " << HIVMAnalyzer::stringifyPipe(entry.first) << ": "
       << entry.second << "\n";
  }

  os << "\nTop operations by weighted cost:\n";
  std::vector<const HIVMOp *> sorted;
  sorted.reserve(operations.size());
  for (const HIVMOp &op : operations)
    sorted.push_back(&op);
  std::sort(sorted.begin(), sorted.end(), [](const HIVMOp *lhs, const HIVMOp *rhs) {
    return lhs->duration * lhs->loopMultiplier > rhs->duration * rhs->loopMultiplier;
  });

  size_t limit = std::min<size_t>(10, sorted.size());
  for (size_t i = 0; i < limit; ++i) {
    const HIVMOp *op = sorted[i];
    os << "  line " << op->lineNumber << " " << op->opName << " ["
       << HIVMAnalyzer::stringifyPipe(op->pipe) << "]: "
       << (op->duration * op->loopMultiplier) << " weighted cycles";
    if (!op->coreType.empty())
      os << ", core=" << op->coreType;
    if (op->bytes > 0)
      os << ", " << op->bytes << " bytes";
    if (op->elements > 0)
      os << ", " << op->elements << " elems";
    os << "\n";
  }
}

void HIVMAnalysisReport::emitPerfettoTrace(llvm::raw_ostream &os,
                                           const HardwareConfig &config) const {
  // Assign each pipe a unique tid.  Pipes are grouped into AIC (Cube core)
  // and AIV (Vector core) processes so that Perfetto renders them separately.
  //   AIC pid=1 : Cube, MTE1, CubeMTE2, FixPipe, Scalar(AIC)
  //   AIV pid=2 : Vector, VectorMTE2, MTE3, Scalar(AIV)
  //   Shared pid=3 : All, Unknown  (cross-core barriers / unclassified)
  constexpr int kPidAIC = 1;
  constexpr int kPidAIV = 2;
  constexpr int kPidShared = 3;
  constexpr int kTidFlowControl = 6;
  constexpr int kTidScalarLoadStore = 7;

  auto pipeTid = [](HIVMPipe pipe) -> int {
    switch (pipe) {
    case HIVMPipe::Cube:
      return 1;
    case HIVMPipe::MTE1:
      return 2;
    case HIVMPipe::CubeMTE2:
      return 3;
    case HIVMPipe::FixPipe:
      return 4;
    case HIVMPipe::Scalar:
      return 5;
    case HIVMPipe::Vector:
      return 1;
    case HIVMPipe::VectorMTE2:
      return 2;
    case HIVMPipe::MTE3:
      return 3;
    case HIVMPipe::All:
      return 1;
    case HIVMPipe::Unknown:
      return 5; // Coalesce with Scalar — metadata/address ops
    }
    return 5;
  };

  auto pipePid = [&](HIVMPipe pipe, llvm::StringRef coreType) -> int {
    switch (pipe) {
    case HIVMPipe::Cube:
    case HIVMPipe::MTE1:
    case HIVMPipe::CubeMTE2:
    case HIVMPipe::FixPipe:
      return kPidAIC;
    case HIVMPipe::Vector:
    case HIVMPipe::VectorMTE2:
    case HIVMPipe::MTE3:
      return kPidAIV;
    case HIVMPipe::Scalar:
      // Scalar exists on both cores; assign by op's core_type.
      if (coreType == "CUBE" || coreType == "AIC")
        return kPidAIC;
      return kPidAIV;
    case HIVMPipe::All:
    case HIVMPipe::Unknown:
      if (coreType == "CUBE" || coreType == "AIC")
        return kPidAIC;
      if (coreType == "VECTOR" || coreType == "AIV")
        return kPidAIV;
      return kPidShared;
    }
    return kPidShared;
  };

  auto isScalarLoadStore = [](const HIVMOp &op) {
    return op.pipe == HIVMPipe::Scalar &&
           (op.opName == "load" || op.opName == "store");
  };
  auto tracePid = [&](const HIVMOp &op) {
    return pipePid(op.pipe, op.coreType);
  };
  auto traceTid = [&](const HIVMOp &op) {
    if (usesFlowControlResource(op))
      return kTidFlowControl;
    if (isScalarLoadStore(op))
      return kTidScalarLoadStore;
    return pipeTid(op.pipe);
  };

  auto cyclesToTraceUs = [&](int64_t cycles) -> double {
    return config.cyclesToMicroseconds(cycles);
  };
  auto joinStrings = [](const std::vector<std::string> &values) {
    std::string joined;
    llvm::raw_string_ostream ss(joined);
    for (size_t i = 0; i < values.size(); ++i) {
      if (i)
        ss << ";";
      ss << jsonEscape(values[i]);
    }
    ss.flush();
    return joined;
  };
  auto joinInts = [](const std::vector<int64_t> &values) {
    std::string joined;
    llvm::raw_string_ostream ss(joined);
    for (size_t i = 0; i < values.size(); ++i) {
      if (i)
        ss << ";";
      ss << values[i];
    }
    ss.flush();
    return joined;
  };
  const bool hasVectorProjection = llvm::any_of(operations, [](const HIVMOp &op) {
    return op.costSource == "ttadapter_semantic_projection" && op.duration > 0;
  });
  const size_t unscheduledSemanticOps = llvm::count_if(
      operations, [&](const HIVMOp &op) {
        if (op.costSource != "ttadapter_semantic_overlay" || op.duration > 0)
          return false;
        return op.pipe != HIVMPipe::Vector || !hasVectorProjection;
      });
  const bool traceTimingComplete =
      !scheduleTruncated && unscheduledSemanticOps == 0 &&
      semanticUnplacedVectorCycles == 0 &&
      ((semanticOverlayApplied && semanticOverlayComplete) ||
       (!semanticOverlayApplied && outlinedCallCount == 0 &&
        zeroByteTransferCount == 0 && zeroWorkScalarOpCount == 0));

  os << "{\n  \"traceEvents\": [\n";
  bool first = true;
  auto emitComma = [&]() {
    if (!first)
      os << ",\n";
    first = false;
  };

  // Process names for AIC, AIV, and Shared groups.
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidAIC
     << ",\"tid\":0,\"name\":\"process_name\",\"args\":{\"name\":\"AIC (Cube Core)\"}}";
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidAIV
     << ",\"tid\":0,\"name\":\"process_name\",\"args\":{\"name\":\"AIV (Vector Core)\"}}";
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidShared
     << ",\"tid\":0,\"name\":\"process_name\",\"args\":{\"name\":\"Shared\"}}";

  // AIC pipes: Cube, MTE1, CubeMTE2, FixPipe, Scalar(AIC)
  for (HIVMPipe pipe :
       {HIVMPipe::Cube, HIVMPipe::MTE1, HIVMPipe::CubeMTE2,
        HIVMPipe::FixPipe}) {
    emitComma();
    os << "    {\"ph\":\"M\",\"pid\":" << kPidAIC
       << ",\"tid\":" << pipeTid(pipe)
       << ",\"name\":\"thread_name\",\"args\":{\"name\":\""
       << HIVMAnalyzer::stringifyPipe(pipe) << "\"}}";
  }
  // Scalar thread under AIC
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidAIC
     << ",\"tid\":" << pipeTid(HIVMPipe::Scalar)
     << ",\"name\":\"thread_name\",\"args\":{\"name\":\"Scalar\"}}";
  for (auto [tid, name] :
       {std::pair<int, llvm::StringRef>{kTidFlowControl, "FLOWCTRL"},
        {kTidScalarLoadStore, "SCALARLDST"}}) {
    emitComma();
    os << "    {\"ph\":\"M\",\"pid\":" << kPidAIC << ",\"tid\":" << tid
       << ",\"name\":\"thread_name\",\"args\":{\"name\":\"" << name
       << "\"}}";
  }

  // AIV pipes: Vector, VectorMTE2, MTE3, Scalar(AIV)
  for (HIVMPipe pipe :
       {HIVMPipe::Vector, HIVMPipe::VectorMTE2, HIVMPipe::MTE3}) {
    emitComma();
    os << "    {\"ph\":\"M\",\"pid\":" << kPidAIV
       << ",\"tid\":" << pipeTid(pipe)
       << ",\"name\":\"thread_name\",\"args\":{\"name\":\""
       << HIVMAnalyzer::stringifyPipe(pipe) << "\"}}";
  }
  // Scalar thread under AIV
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidAIV
     << ",\"tid\":" << pipeTid(HIVMPipe::Scalar)
     << ",\"name\":\"thread_name\",\"args\":{\"name\":\"Scalar\"}}";
  for (auto [tid, name] :
       {std::pair<int, llvm::StringRef>{kTidFlowControl, "FLOWCTRL"},
        {kTidScalarLoadStore, "SCALARLDST"}}) {
    emitComma();
    os << "    {\"ph\":\"M\",\"pid\":" << kPidAIV << ",\"tid\":" << tid
       << ",\"name\":\"thread_name\",\"args\":{\"name\":\"" << name
       << "\"}}";
  }

  // Shared process: cross-core barrier track
  emitComma();
  os << "    {\"ph\":\"M\",\"pid\":" << kPidShared
     << ",\"tid\":" << pipeTid(HIVMPipe::All)
     << ",\"name\":\"thread_name\",\"args\":{\"name\":\""
     << HIVMAnalyzer::stringifyPipe(HIVMPipe::All) << "\"}}";

  for (const HIVMOp &op : operations) {
    // Skip zero-cycle metadata ops that are not real scheduled work.
    if (op.duration <= 0 || op.opName == "pointer_cast" ||
        op.opName == "convert_layout")
      continue;
    if (usesFlowControlResource(op) && op.eventWaitCycles > 0) {
      emitComma();
      os << "    {\"ph\":\"X\",\"pid\":" << tracePid(op)
         << ",\"tid\":" << kTidFlowControl
         << ",\"ts\":"
         << llvm::format("%.3f", cyclesToTraceUs(
                                  op.startCycle - op.eventWaitCycles))
         << ",\"dur\":"
         << llvm::format("%.3f", cyclesToTraceUs(op.eventWaitCycles))
         << ",\"name\":\"" << jsonEscape(op.opName)
         << " wait\",\"args\":{\"blocked\":true,\"cycles\":"
         << op.eventWaitCycles << "}}";
    }
    emitComma();
    os << "    {\"ph\":\"X\",\"pid\":" << tracePid(op)
       << ",\"tid\":" << traceTid(op)
       << ",\"ts\":" << llvm::format("%.3f", cyclesToTraceUs(op.startCycle))
       << ",\"dur\":" << llvm::format("%.3f", cyclesToTraceUs(op.duration))
       << ",\"name\":\"" << jsonEscape(op.opName) << "\",\"args\":{"
       << "\"line\":" << op.lineNumber
       << ",\"cycles\":" << op.duration
       << ",\"issue_duration\":" << op.issueDuration
       << ",\"dependency_latency\":" << op.dependencyLatency
       << ",\"event_wait_cycles\":" << op.eventWaitCycles
       << ",\"resource_release_cycle\":" << op.resourceReleaseCycle
       << ",\"value_ready_cycle\":" << op.valueReadyCycle
       << ",\"loop_multiplier\":" << op.loopMultiplier
       << ",\"bytes\":" << op.bytes
       << ",\"packet_bytes\":" << op.packetBytes
       << ",\"elements\":" << op.elements
       << ",\"event_id\":\"" << jsonEscape(op.eventId) << "\""
       << ",\"event_generation\":" << op.eventGeneration
       << ",\"sender_pipe\":\"" << HIVMAnalyzer::stringifyPipe(op.senderPipe)
       << "\""
       << ",\"receiver_pipe\":\"" << HIVMAnalyzer::stringifyPipe(op.receiverPipe)
       << "\""
       << ",\"read_buffers\":\"" << joinStrings(op.readBuffers) << "\""
       << ",\"write_buffers\":\"" << joinStrings(op.writeBuffers) << "\""
       << ",\"read_versions\":\"" << joinInts(op.readBufferVersions) << "\""
       << ",\"write_versions\":\"" << joinInts(op.writeBufferVersions) << "\""
       << ",\"core_type\":\"" << jsonEscape(op.coreType) << "\""
       << ",\"sync\":" << (op.isSyncOp ? "true" : "false")
       << ",\"barrier\":" << (op.isBarrier ? "true" : "false")
       << ",\"src_space\":\"" << jsonEscape(op.srcSpace) << "\""
       << ",\"dst_space\":\"" << jsonEscape(op.dstSpace) << "\""
       << ",\"elem_type\":\"" << jsonEscape(op.elemType) << "\""
       << ",\"calibrated_cost\":" << (op.calibratedCost ? "true" : "false")
       << ",\"cost_source\":\"" << jsonEscape(op.costSource) << "\""
       << ",\"cost_subpipe\":\"" << jsonEscape(op.costSubpipe) << "\""
       << "}}";
  }

  // Emit flow events linking sync_block_set → sync_block_wait across cores.
  // Build a map from (eventId, generation) → set-op index, then match waits.
  {
    // Key: (eventId, generation, sourceCore)
    using SyncKey = std::tuple<std::string, int64_t, std::string>;
    std::map<SyncKey, std::vector<size_t>> setOps;
    std::map<SyncKey, std::vector<size_t>> waitOps;
    for (size_t i = 0; i < operations.size(); ++i) {
      const HIVMOp &op = operations[i];
      if (op.opName == "sync_block_set" && !op.eventId.empty()) {
        bool isCube = op.coreType == "CUBE" || op.coreType == "AIC";
        SyncKey key{op.eventId, op.eventGeneration, isCube ? "AIC" : "AIV"};
        setOps[key].push_back(i);
      } else if (op.opName == "sync_block_wait" && !op.eventId.empty()) {
        // Wait on CUBE core means waiting for AIV→AIC signal.
        bool isCube = op.coreType == "CUBE" || op.coreType == "AIC";
        SyncKey key{op.eventId, op.eventGeneration, isCube ? "AIV" : "AIC"};
        waitOps[key].push_back(i);
      }
    }
    int64_t flowId = 0;
    for (auto &[key, setIndices] : setOps) {
      auto it = waitOps.find(key);
      if (it == waitOps.end())
        continue;
      auto &waits = it->second;
      size_t pairs = std::min(setIndices.size(), waits.size());
      for (size_t p = 0; p < pairs; ++p) {
        const HIVMOp &setOp = operations[setIndices[p]];
        const HIVMOp &waitOp = operations[waits[p]];
        // Flow start at set-op end time
        emitComma();
        os << "    {\"ph\":\"s\",\"id\":" << flowId
           << ",\"pid\":" << tracePid(setOp)
           << ",\"tid\":" << traceTid(setOp)
           << ",\"ts\":" << llvm::format("%.3f",
                  cyclesToTraceUs(setOp.valueReadyCycle))
           << ",\"name\":\"sync\",\"cat\":\"sync\"}";
        // Flow finish at wait-op start time
        emitComma();
        os << "    {\"ph\":\"f\",\"id\":" << flowId
           << ",\"pid\":" << tracePid(waitOp)
           << ",\"tid\":" << traceTid(waitOp)
           << ",\"ts\":" << llvm::format("%.3f",
                  cyclesToTraceUs(waitOp.startCycle))
           << ",\"name\":\"sync\",\"cat\":\"sync\",\"bp\":\"e\"}";
        ++flowId;
      }
    }
  }

  os << "\n  ],\n  \"displayTimeUnit\": \"us\",\n";
  os << "  \"metadata\": {\"timing_coverage\":\""
     << (traceTimingComplete ? "complete" : "partial")
     << "\",\"semantic_placement\":\""
     << (hasVectorProjection ? "weighted_vcall_heuristic" : "none")
     << "\",\"unscheduled_semantic_ops\":" << unscheduledSemanticOps
     << ",\"unplaced_semantic_vector_cycles\":"
     << semanticUnplacedVectorCycles
     << "}\n}\n";
}

void HIVMAnalysisReport::emitDESGraph(llvm::raw_ostream &os,
                                      const HardwareConfig &config) const {
  auto joinStrVec = [](const std::vector<std::string> &v) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) ss << ",";
      ss << "\"" << jsonEscape(v[i]) << "\"";
    }
    ss << "]";
    ss.flush();
    return s;
  };
  auto joinIntVec = [](const std::vector<int64_t> &v) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) ss << ",";
      ss << v[i];
    }
    ss << "]";
    ss.flush();
    return s;
  };
  auto joinSizeVec = [](const std::vector<size_t> &v) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) ss << ",";
      ss << v[i];
    }
    ss << "]";
    ss.flush();
    return s;
  };
  auto emitCostStatMap = [&](const std::map<std::string, HIVMCostStat> &stats) {
    os << "{";
    size_t idx = 0;
    for (const auto &entry : stats) {
      if (idx++)
        os << ",";
      os << "\"" << jsonEscape(entry.first) << "\":{"
         << "\"ops\":" << entry.second.ops
         << ",\"cycles\":" << entry.second.cycles
         << ",\"weighted_cycles\":" << entry.second.weightedCycles
         << "}";
    }
    os << "}";
  };
  auto emitTopCostStats = [&](const std::map<std::string, HIVMCostStat> &stats,
                              size_t limit) {
    std::vector<std::pair<std::string, HIVMCostStat>> top(stats.begin(),
                                                          stats.end());
    std::sort(top.begin(), top.end(), [](const auto &lhs, const auto &rhs) {
      return lhs.second.weightedCycles > rhs.second.weightedCycles;
    });
    os << "[";
    size_t n = std::min(limit, top.size());
    for (size_t i = 0; i < n; ++i) {
      if (i)
        os << ",";
      os << "{\"key\":\"" << jsonEscape(top[i].first) << "\""
         << ",\"ops\":" << top[i].second.ops
         << ",\"cycles\":" << top[i].second.cycles
         << ",\"weighted_cycles\":" << top[i].second.weightedCycles
         << "}";
    }
    os << "]";
  };

  os << "{\n";
  os << "  \"schema_version\": \"a3_hivm_des_v1\",\n";
  os << "  \"schedule_truncated\": " << (scheduleTruncated ? "true" : "false")
     << ",\n";
  os << "  \"clock_ghz\": " << llvm::format("%.3f", config.getClockFrequencyGHz())
     << ",\n";
  os << "  \"opcode_calibration_version\": \""
     << jsonEscape(config.getOpcodeCalibrationVersion()) << "\",\n";
  os << "  \"opcode_calibration_path\": \""
     << jsonEscape(config.getOpcodeCalibrationPath()) << "\",\n";
  const bool semanticCoverageComplete =
      semanticOverlayApplied && semanticOverlayComplete;
  const bool allOutlinedCallsSummarized =
      semanticCoverageComplete ||
      outlinedCallCount == summarizedOutlinedCallCount;
  const bool coverageComplete =
      semanticCoverageComplete ||
      (outlinedCallCount == 0 && zeroByteTransferCount == 0 &&
       zeroWorkScalarOpCount == 0);
  llvm::StringRef coverageStatus =
      coverageComplete
          ? "complete"
          : (allOutlinedCallsSummarized ? "conservative_partial"
                                        : "incomplete");
  const bool hasVectorProjection = llvm::any_of(operations, [](const HIVMOp &op) {
    return op.costSource == "ttadapter_semantic_projection" && op.duration > 0;
  });
  const size_t unscheduledSemanticOps = llvm::count_if(
      operations, [&](const HIVMOp &op) {
        if (op.costSource != "ttadapter_semantic_overlay" || op.duration > 0)
          return false;
        return op.pipe != HIVMPipe::Vector || !hasVectorProjection;
      });
  const bool traceTimingComplete =
      !scheduleTruncated && unscheduledSemanticOps == 0 &&
      semanticUnplacedVectorCycles == 0 &&
      ((semanticOverlayApplied && semanticOverlayComplete) ||
       (!semanticOverlayApplied && outlinedCallCount == 0 &&
        zeroByteTransferCount == 0 && zeroWorkScalarOpCount == 0));
  os << "  \"model_coverage\": {\n";
  os << "    \"status\": \"" << coverageStatus << "\",\n";
  os << "    \"trace_timing_status\": \""
     << (traceTimingComplete ? "complete" : "partial") << "\",\n";
  os << "    \"semantic_placement\": \""
     << (hasVectorProjection ? "weighted_vcall_heuristic" : "none") << "\",\n";
  os << "    \"unscheduled_semantic_ops\": " << unscheduledSemanticOps
     << ",\n";
  os << "    \"unplaced_semantic_vector_cycles\": "
     << semanticUnplacedVectorCycles << ",\n";
  os << "    \"outlined_calls\": " << outlinedCallCount << ",\n";
  os << "    \"summarized_outlined_calls\": "
     << summarizedOutlinedCallCount << ",\n";
  os << "    \"semantic_covered_outlined_calls\": "
     << (semanticCoverageComplete ? outlinedCallCount : 0) << ",\n";
  os << "    \"zero_byte_transfers\": " << zeroByteTransferCount << ",\n";
  os << "    \"zero_work_scalar_ops\": " << zeroWorkScalarOpCount << ",\n";
  os << "    \"semantic_overlay\": {\n";
  os << "      \"applied\": " << (semanticOverlayApplied ? "true" : "false")
     << ",\n";
  os << "      \"complete\": "
     << (semanticOverlayComplete ? "true" : "false") << ",\n";
  os << "      \"source\": \"" << jsonEscape(semanticSourcePath) << "\",\n";
  os << "      \"vector_ops\": " << semanticVectorOpCount << ",\n";
  os << "      \"cube_ops\": " << semanticCubeOpCount << ",\n";
  os << "      \"scalar_ops\": " << semanticScalarOpCount << ",\n";
  os << "      \"transfer_ops\": " << semanticTransferOpCount << ",\n";
  os << "      \"synthetic_ops\": " << semanticSyntheticOpCount << ",\n";
  os << "      \"unplaced_vector_cycles\": "
     << semanticUnplacedVectorCycles << ",\n";
  os << "      \"unsupported_ops\": " << semanticUnsupportedOpCount << ",\n";
  os << "      \"resolved_loops\": " << semanticResolvedLoopCount << ",\n";
  os << "      \"unresolved_loops\": " << semanticUnresolvedLoopCount
     << ",\n";
  os << "      \"resolved_branches\": " << semanticResolvedBranchCount
     << ",\n";
  os << "      \"model_equivalent_branches\": "
     << semanticEquivalentBranchCount << ",\n";
  os << "      \"unresolved_branches\": " << semanticUnresolvedBranchCount
     << "\n";
  os << "    }\n";
  os << "  },\n";
  os << "  \"latency_summary\": {\n";
  os << "    \"body_cycles\": " << bodyCycles << ",\n";
  os << "    \"body_time_us\": "
     << llvm::format("%.6f", config.cyclesToMicroseconds(bodyCycles))
     << ",\n";
  os << "    \"kernel_launch_overhead_cycles\": "
     << kernelLaunchOverheadCycles << ",\n";
  os << "    \"kernel_launch_overhead_us\": "
     << llvm::format(
            "%.6f", config.cyclesToMicroseconds(kernelLaunchOverheadCycles))
     << ",\n";
  os << "    \"predicted_total_cycles\": " << predictedTotalCycles << ",\n";
  os << "    \"predicted_total_time_us\": "
     << llvm::format("%.6f",
                     config.cyclesToMicroseconds(predictedTotalCycles))
     << ",\n";
  os << "    \"kernel_launch_block_dim\": " << kernelLaunchBlockDim << ",\n";
  os << "    \"kernel_launch_num_waves\": " << kernelLaunchNumWaves << ",\n";
  os << "    \"kernel_launch_model\": \"" << jsonEscape(kernelLaunchModel)
     << "\"\n";
  os << "  },\n";
  os << "  \"calibration_summary\": {\n";
  os << "    \"calibrated_ops\": " << calibratedOpCount << ",\n";
  os << "    \"heuristic_ops\": " << heuristicOpCount << ",\n";
  os << "    \"calibrated_cycles\": " << calibratedCycles << ",\n";
  os << "    \"heuristic_cycles\": " << heuristicCycles << ",\n";
  os << "    \"calibrated_weighted_cycles\": " << calibratedWeightedCycles << ",\n";
  os << "    \"heuristic_weighted_cycles\": " << heuristicWeightedCycles << ",\n";
  os << "    \"sync_issue_cycles\": " << syncIssueCycles << ",\n";
  os << "    \"sync_event_wait_cycles\": " << syncEventWaitCycles << ",\n";
  os << "    \"by_source\": ";
  emitCostStatMap(costSourceStats);
  os << ",\n";
  os << "    \"by_subpipe\": ";
  emitCostStatMap(costSubpipeStats);
  os << ",\n";
  os << "    \"top_unclassified\": ";
  emitTopCostStats(unclassifiedCostStats, 20);
  os << "\n";
  os << "  },\n";
  os << "  \"critical_path_summary\": {\n";
  os << "    \"cycles\": " << criticalPathCycles << ",\n";
  os << "    \"issue_cycles\": " << criticalPathIssueCycles << ",\n";
  os << "    \"event_wait_cycles\": " << criticalPathEventWaitCycles << ",\n";
  os << "    \"ops\": " << joinSizeVec(criticalPathOps) << "\n";
  os << "  },\n";
  os << "  \"loop_diagnostics\": {\n";
  os << "    \"total\": " << loopCount << ",\n";
  os << "    \"resolved\": " << resolvedLoopCount << ",\n";
  os << "    \"unresolved\": " << unresolvedLoopCount << ",\n";
  os << "    \"max_trip_count\": " << maxLoopTripCount << ",\n";
  os << "    \"loops\": [";
  for (size_t i = 0; i < loopDiagnostics.size(); ++i) {
    const HIVMLoopDiagnostic &loop = loopDiagnostics[i];
    if (i) os << ",";
    os << "{"
       << "\"line\":" << loop.lineNumber
       << ",\"lower\":" << loop.lowerBound
       << ",\"upper\":" << loop.upperBound
       << ",\"step\":" << loop.step
       << ",\"trip_count\":" << loop.tripCount
       << ",\"multiplier\":" << loop.multiplier
       << ",\"resolved\":" << (loop.resolved ? "true" : "false")
       << ",\"upper_bound_trip_count_estimate\":" << loop.upperBoundTripCountEstimate
       << ",\"body_first_line\":" << loop.bodyFirstLine
       << ",\"body_last_line\":" << loop.bodyLastLine
       << "}";
  }
  os << "]\n";
  os << "  },\n";
  os << "  \"operations\": [\n";
  for (size_t i = 0; i < operations.size(); ++i) {
    const HIVMOp &op = operations[i];
    if (i) os << ",\n";
    os << "    {"
       << "\"id\":" << op.id
       << ",\"name\":\"" << jsonEscape(op.opName) << "\""
       << ",\"pipe\":\"" << HIVMAnalyzer::stringifyPipe(op.pipe) << "\""
       << ",\"duration\":" << op.duration
       << ",\"issue_duration\":" << op.issueDuration
       << ",\"dependency_latency\":" << op.dependencyLatency
       << ",\"event_wait_cycles\":" << op.eventWaitCycles
       << ",\"start_cycle\":" << op.startCycle
       << ",\"resource_release_cycle\":" << op.resourceReleaseCycle
       << ",\"value_ready_cycle\":" << op.valueReadyCycle
       << ",\"end_cycle\":" << op.endCycle
       << ",\"line\":" << op.lineNumber
       << ",\"depends_on\":" << joinSizeVec(op.dependsOn)
       << ",\"event_depends_on\":" << joinSizeVec(op.eventDependsOn)
       << ",\"is_sync\":" << (op.isSyncOp ? "true" : "false")
       << ",\"is_barrier\":" << (op.isBarrier ? "true" : "false")
       << ",\"event_id\":\"" << jsonEscape(op.eventId) << "\""
       << ",\"event_generation\":" << op.eventGeneration
       << ",\"sender_pipe\":\"" << HIVMAnalyzer::stringifyPipe(op.senderPipe)
       << "\""
       << ",\"receiver_pipe\":\"" << HIVMAnalyzer::stringifyPipe(op.receiverPipe)
       << "\""
       << ",\"core_type\":\"" << jsonEscape(op.coreType) << "\""
       << ",\"bytes\":" << op.bytes
       << ",\"packet_bytes\":" << op.packetBytes
       << ",\"elements\":" << op.elements
       << ",\"flops\":" << op.flops
       << ",\"loop_multiplier\":" << op.loopMultiplier
       << ",\"multi_buffer_slots\":" << op.multiBufferSlots
       << ",\"read_buffers\":" << joinStrVec(op.readBuffers)
       << ",\"write_buffers\":" << joinStrVec(op.writeBuffers)
       << ",\"read_versions\":" << joinIntVec(op.readBufferVersions)
       << ",\"write_versions\":" << joinIntVec(op.writeBufferVersions)
       << ",\"src_space\":\"" << jsonEscape(op.srcSpace) << "\""
       << ",\"dst_space\":\"" << jsonEscape(op.dstSpace) << "\""
       << ",\"elem_type\":\"" << jsonEscape(op.elemType) << "\""
       << ",\"repeat\":" << op.repeat
       << ",\"mask\":" << op.mask
       << ",\"calibrated_cost\":" << (op.calibratedCost ? "true" : "false")
       << ",\"cost_source\":\"" << jsonEscape(op.costSource) << "\""
       << ",\"cost_subpipe\":\"" << jsonEscape(op.costSubpipe) << "\""
       << "}";
  }
  os << "\n  ]\n}\n";
}
