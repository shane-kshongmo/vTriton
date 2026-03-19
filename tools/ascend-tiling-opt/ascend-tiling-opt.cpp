//===----------------------------------------------------------------------===//
// ascend-tiling-opt.cpp - Tiling Optimization Tool for Ascend NPU
//
// Finds optimal tiling configurations using the unified cost model.
// Reuses existing modeling code from lib/AscendModel/Analysis.
//
// Usage:
//   ascend-tiling-opt --op=matmul --shape=1024,1024,512
//   ascend-tiling-opt --op=flash-attention --shape=1,32,2048,128 --pareto
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/HardwareConfig.h"
#include "AscendModel/Analysis/UnifiedTilingCostModel.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ToolOutputFile.h"

#include <sstream>
#include <algorithm>

using namespace llvm;
using namespace mlir::ascend;

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static cl::OptionCategory OptCat("Tiling Optimization Options");

static cl::opt<std::string> opType("op",
    cl::desc("Operation: matmul, bmm, softmax, flash-attention"),
    cl::Required, cl::cat(OptCat));

static cl::opt<std::string> shape("shape",
    cl::desc("Shape: M,N,K for matmul; B,M,N,K for bmm; B,H,S,D for flash-attn"),
    cl::Required, cl::cat(OptCat));

static cl::opt<std::string> precision("precision",
    cl::desc("Precision: fp16 (default), fp32, bf16, int8"),
    cl::init("fp16"), cl::cat(OptCat));

static cl::opt<std::string> hwConfigFile("hw-config",
    cl::desc("Hardware config JSON file"), cl::init(""), cl::cat(OptCat));

static cl::opt<std::string> objective("objective",
    cl::desc("Objective: min-cycles (default), min-memory, balanced"),
    cl::init("min-cycles"), cl::cat(OptCat));

static cl::opt<int64_t> minTile("min-tile", cl::desc("Min tile"), cl::init(16));
static cl::opt<int64_t> maxTile("max-tile", cl::desc("Max tile"), cl::init(256));
static cl::opt<int64_t> tileStep("tile-step", cl::desc("Step"), cl::init(16));

static cl::opt<std::string> outputFile("o", cl::desc("Output file"), cl::init("-"));
static cl::opt<std::string> outputFmt("format", cl::desc("text/json/csv"), cl::init("text"));
static cl::opt<bool> showPareto("pareto", cl::desc("Show Pareto frontier"));
static cl::opt<bool> showBreakdown("breakdown", cl::desc("Show cost breakdown"));
static cl::opt<bool> verbose("v", cl::desc("Verbose output"));

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

std::vector<int64_t> parseShape(const std::string &s) {
  std::vector<int64_t> r;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (!tok.empty()) r.push_back(std::stoll(tok));
  return r;
}

int64_t elemBytes(const std::string &p) {
  if (p == "fp32") return 4;
  if (p == "int8") return 1;
  return 2;
}

TilingCostFunction::Objective parseObj(const std::string &o) {
  if (o == "min-memory") return TilingCostFunction::Objective::MinMemoryAccess;
  if (o == "balanced") return TilingCostFunction::Objective::Balanced;
  return TilingCostFunction::Objective::MinCycles;
}

//===----------------------------------------------------------------------===//
// Operation Builders (reuse OpDesc from UnifiedTilingCostModel.h)
//===----------------------------------------------------------------------===//

OpDesc makeMatmul(int64_t M, int64_t N, int64_t K, int64_t eb) {
  OpDesc op;
  op.name = "matmul"; op.type = OpType::Matmul;
  op.hwUnit = HWUnit::Cube; op.flopsPerOutputElement = 2 * K;
  
  op.dims['m'] = {'m', M, M, false, false};
  op.dims['n'] = {'n', N, N, false, false};
  op.dims['k'] = {'k', K, K, true, false};
  
  TensorDesc A{"A", {'m','k'}, {{'m',M},{'k',K}}, eb, {'n'}};
  TensorDesc B{"B", {'k','n'}, {{'k',K},{'n',N}}, eb, {'m'}};
  op.inputs = {A, B};
  op.output = {"C", {'m','n'}, {{'m',M},{'n',N}}, eb, {}};
  return op;
}

OpDesc makeBMM(int64_t B, int64_t M, int64_t N, int64_t K, int64_t eb) {
  OpDesc op;
  op.name = "bmm"; op.type = OpType::BatchedMatmul;
  op.hwUnit = HWUnit::Cube; op.flopsPerOutputElement = 2 * K;
  
  op.dims['b'] = {'b', B, B, false, true};
  op.dims['m'] = {'m', M, M, false, false};
  op.dims['n'] = {'n', N, N, false, false};
  op.dims['k'] = {'k', K, K, true, false};
  
  TensorDesc A{"A", {'b','m','k'}, {{'b',B},{'m',M},{'k',K}}, eb, {'n'}};
  TensorDesc tB{"B", {'b','k','n'}, {{'b',B},{'k',K},{'n',N}}, eb, {'m'}};
  op.inputs = {A, tB};
  op.output = {"C", {'b','m','n'}, {{'b',B},{'m',M},{'n',N}}, eb, {}};
  return op;
}

OpDesc makeSoftmax(int64_t M, int64_t N, int64_t eb) {
  OpDesc op;
  op.name = "softmax"; op.type = OpType::Softmax;
  op.hwUnit = HWUnit::Vector; op.flopsPerOutputElement = 5;
  
  op.dims['m'] = {'m', M, M, false, false};
  op.dims['n'] = {'n', N, N, false, false};
  
  TensorDesc in{"in", {'m','n'}, {{'m',M},{'n',N}}, eb, {}};
  op.inputs = {in};
  op.output = {"out", {'m','n'}, {{'m',M},{'n',N}}, eb, {}};
  return op;
}

std::vector<OpDesc> makeFlashAttn(int64_t B, int64_t H, int64_t S, int64_t D, int64_t eb) {
  std::vector<OpDesc> ops;
  int64_t batch = B * H;
  
  // QK matmul
  OpDesc qk;
  qk.name = "qk"; qk.type = OpType::BatchedMatmul;
  qk.hwUnit = HWUnit::Cube; qk.flopsPerOutputElement = 2 * D;
  qk.dims['b'] = {'b', batch, batch, false, true};
  qk.dims['m'] = {'m', S, S, false, false};
  qk.dims['n'] = {'n', S, S, false, false};
  qk.dims['k'] = {'k', D, D, true, false};
  TensorDesc Q{"Q", {'b','m','k'}, {{'b',batch},{'m',S},{'k',D}}, eb, {'n'}};
  TensorDesc K{"K", {'b','k','n'}, {{'b',batch},{'k',D},{'n',S}}, eb, {'m'}};
  qk.inputs = {Q, K};
  qk.output = {"QK", {'b','m','n'}, {{'b',batch},{'m',S},{'n',S}}, eb, {}};
  qk.consumerOps = {"softmax"};
  ops.push_back(qk);
  
  // Softmax  
  OpDesc sm;
  sm.name = "softmax"; sm.type = OpType::Softmax;
  sm.hwUnit = HWUnit::Vector; sm.flopsPerOutputElement = 5;
  sm.dims['b'] = {'b', batch, batch, false, true};
  sm.dims['m'] = {'m', S, S, false, false};
  sm.dims['n'] = {'n', S, S, false, false};
  TensorDesc QKin{"QK", {'b','m','n'}, {{'b',batch},{'m',S},{'n',S}}, eb, {}};
  sm.inputs = {QKin};
  sm.output = {"P", {'b','m','n'}, {{'b',batch},{'m',S},{'n',S}}, eb, {}};
  sm.producerOps = {"qk"}; sm.consumerOps = {"pv"};
  ops.push_back(sm);
  
  // PV matmul
  OpDesc pv;
  pv.name = "pv"; pv.type = OpType::BatchedMatmul;
  pv.hwUnit = HWUnit::Cube; pv.flopsPerOutputElement = 2 * S;
  pv.dims['b'] = {'b', batch, batch, false, true};
  pv.dims['m'] = {'m', S, S, false, false};
  pv.dims['n'] = {'n', D, D, false, false};
  pv.dims['k'] = {'k', S, S, true, false};
  TensorDesc P{"P", {'b','m','k'}, {{'b',batch},{'m',S},{'k',S}}, eb, {'n'}};
  TensorDesc V{"V", {'b','k','n'}, {{'b',batch},{'k',S},{'n',D}}, eb, {'m'}};
  pv.inputs = {P, V};
  pv.output = {"O", {'b','m','n'}, {{'b',batch},{'m',S},{'n',D}}, eb, {}};
  pv.producerOps = {"softmax"};
  ops.push_back(pv);
  
  return ops;
}

//===----------------------------------------------------------------------===//
// Output Formatters
//===----------------------------------------------------------------------===//

void printText(raw_ostream &os, const UnifiedTilingConfig &cfg,
               const CostBreakdown &cost, const std::vector<OpDesc> &ops) {
  os << "\n";
  os << "╔═══════════════════════════════════════════════════════════════╗\n";
  os << "║           Ascend Tiling Optimization Results                  ║\n";
  os << "╚═══════════════════════════════════════════════════════════════╝\n\n";
  
  if (!ops.empty()) {
    os << "Operation: " << ops[0].name;
    if (ops.size() > 1) os << " (+" << (ops.size()-1) << " fused)";
    os << "\nShape:     ";
    for (const auto &[l, d] : ops[0].dims) os << l << "=" << d.size << " ";
    os << "\n\n";
  }
  
  os << "┌───────────────────────────────────────────────────────────────┐\n";
  os << "│ Optimal Tiling                                                │\n";
  os << "├───────────────────────────────────────────────────────────────┤\n";
  for (const auto &[d, s] : cfg.tileSizes) {
    int64_t numTiles = 1;
    if (!ops.empty()) {
      auto it = ops[0].dims.find(d);
      if (it != ops[0].dims.end()) numTiles = (it->second.size + s - 1) / s;
    }
    os << format("│   tile_%c = %-6ld  (%ld tiles)\n", d, s, numTiles);
  }
  os << "└───────────────────────────────────────────────────────────────┘\n\n";
  
  os << "┌───────────────────────────────────────────────────────────────┐\n";
  os << "│ Performance Metrics                                          │\n";
  os << "├───────────────────────────────────────────────────────────────┤\n";
  os << format("│   Total Cycles:          %-12ld\n", cost.totalCycles);
  os << format("│   Memory Accesses:       %.2f MB\n", cost.totalMemoryAccesses / 1e6);
  os << format("│   Operational Intensity: %.2f FLOPs/byte\n", cost.operationalIntensity);
  os << format("│   Pipeline Efficiency:   %.1f%%\n", cost.pipelineEfficiency * 100);
  os << format("│   Hardware Utilization:  %.1f%%\n", cost.hardwareUtilization * 100);
  os << "└───────────────────────────────────────────────────────────────┘\n";
  
  if (showBreakdown) {
    os << "\n┌───────────────────────────────────────────────────────────────┐\n";
    os << "│ Detailed Breakdown                                            │\n";
    os << "├───────────────────────────────────────────────────────────────┤\n";
    os << format("│   Cube cycles:     %-12ld\n", cost.cubeCycles);
    os << format("│   Vector cycles:   %-12ld\n", cost.vectorCycles);
    os << format("│   HBM cycles:      %-12ld\n", cost.hbmAccessCycles);
    os << format("│   CV balance:      %.3f\n", cost.cvBalanceRatio);
    os << format("│   Serial cycles:   %-12ld\n", cost.serialCycles);
    os << format("│   Pipelined:       %-12ld\n", cost.pipelinedCycles);
    os << "└───────────────────────────────────────────────────────────────┘\n";
  }
}

void printPareto(raw_ostream &os,
                 const std::vector<std::pair<UnifiedTilingConfig, CostBreakdown>> &pts) {
  os << "\n┌───────────────────────────────────────────────────────────────┐\n";
  os << "│ Pareto Frontier (Memory vs Cycles)                           │\n";
  os << "├───────────────────────────────────────────────────────────────┤\n";
  os << "│  #   Cycles(K)   Memory(MB)    OI      Tiling                │\n";
  os << "├───────────────────────────────────────────────────────────────┤\n";
  
  auto sorted = pts;
  std::sort(sorted.begin(), sorted.end(),
    [](const auto &a, const auto &b) { return a.second.totalCycles < b.second.totalCycles; });
  
  int i = 1;
  for (const auto &[c, cost] : sorted) {
    std::string t;
    for (const auto &[d, s] : c.tileSizes) t += std::string(1,d) + "=" + std::to_string(s) + " ";
    os << format("│ %2d  %9.1f   %9.2f   %6.1f  %-18s│\n", i++,
                 cost.totalCycles/1000.0, cost.totalMemoryAccesses/1e6,
                 cost.operationalIntensity, t.c_str());
    if (i > 10) { os << "│ ... (" << (sorted.size()-10) << " more)\n"; break; }
  }
  os << "└───────────────────────────────────────────────────────────────┘\n";
}

void printJSON(raw_ostream &os, const UnifiedTilingConfig &cfg,
               const CostBreakdown &cost, const std::vector<OpDesc> &ops,
               const std::vector<std::pair<UnifiedTilingConfig, CostBreakdown>> *pts) {
  json::Object root;
  
  if (!ops.empty()) {
    json::Object opJ;
    opJ["name"] = ops[0].name;
    opJ["fused_count"] = static_cast<int64_t>(ops.size());
    json::Object shJ;
    for (const auto &[l,d] : ops[0].dims) shJ[std::string(1,l)] = d.size;
    opJ["shape"] = std::move(shJ);
    root["operation"] = std::move(opJ);
  }
  
  json::Object tJ;
  for (const auto &[d,s] : cfg.tileSizes) tJ["tile_" + std::string(1,d)] = s;
  root["optimal_tiling"] = std::move(tJ);
  
  json::Object mJ;
  mJ["total_cycles"] = cost.totalCycles;
  mJ["memory_bytes"] = cost.totalMemoryAccesses;
  mJ["operational_intensity"] = cost.operationalIntensity;
  mJ["pipeline_efficiency"] = cost.pipelineEfficiency;
  mJ["hardware_utilization"] = cost.hardwareUtilization;
  mJ["cube_cycles"] = cost.cubeCycles;
  mJ["vector_cycles"] = cost.vectorCycles;
  mJ["cv_balance"] = cost.cvBalanceRatio;
  root["metrics"] = std::move(mJ);
  
  if (pts && !pts->empty()) {
    json::Array arr;
    for (const auto &[c, cst] : *pts) {
      json::Object p;
      json::Object tc;
      for (const auto &[d,s] : c.tileSizes) tc["tile_" + std::string(1,d)] = s;
      p["tiling"] = std::move(tc);
      p["cycles"] = cst.totalCycles;
      p["memory_bytes"] = cst.totalMemoryAccesses;
      p["oi"] = cst.operationalIntensity;
      arr.push_back(std::move(p));
    }
    root["pareto_frontier"] = std::move(arr);
  }
  
  os << json::Value(std::move(root)) << "\n";
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::HideUnrelatedOptions(OptCat);
  cl::ParseCommandLineOptions(argc, argv,
      "Ascend Tiling Optimizer - Find optimal tiling configurations\n\n"
      "Examples:\n"
      "  ascend-tiling-opt --op=matmul --shape=1024,1024,512\n"
      "  ascend-tiling-opt --op=bmm --shape=32,128,128,64 --objective=min-memory\n"
      "  ascend-tiling-opt --op=flash-attention --shape=1,32,2048,128 --pareto\n"
      "  ascend-tiling-opt --op=matmul --shape=4096,4096,4096 --format=json\n");
  
  // Load hardware config
  HardwareConfig hwConfig;
  if (!hwConfigFile.empty()) {
    if (!hwConfig.loadFromFile(hwConfigFile)) {
      errs() << "Error: Failed to load hw config from " << hwConfigFile << "\n";
      return 1;
    }
  }
  
  // Parse shape
  auto dims = parseShape(shape);
  int64_t eb = elemBytes(precision);
  
  // Build operation(s)
  std::vector<OpDesc> ops;
  
  if (opType == "matmul") {
    if (dims.size() != 3) { errs() << "matmul needs M,N,K\n"; return 1; }
    ops.push_back(makeMatmul(dims[0], dims[1], dims[2], eb));
  } else if (opType == "bmm") {
    if (dims.size() != 4) { errs() << "bmm needs B,M,N,K\n"; return 1; }
    ops.push_back(makeBMM(dims[0], dims[1], dims[2], dims[3], eb));
  } else if (opType == "softmax") {
    if (dims.size() != 2) { errs() << "softmax needs M,N\n"; return 1; }
    ops.push_back(makeSoftmax(dims[0], dims[1], eb));
  } else if (opType == "flash-attention") {
    if (dims.size() != 4) { errs() << "flash-attention needs B,H,S,D\n"; return 1; }
    ops = makeFlashAttn(dims[0], dims[1], dims[2], dims[3], eb);
  } else {
    errs() << "Unknown op: " << opType << "\n";
    return 1;
  }
  
  // Run optimization
  UnifiedTilingOptimizer optimizer(hwConfig);
  optimizer.setDefaultRange(minTile, maxTile, tileStep);
  optimizer.setVerbose(verbose);
  
  auto result = (ops.size() == 1) ?
    optimizer.optimize(ops[0], parseObj(objective)) :
    optimizer.optimizeFused(ops, parseObj(objective));
  
  // Output
  raw_ostream *os = &outs();
  std::unique_ptr<ToolOutputFile> outF;
  if (outputFile != "-") {
    std::error_code EC;
    outF = std::make_unique<ToolOutputFile>(outputFile, EC, sys::fs::OF_Text);
    if (EC) { errs() << "Cannot open " << outputFile << "\n"; return 1; }
    os = &outF->os();
  }
  
  if (outputFmt == "json") {
    printJSON(*os, result.bestConfig, result.bestCost, ops,
              showPareto ? &result.paretoFrontier : nullptr);
  } else if (outputFmt == "csv") {
    *os << "tile_m,tile_n,tile_k,cycles,memory_bytes,oi\n";
    for (const auto &[c, cst] : result.paretoFrontier) {
      *os << c.getTile('m',0) << "," << c.getTile('n',0) << ","
          << c.getTile('k',0) << "," << cst.totalCycles << ","
          << cst.totalMemoryAccesses << "," << cst.operationalIntensity << "\n";
    }
  } else {
    printText(*os, result.bestConfig, result.bestCost, ops);
    if (showPareto && !result.paretoFrontier.empty())
      printPareto(*os, result.paretoFrontier);
  }
  
  if (verbose)
    errs() << "Explored " << result.configurationsExplored << " configurations\n";
  
  if (outF) outF->keep();
  return 0;
}
