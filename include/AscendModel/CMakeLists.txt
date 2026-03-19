# TableGen for dialect
set(LLVM_TARGET_DEFINITIONS IR/AscendModelOps.td)
mlir_tablegen(IR/AscendModelOps.h.inc -gen-op-decls)
mlir_tablegen(IR/AscendModelOps.cpp.inc -gen-op-defs)
mlir_tablegen(IR/AscendModelDialect.h.inc -gen-dialect-decls)
mlir_tablegen(IR/AscendModelDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(IR/AscendModelTypes.h.inc -gen-typedef-decls)
mlir_tablegen(IR/AscendModelTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(AscendModelOpsIncGen)

# TableGen for interfaces
set(LLVM_TARGET_DEFINITIONS IR/AscendModelInterfaces.td)
mlir_tablegen(IR/AscendModelInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(IR/AscendModelInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(AscendModelInterfacesIncGen)

# TableGen for passes
set(LLVM_TARGET_DEFINITIONS Transforms/Passes.td)
mlir_tablegen(Transforms/Passes.h.inc -gen-pass-decls -name AscendModel)
add_public_tablegen_target(AscendModelPassIncGen)
