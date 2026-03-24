# Add IR/ subdirectory to TableGen include path so .td files can find siblings
set(LLVM_TABLEGEN_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/IR")
 
# TableGen for interfaces (must come before ops since ops depend on interfaces)
set(LLVM_TARGET_DEFINITIONS IR/AscendModelInterfaces.td)
mlir_tablegen(IR/AscendModelInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(IR/AscendModelInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(AscendModelInterfacesIncGen)
 
# TableGen for dialect and ops
set(LLVM_TARGET_DEFINITIONS IR/AscendModelOps.td)
mlir_tablegen(IR/AscendModelOps.h.inc -gen-op-decls)
mlir_tablegen(IR/AscendModelOps.cpp.inc -gen-op-defs)
mlir_tablegen(IR/AscendModelOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(IR/AscendModelOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(IR/AscendModelDialect.h.inc -gen-dialect-decls)
mlir_tablegen(IR/AscendModelDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(IR/AscendModelOpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(IR/AscendModelOpsAttrDefs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(IR/AscendModelOpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen(IR/AscendModelOpsTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(AscendModelOpsIncGen)
add_dependencies(AscendModelOpsIncGen AscendModelInterfacesIncGen)

# TableGen for passes
set(LLVM_TARGET_DEFINITIONS Transforms/Passes.td)
mlir_tablegen(Transforms/Passes.h.inc -gen-pass-decls -name AscendModel)
add_public_tablegen_target(AscendModelTransformsPassIncGen)
 
# Reset flags
set(LLVM_TABLEGEN_FLAGS "")
