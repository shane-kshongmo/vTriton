from scripts.clean_npuir import extract_unique_function_blocks


def test_extract_unique_function_blocks_deduplicates_failure_dumps():
    text = """\
// IR Dump After GraphSyncSolver
func.func @mix_aic(%arg0: i32) attributes {hacc.entry} {
  %c0 = arith.constant 0 : i32
  scf.for %i = %c0 to %arg0 step %arg0 : i32 {
    hivm.hir.pipe_barrier[<PIPE_ALL>]
  }
  return
}
// IR Dump After GraphSyncSolver
func.func @mix_aiv(%arg0: i32) attributes {hacc.entry} {
  return
}
"func.func"() <{sym_name = "outlined_noise"}> ({
  %noise = "arith.constant"() <{value = 1 : i32}> : () -> i32
}) : () -> ()
// repeated compiler diagnostic
func.func @mix_aic(%arg0: i32) attributes {hacc.entry} {
  return
}
func.func @mix_aiv(%arg0: i32) attributes {hacc.entry} {
  return
}
"""

    lines = extract_unique_function_blocks(text)
    cleaned = "".join(lines)

    assert cleaned.count("func.func @mix_aic") == 1
    assert cleaned.count("func.func @mix_aiv") == 1
    assert "outlined_noise" not in cleaned
    assert "scf.for" in cleaned
