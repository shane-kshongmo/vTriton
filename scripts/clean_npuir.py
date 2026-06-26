#!/usr/bin/env python3
"""Clean and fix .npuir.mlir for LLVM 19 compatibility.

Filters bishengir-compile stderr garbage and fixes collapse_shape strides.

Parsing uses an explicit, dependency-free tokenizer (no regular expressions)
so the transform stays robust on nested MLIR type syntax and the script keeps
running inside the calibration containers, which are not guaranteed to ship a
parsing library such as lark.
"""
import sys

_MLIR_PREFIXES = (
    '//', 'func', '%', 'hivm', 'arith', 'memref', 'scf', 'return', 'module',
    '#', '{', '}', 'annotation', 'affine', 'cf', 'test', 'linalg', 'builtin',
    'vector', 'math', 'ub', 'bufferization', 'tensor', 'transform',
)


def is_garbage(stripped, raw):
    """Return True for bishengir-compile stderr noise that must be dropped."""
    if stripped.startswith('loc(') and 'warning' in stripped:
        return True
    if stripped.startswith(('ld.lld:', '[ERROR]', '[WARNING]', '[INFO]')):
        return True
    if stripped.startswith('warning') or raw.endswith('warning generated.\n'):
        return True
    return False


def filter_lines(text):
    """Keep blank lines and MLIR-looking lines; drop compiler noise."""
    out = []
    for raw in text.splitlines(True):
        s = raw.lstrip()
        if is_garbage(s, raw):
            continue
        if s.strip() and not s.startswith(_MLIR_PREFIXES):
            continue
        out.append(raw)
    return out


def parse_collapse_shape(line):
    """Parse a ``memref.collapse_shape`` line into its structural parts.

    Returns ``(var, src_dims, groups)`` or ``None`` when the line is not a
    collapse_shape op.  ``src_dims`` are the source memref leading dims (ints,
    or ``'?'`` for dynamic); ``groups`` is the reassociation index list.
    """
    stripped = line.lstrip()
    if not stripped.startswith('%'):
        return None
    # var = %<word>
    j = 1
    while j < len(stripped) and (stripped[j].isalnum() or stripped[j] == '_'):
        j += 1
    var = stripped[:j]
    if j == 1:
        return None
    rest = stripped[j:].lstrip()
    if not rest.startswith('='):
        return None
    rest = rest[1:].lstrip()
    if not rest.startswith('memref.collapse_shape'):
        return None

    into_pos = line.rfind(' into ')
    if into_pos == -1:
        return None
    pre_into = line[:into_pos]
    src_start = pre_into.rfind(' : memref<')
    if src_start == -1:
        return None
    src_str = pre_into[src_start + 3:]  # skip ' : ' -> 'memref<...'

    # Source dims: take the text from 'memref<' up to the final '>' (matching
    # the previous greedy behaviour), then the first comma-segment, split on x.
    if not src_str.startswith('memref<'):
        return None
    gt = src_str.rfind('>')
    if gt <= len('memref<'):
        return None
    inner = src_str[len('memref<'):gt]
    dims_elem = inner.split(',')[0].strip()
    tokens = dims_elem.split('x')
    if len(tokens) < 2:
        return None
    dim_strs = tokens[:-1]  # drop element type
    src_dims = [int(d) if d.strip().isdigit() else d.strip() for d in dim_strs]

    # Reassociation [[...], [...]] — bracket-depth aware.
    rb = line.find('[[')
    if rb == -1:
        return None
    depth = 0
    raw = None
    for k in range(rb, len(line)):
        ch = line[k]
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                raw = line[rb + 2:k]
                break
    if raw is None:
        return None
    raw = raw.rstrip(']').rstrip()
    groups = []
    for g in raw.split('], ['):
        idxs = [int(p.strip()) for p in g.split(',') if p.strip().isdigit()]
        if idxs:
            groups.append(idxs)
    if not groups:
        return None
    return var, src_dims, groups


def find_single_strided(result_part):
    """Find the first ``strided<[N]>`` whose content is a single ?/int token.

    Mirrors the previous ``strided<\\[([?\\d]+)\\]`` match: only single-token
    leading-stride lists are rewritten.  Returns ``(old_val, old_strided)`` or
    ``None``.
    """
    key = 'strided<['
    start = 0
    while True:
        p = result_part.find(key, start)
        if p == -1:
            return None
        content_start = p + len(key)
        rb = result_part.find(']', content_start)
        if rb != -1:
            content = result_part[content_start:rb]
            if content and all(c.isdigit() or c == '?' for c in content):
                return content, key + content + ']'
        start = p + 1


def compute_new_strides(src_dims, groups):
    """Leading stride per reassociation group (1, or '?' for dynamic)."""
    new_strides = []
    for g in groups:
        ref = list(g)
        while len(ref) > 1 and src_dims[ref[-1]] == 1:
            ref.pop()
        ld = src_dims[ref[-1]]
        if ld == '?' and len(ref) > 1:
            new_strides.append('?')
        else:
            new_strides.append(1)
    return new_strides


def fix_strides(lines):
    """In-place collapse_shape stride fix; returns {var: (old, new)}."""
    fixes = {}
    for i, line in enumerate(lines):
        parsed = parse_collapse_shape(line)
        if parsed is None:
            continue
        var, src_dims, groups = parsed
        new_strides = compute_new_strides(src_dims, groups)

        into_pos = line.rfind(' into ')
        result_part = line[into_pos + 6:]
        found = find_single_strided(result_part)
        if found is None:
            continue
        old_val, old_strided = found
        new_val = ', '.join(str(s) for s in new_strides)
        if old_val == new_val:
            continue
        new_strided = 'strided<[' + new_val + ']'
        lines[i] = line[:into_pos + 6] + result_part.replace(
            old_strided, new_strided, 1)
        fixes[var] = (old_strided, new_strided)

    # Propagate fixes to downstream SSA references.
    for var, (old_s, new_s) in fixes.items():
        for k in range(len(lines)):
            if var in lines[k] and old_s in lines[k]:
                lines[k] = lines[k].replace(old_s, new_s)
    return fixes


def main(argv):
    src, dst = argv[1], argv[2]
    with open(src) as f:
        text = f.read()

    lines = filter_lines(text)
    fixes = fix_strides(lines)

    with open(dst, 'w') as f:
        f.writelines(lines)

    print(f"Lines: {len(lines)}")
    print(f"Stride fixes: {len(fixes)}")
    for var, (o, n) in sorted(fixes.items()):
        print(f"  {var}: {o} -> {n}")


if __name__ == '__main__':
    main(sys.argv)
