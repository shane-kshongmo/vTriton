#!/usr/bin/env python3
"""Clean and fix .npuir.mlir for LLVM 19 compatibility.

Filters bishengir-compile stderr garbage and fixes collapse_shape strides.
"""
import re
import sys

src = sys.argv[1]
dst = sys.argv[2]

with open(src) as f:
    text = f.read()

# Step 1: Filter garbage lines.
lines = []
for line in text.splitlines(True):
    s = line.lstrip()
    if s.startswith('loc(') and 'warning' in s: continue
    if s.startswith(('ld.lld:', '[ERROR]', '[WARNING]', '[INFO]')): continue
    if s.startswith('warning') or s.endswith('warning generated.\n'): continue
    if s.strip():
        if not s.startswith(('//', 'func', '%', 'hivm', 'arith', 'memref',
                             'scf', 'return', 'module', '#', '{', '}',
                             'annotation', 'affine', 'cf', 'test', 'linalg',
                             'builtin', 'vector', 'math', 'ub',
                             'bufferization', 'tensor', 'transform')):
            continue
    lines.append(line)

fixes = {}

for i, line in enumerate(lines):
    m = re.match(r'\s*(%\w+)\s*=\s*memref\.collapse_shape', line)
    if not m: continue
    var = m.group(1)

    into_pos = line.rfind(' into ')
    if into_pos == -1: continue
    pre_into = line[:into_pos]
    src_start = pre_into.rfind(' : memref<')
    if src_start == -1: continue
    src_str = pre_into[src_start + 3:]

    # Parse dims: split by ',', take first segment, split by 'x', drop last (elem type)
    cm = re.match(r'^memref<(.+)>', src_str)
    if not cm: continue
    inner = cm.group(1)
    dims_elem = inner.split(',')[0].strip()
    tokens = dims_elem.split('x')
    if len(tokens) < 2: continue
    dim_strs = tokens[:-1]
    src_dims = [int(d) if d.strip().isdigit() else d.strip() for d in dim_strs]

    # Parse reassociation [[0, 1]]
    rb = line.find('[[')
    if rb == -1: continue
    # Bracket count: open=2 for [[, close=0 for ]]
    depth = 0; raw = None
    for j, ch in enumerate(line[rb:], rb):
        if ch == '[': depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0: raw = line[rb + 2:j]; break
    if raw is None: continue
    # Strip trailing stray ']' that may remain from single-group [[0, 1]]
    raw = raw.rstrip(']').rstrip()
    groups = []
    for g in re.split(r'\],\s*\[', raw):
        idxs = [int(p.strip()) for p in g.split(',') if p.strip().isdigit()]
        if idxs: groups.append(idxs)
    if not groups: continue

    # Compute correct strides
    new_strides = []
    for g in groups:
        ref = list(g)
        while len(ref) > 1 and src_dims[ref[-1]] == 1: ref.pop()
        ld = src_dims[ref[-1]]
        if ld == '?' and len(ref) > 1:
            new_strides.append('?')
        else:
            new_strides.append(1)

    result_part = line[into_pos + 6:]
    old_sm = re.search(r'strided<\[([?\d]+)\]', result_part)
    if not old_sm: continue
    old_val = old_sm.group(1)
    new_val = ', '.join(str(s) for s in new_strides)
    if old_val == new_val: continue

    old_strided = old_sm.group(0)
    new_strided = 'strided<[' + new_val + ']'
    lines[i] = line[:into_pos + 6] + result_part.replace(old_strided, new_strided, 1)
    fixes[var] = (old_strided, new_strided)

# Step 3: Propagate fixes to downstream SSA references.
for var, (old_s, new_s) in fixes.items():
    for k in range(len(lines)):
        if var in lines[k] and old_s in lines[k]:
            lines[k] = lines[k].replace(old_s, new_s)

with open(dst, 'w') as f:
    f.writelines(lines)

print(f"Lines: {len(lines)}")
print(f"Stride fixes: {len(fixes)}")
for var, (o, n) in sorted(fixes.items()):
    print(f"  {var}: {o} -> {n}")
