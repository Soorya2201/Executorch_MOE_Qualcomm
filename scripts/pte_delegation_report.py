"""Count what a .pte executes: delegate calls (QNN/NPU) vs. CPU kernel calls, per method."""
import collections
import sys

from executorch.exir._serialize._program import deserialize_pte_binary

with open(sys.argv[1], "rb") as f:
    program = deserialize_pte_binary(f.read()).program

for plan in program.execution_plan:
    delegates = [d.id for d in plan.delegates]
    ops = [f"{o.name}.{o.overload}" if o.overload else o.name for o in plan.operators]
    counts = collections.Counter()
    for chain in plan.chains:
        for ins in chain.instructions:
            kind = type(ins.instr_args).__name__
            if kind == "KernelCall":
                counts["CPU kernel: " + ops[ins.instr_args.op_index]] += 1
            elif kind == "DelegateCall":
                counts["NPU delegate call (" + delegates[ins.instr_args.delegate_index] + ")"] += 1
            else:
                counts["runtime bookkeeping: " + kind] += 1
    print(f"== method '{plan.name}': {sum(counts.values())} instructions")
    for k, v in counts.most_common():
        print(f"   {v:4d}  {k}")
