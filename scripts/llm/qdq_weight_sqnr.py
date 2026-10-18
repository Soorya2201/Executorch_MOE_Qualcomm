"""Where does 8-bit weight quantization hurt Phi-4-mini? Read the exported QDQ graph (decode_qdq.pt2) and, for
every weight that goes through quantize->dequantize, measure how much signal survives the rounding:

    SQNR (dB) = 10 * log10( ||w||^2 / ||w - dq(q(w))||^2 )     higher is better; every 6 dB ~ 1 bit

Writes a CSV of every weight (worst first) and prints a summary by layer and by weight type, so a recipe can
give only the weak layers 16-bit weights. Needs ~25 GB RAM (the graph is loaded once); no forward pass.

  . C:\\et\\et-env.ps1;  python C:\\et\\llm\\qdq_weight_sqnr.py C:\\et\\llm\\phi4mini_w8a16_2k\\decode_qdq.pt2
"""
import csv
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

QPC = torch.ops.quantized_decomposed.quantize_per_channel.default
DQPC = torch.ops.quantized_decomposed.dequantize_per_channel.default
QPT = torch.ops.quantized_decomposed.quantize_per_tensor.default


def attr(gm, target):
    x = gm
    for a in target.split("."):
        x = getattr(x, a)
    return x


def arg(gm, a):
    return attr(gm, a.target) if isinstance(a, torch.fx.Node) and a.op == "get_attr" else a


def fake_quant(w, scale, zp, axis, qmin, qmax):
    shape = [1] * w.dim()
    shape[axis] = -1
    s = scale.reshape(shape).to(torch.float32)
    z = zp.reshape(shape).to(torch.float32)
    q = torch.clamp(torch.round(w / s) + z, qmin, qmax)
    return (q - z) * s


def main(path: str):
    t0 = time.time()
    gm = torch.export.load(path).module()
    print(f"loaded {path} in {time.time() - t0:.0f} s", flush=True)
    rows, folded = [], 0
    for n in gm.graph.nodes:
        if n.target is QPC and n.args[0].op == "get_attr":
            w = attr(gm, n.args[0].target).detach().to(torch.float32)
            scale, zp = arg(gm, n.args[1]), arg(gm, n.args[2])
            axis, qmin, qmax = n.args[3], n.args[4], n.args[5]
            err = w - fake_quant(w, scale, zp, axis, qmin, qmax)
            sig, noise = float((w * w).sum()), float((err * err).sum())
            sqnr = 10 * math.log10(sig / noise) if noise > 0 else float("inf")
            # how outlier-heavy is each output channel: max |w| / rms over the reduced dims
            red = [d for d in range(w.dim()) if d != axis]
            ratio = float((w.abs().amax(dim=red) / (w.pow(2).mean(dim=red).sqrt() + 1e-12)).max())
            rows.append({"weight": n.args[0].target, "shape": "x".join(map(str, w.shape)), "bits": int(math.log2(qmax - qmin + 1)),
                         "sqnr_db": round(sqnr, 2), "worst_channel_max_over_rms": round(ratio, 1)})
        elif n.target is DQPC and n.args[0].op == "get_attr" and attr(gm, n.args[0].target).dtype in (torch.int8, torch.uint8):
            folded += 1
    if not rows:
        print(f"No quantize_per_channel on float weights found ({folded} folded int8 weights). The graph stores "
              "already-quantized weights, so the error needs the original checkpoint for comparison.")
        return
    rows.sort(key=lambda r: r["sqnr_db"])
    out = Path(path).with_name("weight_sqnr.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    vals = [r["sqnr_db"] for r in rows]
    med = statistics.median(vals)
    print(f"\n{len(rows)} quantized weights; SQNR median {med:.1f} dB, min {vals[0]:.1f} dB, max {vals[-1]:.1f} dB")
    by_type, by_layer = defaultdict(list), defaultdict(list)
    for r in rows:
        m = re.search(r"layers[._](\d+)[._].*?(w[a-z0-9]+|wq|wk|wv|wo)_?conv", r["weight"])
        kind = re.sub(r"layers[._]\d+[._]", "", r["weight"])
        by_type[re.sub(r"\.weight$|_weight$", "", kind)].append(r["sqnr_db"])
        if m:
            by_layer[int(m.group(1))].append(r["sqnr_db"])
    print("\nby weight type (median dB):")
    for k, v in sorted(by_type.items(), key=lambda kv: statistics.median(kv[1])):
        print(f"  {k:55s} {statistics.median(v):6.1f}  (min {min(v):.1f}, n={len(v)})")
    print("\nby layer (min dB over its weights):")
    for k in sorted(by_layer):
        print(f"  layer {k:2d}: {min(by_layer[k]):6.1f}")
    print("\nworst 15 weights:")
    for r in rows[:15]:
        print(f"  {r['sqnr_db']:6.1f} dB  {r['weight']}  ({r['shape']}, outlier ratio {r['worst_channel_max_over_rms']})")
    weak = [r for r in rows if r["sqnr_db"] < med - 6]
    print(f"\n{len(weak)} weights are more than 6 dB (~1 bit) below the median -> candidates for 16-bit weights.")
    print(f"CSV: {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1])
