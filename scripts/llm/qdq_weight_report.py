"""What did 8-bit quantization do to Phi-4-mini? Reads the exported QDQ graph (decode_qdq.pt2) and reports:

1. Weights. The graph stores them already quantized (int8, symmetric, one scale per output channel). For each
   weight it computes the SQNR of the 8-bit copy against Microsoft's original bf16 weights (exact), and the
   outlier ratio max|w| / rms per output channel, which is what decides the SQNR:
       step = max|w| / 127   ->   SQNR ~= 10*log10(12 * 127^2) - 20*log10(max/rms) = 52.9 dB - 20*log10(max/rms)
2. Activations (16-bit, one scale per tensor): the range each quantized activation must cover, and the step size.

  . C:\\et\\et-env.ps1
  python qdq_weight_report.py C:\\et\\llm\\phi4mini_w8a16_2k\\decode_qdq.pt2 <HF snapshot dir of Phi-4-mini-instruct>
"""
import csv
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.ao.quantization.fx._decomposed  # noqa: F401  (registers torch.ops.quantized_decomposed)
from safetensors import safe_open

DQPC = torch.ops.quantized_decomposed.dequantize_per_channel.default
QPT = torch.ops.quantized_decomposed.quantize_per_tensor.default
KINDS = ["wq", "wk", "wv", "wo", "w1 (gate)", "w3 (up)", "w2 (down)"]   # order inside each layer in the graph


def hf_weights(snapshot: Path) -> dict:
    out = {}
    for f in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            for k in st.keys():
                out[k] = (f, k)
    return out


def load(hf, key):
    f, k = hf[key]
    with safe_open(str(f), "pt") as st:
        return st.get_tensor(k).to(torch.float32)


def original(hf, layer, kind, q_dim=3072, kv_dim=1024, hidden=8192):
    p = f"model.layers.{layer}."
    if kind in ("wq", "wk", "wv"):
        qkv = load(hf, p + "self_attn.qkv_proj.weight")
        return {"wq": qkv[:q_dim], "wk": qkv[q_dim:q_dim + kv_dim], "wv": qkv[q_dim + kv_dim:]}[kind]
    if kind == "wo":
        return load(hf, p + "self_attn.o_proj.weight")
    if kind.startswith("w1") or kind.startswith("w3"):
        gu = load(hf, p + "mlp.gate_up_proj.weight")
        return gu[:hidden] if kind.startswith("w1") else gu[hidden:]
    return load(hf, p + "mlp.down_proj.weight")


def main(qdq_path: str, snapshot: str):
    t0 = time.time()
    gm = torch.export.load(qdq_path).module()
    print(f"loaded the QDQ graph in {time.time() - t0:.0f} s", flush=True)

    def attr(t):
        x = gm
        for a in t.split("."):
            x = getattr(x, a)
        return x

    def val(a):
        return attr(a.target) if isinstance(a, torch.fx.Node) else a

    hf = hf_weights(Path(snapshot))
    weights, acts = [], []
    for n in gm.graph.nodes:
        if n.target is DQPC and n.args[0].op == "get_attr":
            weights.append((attr(n.args[0].target), val(n.args[1]), n.args[4], n.args[5]))
        elif n.target is QPT:
            scale, qmin, qmax = float(val(n.args[1])), n.args[3], n.args[4]
            acts.append((n.args[0].name, scale, qmin, qmax))

    rows = []
    for i, (q, scale, qmin, qmax) in enumerate(weights):
        layer, kind = divmod(i, 7)
        if layer >= 32:
            continue                                                      # output head: reported separately
        kind = KINDS[kind]
        w_hat = q.to(torch.float32).flatten(1) * scale.reshape(-1, 1).to(torch.float32)
        w = original(hf, layer, kind)
        assert w.shape == w_hat.shape, (layer, kind, w.shape, w_hat.shape)
        err = w - w_hat
        sqnr = 10 * math.log10(float((w * w).sum()) / float((err * err).sum()))
        rms = w.pow(2).mean(dim=1).sqrt()
        ratio = w.abs().amax(dim=1) / (rms + 1e-12)
        rows.append({"layer": layer, "weight": kind, "shape": "x".join(map(str, w.shape)), "sqnr_db": round(sqnr, 2),
                     "median_outlier_ratio": round(float(ratio.median()), 1),
                     "worst_outlier_ratio": round(float(ratio.max()), 1),
                     "worst_channel_sqnr_db": round(52.9 - 20 * math.log10(float(ratio.max())), 1)})
        if layer % 8 == 0 and kind == "wq":
            print(f"  ...layer {layer}", flush=True)

    out = Path(qdq_path).with_name("weight_sqnr.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    vals = sorted(r["sqnr_db"] for r in rows)
    print(f"\nWEIGHTS: {len(rows)} quantized to int8 (symmetric, per output channel), compared with the bf16 original")
    print(f"  SQNR median {statistics.median(vals):.1f} dB, min {vals[0]:.1f} dB, max {vals[-1]:.1f} dB "
          f"(ideal 8-bit with no outliers: ~52.9 dB; each 6 dB ~ 1 bit)")
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["weight"]].append(r)
    print("\n  by weight type:   median SQNR   min SQNR   median outlier ratio   worst outlier ratio")
    for k in KINDS:
        rs = by_kind[k]
        print(f"  {k:12s} {statistics.median(r['sqnr_db'] for r in rs):10.1f} {min(r['sqnr_db'] for r in rs):10.1f}"
              f" {statistics.median(r['median_outlier_ratio'] for r in rs):18.1f} {max(r['worst_outlier_ratio'] for r in rs):20.1f}")
    print("\n  by layer (min SQNR over its 7 weights):")
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r["sqnr_db"])
    print("  " + "  ".join(f"L{k}:{min(v):.1f}" for k, v in sorted(by_layer.items())))
    print("\n  worst 10 weights:")
    for r in sorted(rows, key=lambda r: r["sqnr_db"])[:10]:
        print(f"    {r['sqnr_db']:5.1f} dB  layer {r['layer']:2d} {r['weight']:10s} worst channel outlier ratio "
              f"{r['worst_outlier_ratio']} (that channel ~{r['worst_channel_sqnr_db']} dB)")

    print(f"\nACTIVATIONS: {len(acts)} quantize points, 16-bit, one scale per tensor")
    ranges = sorted(((s * (qmax - qmin), s, name) for name, s, qmin, qmax in acts), reverse=True)
    rs = [r for r, _, _ in ranges]
    print(f"  range covered: median {statistics.median(rs):.3g}, max {rs[0]:.4g}")
    print("  widest 10 (range, step size, node):")
    for r, s, name in ranges[:10]:
        print(f"    {r:12.4g}  step {s:.3g}  {name}")
    print(f"\nCSV: {out}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
