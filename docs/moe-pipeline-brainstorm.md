# Building an MoE pipeline for ExecuTorch's Qualcomm (QNN/HTP) backend

Status: brainstorm / design notes, 2026-10-17. Nothing here is implemented yet.

## The core problem

Every operator MoE routing needs already lowers to the NPU. The issue is **efficiency at real
model size**, because of how ExecuTorch expresses MoE today.

Evidence from ExecuTorch v1.5.1 and QAIRT 2.37:
- QNN 2.37 `QnnOpDef.h` has `TopK`, `Gather`, `GatherElements`, `GatherNd`, `ScatterElements`,
  `ScatterNd`, `OneHot`, `MatMul`, `FullyConnected`, `ElementWiseSelect`, `NonZero`, `CumulativeSum`.
- ExecuTorch has QNN builders `op_topk`, `op_gather`, `op_index`, `op_index_select`,
  `op_scatter_elements`, `op_bmm`, `op_matmul`, `op_where`, `op_sort`.
- `backends/qualcomm/tests/test_qnn_delegate.py::test_qnn_backend_moe_feed_forward` lowers a
  **toy** `MOEFeedForward` (dim = 32).
- No Qualcomm example models use MoE. ExecuTorch's MoE models (`qwen3_5_moe`, `gemma4_31b`,
  `muse_glimmer`) target CUDA and MLX/Metal.

The problem line is in `examples/models/llama/llama_transformer.py` (`ConditionalFeedForward`):

```python
w1_weights = self.w1[expert_indices].transpose(-1, -2)  # gather WEIGHTS at runtime
```

HTP is fast because weights are **static constants** it can pre-pack and keep in INT4/INT8. A
runtime gather makes the matmul activation × activation instead, so that fast path is lost.

## Where it breaks at scale

| # | Problem | Why | Difficulty |
|---|---|---|---|
| 1 | Runtime weight selection | loses static INT4 weight packing | hard (core) |
| 2 | Quantization through gather | per-expert scales vs a shared encoding on the stacked tensor | hard, needs experiments |
| 3 | Memory | all experts must be resident; a 26B-A4B model is about 15 GB at 4-bit, against 4 GB per HTP context | medium |
| 4 | Prefill routing | variable tokens per expert vs static shapes | medium; fix: Switch-Transformer-style capacity padding |
| 5 | Calibration | rare experts get few calibration samples | medium |
| 6 | Runner orchestration | many graphs, KV cache, per-token routing | medium (engineering) |
| 7 | Vision encoder | dense, already solved (InternVL3 / SmolVLM path) | easy |

## Design options

**A. Dense compute + mask.** Run every expert and zero out the unselected ones. Static and simple,
but it reads every expert for every token, so it runs like the dense model. Useful only as a
correctness baseline.

**B. In-graph gather (today's toy).** All on the NPU, but problems 1 and 2 apply in full.
Measure it, but it's risky.

**C. Hybrid NPU + CPU.** Attention, router and shared expert on the NPU; routed experts on CPU INT4
kernels (v1.5 added a quantized MoE FFN op with INT4 packing). Uses ExecuTorch's existing partial
delegation. About 2 CPU↔NPU handoffs per layer per token. **Most realistic near term.**

**D. Host-orchestrated expert graphs.** The NPU computes the router, the CPU reads the top-k IDs and
calls precompiled per-expert graphs (static INT4 weights; v73 supports weight sharing). Keeps the
fast path, fully on the NPU. Cost: about layers × k graph calls per token, plus many graphs to
package. **Best end state if per-call overhead is small.**

## Phased plan and effort

Estimates assume one engineer experienced with ExecuTorch and QNN. Rough.

| Phase | Goal | Effort | Risk |
|---|---|---|---|
| 0 | Micro-benchmarks: per-graph-execute overhead; runtime-gather vs static-weight expert matmul; addressable-memory ceiling across contexts | 1–2 weeks | low; decides C vs D |
| 1 | Small MoE text end-to-end via option C (e.g. OLMoE-1B-7B, Granite-3.0-1B-A400M) with accuracy eval | 3–6 weeks | medium |
| 2 | Fully-NPU option D: expert graphs, runner routing, capacity-padded prefill | +1–3 months | high (SDK limits) |
| 3 | Large VLM MoE (Gemma 4 26B-A4B class): vision encoder, calibration, memory tuning | +1–2 months | high |

To check first: whether newer QAIRT releases (ExecuTorch main pins 2.50) add MoE or
dynamic-weight support, which would make phase 2 much easier.
