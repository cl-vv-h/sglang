# NPU megakernel providers

Enable an installed provider with `--enable-npu-megakernel`. The host loads the
entry point for the model's architecture after loading weights, allocating KV
and initializing attention. For example, an external distribution can declare

```toml
[project.entry-points."sglang.srt.npu_megakernels"]
GlmMoeDsaForCausalLM = "my_provider.glm:TARGET"
GlmMoeDsaForCausalLMNextN = "my_provider.glm:NEXTN"
```

Each entry point is a `MegakernelSpec` with role `target` or `nextn`. Its
`prepare` inspects the processed model, allocated KV pool and resolved
configuration. It returns a binding or a `Decline` describing a refusal.
SGLang compares the prepare results across TP ranks before calling `build`.
Preparation must complete without TP collectives.

A binding can supply a backend for the existing NPU decode graph runner.
A backend that declares `uses_capture_batch` receives the runner's capture batch.
The provider can capture an independent BlockRT session alongside the native
fallback graph. Replay uses an existing matching session and takes a TP-wide
route decision before model execution begins. Execution failures propagate to
the caller.

A `nextn` binding supplies an EAGLE draft runtime for draft decode. Before
native draft execution the runtime receives the current batch and exact graph
key, or a missing key for a rank ineligible for graph replay. A successful
replay returns the seed and recurrent tokens in an owned tensor. The worker
uses the standard EAGLE proposal and verify path. Prefill and draft extend use
native execution. The Ascend provider enables fused draft decode when
`SGLANG_NPU_FUSE_DRAFT_DECODE=1`. Its default is `0`.
