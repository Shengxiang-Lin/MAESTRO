# Untrusted orchestration (future dataset workflow)

This directory is **reserved for a separate, not-yet-implemented** dataset
generator. Keep the existing trusted `environment/base_orchestration` pipeline
unchanged: it models correct feedback and gold-consistent scheduling.

Planned inputs are the trusted Scene, Macro graph, reviewed report bank and
a declarative attack scenario. The future workflow should distinguish
communication tampering, persistent compromised Sub-Agent backends, and
source/tool-result poisoning; expose only Main-visible history, not the hidden
attack label. Outputs should be per-decision Main inputs/actions, attack and
provenance metadata in a **private** sidecar, and verification/recovery labels.

Before implementing, specify a versioned attack event contract, cost
accounting, action vocabulary (accept / verify / retry / switch / reject),
train/test separation, and replay/visibility checks. The runner should
read a fourth independent YAML in `environment/configs` when implemented.
