# lerobot-curate

**CPU-only curation and selection of [LeRobot](https://github.com/huggingface/lerobot) v3 robot datasets, before VLA fine-tuning.**

`lerobot-curate` streams a LeRobot v3 dataset from the Hugging Face Hub (no full
download), scores its episodes on a laptop CPU, and produces a smaller,
deduplicated, diverse, mislabel-checked subset — optionally materialized as a new
ready-to-train LeRobot v3 dataset. Core is **torch-free**.

> [!IMPORTANT]
> **Scope disclaimer.** lerobot-curate is a CPU reference implementation of the
> SCIZOR (embedding dedup/suboptimal) and FAKTUAL (signature-kernel diversity)
> methods. It **does NOT reproduce their reported policy-performance gains**
> (those require GPU policy training and are out of scope). Validated on
> algorithm-correctness metrics only. See [Scope & honesty](#scope--honesty).

---

## What it does (a1 capabilities)

| Capability | Method | Status |
|---|---|---|
| **Dedup + suboptimal detection** | SSL-embedding K-means + cosine-prune; model-free suboptimal proxies (embedding-jump p95, reversal ratio, action quiescence) | a1 |
| **Diversity subset selection** | Truncated path-signature kernel (depth 3) → random Fourier features → von-Neumann/Vendi diversity → budget subset (greedy/DPP) | a1 |
| **Cross-modal mislabel flagging** | task-text vs frame-image cosine; missing/degenerate descriptions reported as `not-evaluated` (never silently passed) | a1 |
| Cross-dataset coverage (optimal transport) | — | v0.1.1 roadmap |
| kNN-MI redundancy | use [`democlean`](https://github.com/dipampaul17/democlean) (a separate tool) | out of scope |

## Install

```bash
pip install lerobot-curate            # core (torch-free): dedup, diversity, mislabel, stub embedder
pip install "lerobot-curate[onnx]"    # real CPU embedder (Xenova SigLIP ONNX int8, weights fetched on demand)
pip install "lerobot-curate[video]"   # PyAV frame decode for real datasets
pip install "lerobot-curate[fiftyone]" # FiftyOne export
```

Model weights are **never bundled**; the ONNX embedder downloads its weights to
your Hugging Face cache on demand (only when you run the real embedder).

## Quickstart

```bash
# Inspect available backends honestly (which embedder/decoder are installed)
lerobot-curate doctor

# Curate a dataset down to a 200-episode budget and write a derived LeRobot v3 dataset
lerobot-curate curate <hf_repo_id> --budget 200 --push-to ./_materialized/my-subset --html report.html
```

Example output (illustrative, regenerated from `results/` at release):

```
# illustrative shape of the output (not measured numbers)
kept 200 / 1000 episodes   dropped: 612 near-duplicate, 173 low-diversity, 15 mislabel-suspected
bytes downloaded:   0 GB (streamed)
```

## Scope & honesty

- **No policy-performance claims.** The upstream papers report downstream
  improvements when a policy is trained on the selected data; reproducing those
  needs GPU policy training and is **out of scope** here. We validate only that
  the selection algorithms behave correctly (dedup recall/precision on injected
  duplicates, mislabel precision on injected label swaps, monotone diversity
  under injected redundancy, near-zero false positives on clean data).
- **Measured numbers** below are generated from
  `results/v0.1.0a1_metrics.json` — not hand-written.
- **Mislabel results degrade honestly**: episodes without usable task text are
  reported as `not-evaluated`, never as `ok`.

## Validation results

Algorithm-correctness metrics on synthetic data with injected ground truth
(deterministic, CPU; regenerate with `python scripts/run_metrics.py`). These
measure selection-algorithm correctness only — not policy performance.

<!-- METRICS:START -->
| metric | value |
|---|---|
| dedup recall (injected exact duplicates) | 1.0 |
| dedup precision | 1.0 |
| mislabel precision (injected label swaps) | 1.0 |
| mislabel recall | 1.0 |
| diversity monotone under redundancy (Spearman rho) | -0.9999999999999999 |
| clean-data false-positive rate | 0.0 |

_mode: synthetic; python 3.12.3; lerobot-curate 0.1.0a1; seed 0_
<!-- METRICS:END -->

## License

MIT. See [LICENSE](LICENSE) for third-party
component and model-weight attribution.
