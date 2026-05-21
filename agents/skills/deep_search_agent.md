# Deep Search Agent Operating Procedure (GPT Researcher)

## Operating Rules
1. Do not assume a single retriever is reliable; expect rate limits and partial failures.
2. If a retriever returns empty results or errors, continue with other sources rather than failing the task.
3. Prefer fewer, higher-signal results over many low-signal results.
4. Keep outputs grounded: cite sources for claims; mark speculation explicitly.
5. Use only retrieved URLs in the report. If a link is not present in the evidence, do not guess it.

## Query Strategy
Generate queries in tiers:
1. Broad survey query (captures the field overview).
2. Dataset/benchmark query (identifies labeled data and evaluation setups).
3. Method query (SOTA, weak supervision, LLM labeling, retrieval alignment).
4. Failure mode query (limitations, bias, domain shift).
5. Reproducibility query (code, datasets, licenses, implementations).

For each tier, produce 1–2 concise queries, not long sentences.
For arXiv-style scholarly APIs, use compact Boolean/keyword queries such as `"patent classification" AND SDG`, never a full paragraph prompt.

## Evidence Extraction Checklist (per key paper)
Extract these fields when possible:
- citation (title, authors, year, venue)
- task framing and label space (multi-label? hierarchical?)
- dataset description and how labels were produced
- train/val/test splits, leakage risks, and metrics
- baseline comparisons and ablations
- key results (with numbers if present)
- limitations and known failure modes
- released artifacts (code/data) and license constraints

## Report Output Contract
Write the report as:
1. Executive summary (5–10 bullets, evidence-grounded)
2. Evidence map (table-like list: paper -> contribution -> dataset -> metric)
3. Methods landscape (grouped by approach family)
4. Datasets/benchmarks (what exists, what’s missing)
5. Evaluation pitfalls (label noise, leakage, temporal drift)
6. Open problems and next experiments (actionable)

Always include a “Sources” section listing all URLs used.
