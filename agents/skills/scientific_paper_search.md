# Scientific Paper Search Playbook (State of the Art)

## Mission
Find the *best available primary literature* and *reproducible artifacts* for the user query, then synthesize a report that is evidence-grounded and auditable.

## Search Principles
1. Prefer primary sources: peer-reviewed papers, arXiv preprints, official datasets, benchmark pages, code repositories, and institutional reports.
2. Triangulate: do not rely on a single venue. Cross-check claims across at least 2 independent sources when feasible.
3. Time-aware: include seminal foundations plus the most recent 24-36 months of advances.
4. Evidence-first: extract concrete items (task definition, datasets, label schemas, metrics, baselines, ablations, failure modes).
5. Reproducibility: prioritize papers with released code, data, or clear experimental details.

## Query Decomposition
Convert the user task into 5–8 facets; search each facet separately.

Facet templates:
- Task: definition, label space, taxonomy alignment, ambiguity
- Data: datasets, annotation guidelines, label noise, sampling
- Methods: classical baselines, transformer baselines, weak supervision, LLM-assisted labeling
- Evaluation: metrics, splits, leakage risks, error analysis
- Domain: patents (fields, jurisdictions, language), SDGs (targets, granularity)
- Deployment: drift, monitoring, policy sensitivity, calibration
- Benchmarks: shared tasks, leaderboards, open challenges
- Related tasks: CPC/IPC classification, topic modeling, citation-based signals

## Search Operators (portable)
Use multi-pronged query styles:
- Title/phrase: `"patent" "SDG" classification`
- Year windows: `2022..2026` (or explicit years in queries)
- Dataset keywords: `dataset`, `benchmark`, `annotation`, `labeling`, `gold standard`, `silver label`
- Method keywords: `weak supervision`, `distant supervision`, `multi-label`, `hierarchical`, `retrieval`, `ontology`
- Evaluation keywords: `macro F1`, `micro F1`, `MAP`, `nDCG`, `calibration`, `error analysis`
- Artifact keywords: `GitHub`, `code`, `repository`, `reproducible`, `supplementary`

## Academic Sources: How To Use Them
### arXiv (fast preprints)
- Use shorter, focused queries (arXiv rate limits; avoid giant queries).
- Prefer topic + keyphrase queries rather than full sentences.
- Retrieve 10–25 results, then refine.

### Semantic Scholar / PubMed Central
- Use when you need structured metadata, citations, and related-work graphs.
- Good for finding “hidden” relevant work not tagged by obvious keywords.

### Web search (DDG/Searx)
- Use for datasets, benchmarks, industry whitepapers, and code repos.
- Use it to locate PDFs and supplementary materials.

## Ranking Heuristics (what to keep)
Score sources higher if they have:
- explicit dataset + labeling protocol
- clear evaluation split and metrics
- ablations and comparisons to strong baselines
- released artifacts (code/data)
- explicit limitations and failure modes

Score lower if they are:
- opinion pieces with no evidence
- blog posts without references
- marketing pages without method detail

## PDF Acquisition Targets
Prefer PDF links from:
- publisher PDF / arXiv PDF
- author’s institutional page
- proceedings PDF

When downloading PDFs, keep filenames stable and avoid duplicates.

