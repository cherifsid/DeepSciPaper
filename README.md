# DeepSciPaper

![DeepSciPaper Banner](asset/Gemini_Generated_Image_vof9oxvof9oxvof9.png)

DeepSciPaper is a production-oriented Research Copilot for evidence-first scientific workflows. It combines:

- Agentic deep search via a local `gpt-researcher` engine (autonomous crawling and report generation into your workspace)
- Agentic multimodal RAG via MinerU, Ollama enrichment, and local Qdrant evidence retrieval
- A Streamlit workspace UI for discovery, indexing, grounded Q\&A, and LaTeX authoring/compilation
- Dual LLM backends: Cloud (OpenAI/DeepSeek) and Local (Ollama)

The UI is workspace-agnostic: you can rename the workspace title and point the app at any folder paths directly from the interface.

## System Overview

DeepSciPaper now uses a fully local-first multimodal retrieval architecture for indexed workspace reasoning.

The flow is split into two cooperating systems:

- `Deep Search`: `gpt-researcher` explores the web, repositories, and scholarly sources, then writes reports and discovered PDFs into `bib_pdf/`.
- `Agentic Multimodal RAG`: staged PDFs are parsed with MinerU, transformed into multimodal artifacts, indexed into local Qdrant, and then synthesized by the active chat model selected in the UI.

The result is a workspace where discovery and retrieval are separate on purpose:

- discovery is optimized for breadth, source acquisition, and report generation
- retrieval is optimized for grounded answers over local evidence you can inspect
- generation is decoupled from indexing, so you can switch between Ollama and cloud models at chat time without rebuilding the index

## What You Get

- `Deep Search & Multimodal Indexing`: run GPT Researcher, stage PDFs, parse with MinerU, enrich artifacts, and index into local Qdrant
- `Agentic Multimodal RAG Chat`: answers grounded in retrieved text, tables, figures, captions, and artifact paths
- `Live LaTeX Studio & Local Compiling`: AI-assisted LaTeX edits plus local `pdflatex` compilation

## How The RAG Chat Works

The chat system is agentic, multimodal, and evidence-first. It does not answer from raw PDFs directly. Instead, it answers from an indexed evidence store built from your workspace documents.

### Retrieval Pipeline

1. PDFs are staged in `bib_pdf/`.
2. `multimodal_pipeline.py ingest` parses each PDF with MinerU.
3. MinerU outputs structured Markdown, JSON, and extracted visual assets.
4. The pipeline separates the document into:
   - text sections and section chunks
   - tables
   - images and figures
5. Text chunks are prepared for retrieval.
6. Tables are summarized with the configured `--text-model`.
7. Figures and images are captioned with the configured `--image-model`.
8. Every retrieval record is embedded with the fixed model `sentence-transformers/all-mpnet-base-v2`.
9. Embedded records are stored in local Qdrant with pointers back to the original artifact paths.

### Query-Time Pipeline

1. The user submits a question in the `Agentic Multimodal RAG Chat` tab.
2. The app embeds the question with the same fixed mpnet embedding model.
3. Qdrant returns the top evidence records across text, table, and image modalities.
4. The app builds an evidence bundle containing:
   - retrieval score
   - modality
   - source PDF name
   - section title when available
   - artifact path
   - display text, summary, or caption
5. The synthesizer skill in `agents/skills/multimodal_synthesizer_chat.md` instructs the active LLM to:
   - answer only from retrieved evidence
   - cite claims with evidence IDs such as `[E1]`
   - preserve uncertainty when evidence is weak or conflicting
   - surface table and image artifacts when relevant
6. The final answer is rendered in Markdown in the chat UI, with expandable artifact previews.

### Generation Model Routing

The retrieval index is model-stable, but the answer-generation model is runtime-switchable:

- `Local Ollama`: the answer is generated with the selected Ollama model and the current UI hyperparameters
- `Cloud API`: the answer is generated with the selected OpenAI or DeepSeek-compatible model and the current UI hyperparameters

This means indexing and answer generation are intentionally separated:

- indexing uses MinerU + enrichment models + fixed mpnet embeddings
- chat generation uses whichever inference backend the user chooses at runtime

## RAG Graph

```text
                           +----------------------+
                           |   User Deep Search   |
                           |   Query / Web Task   |
                           +----------+-----------+
                                      |
                                      v
                      +----------------------------------+
                      |   GPT Researcher Deep Search     |
                      |   reports + discovered PDFs      |
                      +----------------+-----------------+
                                       |
                                       v
                               +---------------+
                               |   bib_pdf/    |
                               | staged PDFs   |
                               +-------+-------+
                                       |
                                       v
                    +----------------------------------------+
                    | multimodal_pipeline.py ingest          |
                    | MinerU parse + multimodal enrichment   |
                    +----------------+-----------------------+
                                     |
          +--------------------------+---------------------------+
          |                          |                           |
          v                          v                           v
+------------------+      +--------------------+      +--------------------+
| Text sections    |      | Table extraction   |      | Figure / image     |
| + markdown chunks|      | + LLM summaries    |      | + vision captions   |
+---------+--------+      +----------+---------+      +----------+---------+
          |                          |                           |
          +--------------------------+---------------------------+
                                     |
                                     v
                 +-----------------------------------------------+
                 | fixed embeddings: all-mpnet-base-v2           |
                 | records keep source PDF + section + path      |
                 +----------------------+------------------------+
                                        |
                                        v
                             +----------------------+
                             |  Local Qdrant Index  |
                             | multimodal evidence  |
                             +----------+-----------+
                                        ^
                                        |
                             user question embedded
                                        |
                                        v
                     +------------------------------------------+
                     | Agentic Multimodal RAG Chat in app.py    |
                     | retrieve top evidence bundle             |
                     +------------------+-----------------------+
                                        |
                                        v
                 +------------------------------------------------------+
                 | Active synthesis model selected in UI                |
                 | Ollama or Cloud API + current inference parameters   |
                 +------------------+-----------------------------------+
                                    |
                                    v
                      +----------------------------------+
                      | Markdown answer with evidence    |
                      | IDs and artifact references      |
                      +----------------------------------+
```

## Repo Layout

- `app.py`: monolithic Streamlit app
- `agents/skills/`: agent skill packs (Deep Search + RAG Chat quality playbooks)
- `engines/gpt-researcher/`: GPT Researcher engine (vendored as a local engine)
- `bib_pdf/`: runtime PDF staging (ignored in git)
- `multimodal_pipeline.py`: batch/CLI pipeline for MinerU parsing, multimodal enrichment, Qdrant indexing, and local agentic RAG
- `multimodal_store/`: runtime multimodal document store and Qdrant local index (ignored in git)

## Core Design Choices

- No PageIndex remains in the active architecture. The retrieval system is now fully based on MinerU parsing, multimodal enrichment, and Qdrant-backed evidence retrieval.
- Embeddings are fixed to `sentence-transformers/all-mpnet-base-v2` to keep index dimensionality stable and avoid Ollama embedding failures.
- Summaries and captions are contextualized with source-paper and section metadata before indexing.
- The synthesizer is guided by a dedicated agent skill so the final answer stays grounded, citation-aware, and artifact-aware.
- The answer model is swappable at runtime without forcing re-indexing.

## Prerequisites

- Conda (or mamba)
- Python 3.11
- Ollama (optional, for local models)
- MinerU (optional but recommended, for structured multimodal PDF parsing)
- TeX Live / `pdflatex` (optional, for local LaTeX compilation)

## Setup (Recommended)

1) Create and activate the environment:

```bash
conda create -n research_copilot python=3.11 -y
conda activate research_copilot
```

2) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3) Create your `.env`:

```bash
cp .env.example .env
```

Fill in at least one of:

- `TAVILY_API_KEY` (recommended for better web search)
- `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` (if you use Cloud API backend)

If you use local models:

- Ensure Ollama is running at `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- Pull a model, for example:

```bash
ollama pull gpt-oss:20b
```

For image/graph captioning, pull a vision-capable Ollama model and set `MULTIMODAL_IMAGE_MODEL` in `.env`:

```bash
ollama pull llama3.2-vision:11b
```

4) Run the app:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open:

- http://localhost:8501

## Usage Notes

- Add PDFs by dropping them into `bib_pdf/` (or use Deep Search to acquire them).
- Use `Index Workspace With MinerU + Multimodal RAG` to parse staged PDFs, extract artifacts, caption figures, summarize tables, embed records, and index into local Qdrant.
- The RAG chat retrieves from Qdrant and answers in Markdown with evidence IDs plus image/table artifact paths where available.
- The embedding model for multimodal retrieval is always `sentence-transformers/all-mpnet-base-v2`, even if you change text or vision generation models.
- The text model and image model used during ingest affect table summaries and figure captions, while the chat model selected in the UI affects only answer generation.

## Multimodal CLI

Clean a previous local multimodal test store:

```bash
python multimodal_pipeline.py clean --store ./multimodal_store
```

Batch ingest staged PDFs with MinerU, Ollama table summaries, Ollama image captions, and local Qdrant indexing:

```bash
python multimodal_pipeline.py ingest \
  --pdf-dir ./bib_pdf \
  --store ./multimodal_store \
  --mineru-backend pipeline \
  --mineru-method auto \
  --mineru-lang en \
  --mineru-timeout 1800 \
  --text-model gpt-oss:20b \
  --image-model llama3.2-vision:11b \
  --embed-model sentence-transformers/all-mpnet-base-v2 \
  --enrich-images \
  --force
```

Ingest without image captions for a faster first pass:

```bash
python multimodal_pipeline.py ingest --pdf-dir ./bib_pdf --store ./multimodal_store --mineru-backend pipeline
```

Ask the local multimodal index:

```bash
python multimodal_pipeline.py query "Which papers provide benchmark evidence for SDG classification?" --format markdown --limit 10
```

## Security

- Do not commit `.env` (this repo ignores it by default).
- If you accidentally committed secrets earlier, rotate them and rewrite git history before making the repo public.
