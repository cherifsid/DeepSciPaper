# DeepSciPaper

![DeepSciPaper Banner](asset/Gemini_Generated_Image_vof9oxvof9oxvof9.png)

DeepSciPaper is a production-oriented Research Copilot for evidence-first scientific workflows. It combines:

- Agentic deep search via a local `gpt-researcher` engine (autonomous crawling and report generation into your workspace)
- Vectorless retrieval via VectifyAI PageIndex (PDFs are indexed into semantic section trees, no chunking + no vector DB)
- A Streamlit workspace UI for discovery, indexing, grounded Q\&A, and LaTeX authoring/compilation
- Dual LLM backends: Cloud (OpenAI/DeepSeek) and Local (Ollama)

The UI is workspace-agnostic: you can rename the workspace title and point the app at any folder paths directly from the interface.

## What You Get

- `Deep Search & PageIndex Curation`: run GPT Researcher, stage PDFs, (re)index with PageIndex
- `Vectorless Tree-Reasoning RAG Chat`: answers grounded in PageIndex trees + page-range evidence with strict citations
- `Live LaTeX Studio & Local Compiling`: AI-assisted LaTeX edits plus local `pdflatex` compilation

## Repo Layout

- `app.py`: monolithic Streamlit app
- `agents/skills/`: agent skill packs (Deep Search + RAG Chat quality playbooks)
- `engines/gpt-researcher/`: GPT Researcher engine (vendored as a local engine)
- `engines/PageIndex/`: PageIndex engine (vendored as a local engine)
- `bib_pdf/`: runtime PDF staging (ignored in git)
- `pageindex_cache/`: runtime PageIndex cache (ignored in git)

## Prerequisites

- Conda (or mamba)
- Python 3.11
- Ollama (optional, for local models)
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

4) Run the app:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open:

- http://localhost:8501

## Usage Notes

- Add PDFs by dropping them into `bib_pdf/` (or use Deep Search to acquire them).
- Use `Re-Index Workspace Base via PageIndex` to build semantic section trees into `pageindex_cache/`.
- The RAG chat is vectorless: it reasons over the PageIndex tree, then cites evidence with:
  `[filename :: node_id :: pp. start-end :: title]`.

## Security

- Do not commit `.env` (this repo ignores it by default).
- If you accidentally committed secrets earlier, rotate them and rewrite git history before making the repo public.

