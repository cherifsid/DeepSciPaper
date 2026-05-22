from __future__ import annotations

import asyncio
import hashlib
import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_LOGGER = logging.getLogger("research_copilot.pageindex")
if not PAGEINDEX_LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    PAGEINDEX_LOGGER.addHandler(_handler)
PAGEINDEX_LOGGER.setLevel(logging.INFO)
PAGEINDEX_LOGGER.propagate = False

ROOT = Path(__file__).resolve().parent
GPT_RESEARCHER_ENGINE_PATH = (ROOT / os.getenv("GPT_RESEARCHER_ENGINE_PATH", "./engines/gpt-researcher")).resolve()
PAGEINDEX_ENGINE_PATH = (ROOT / os.getenv("PAGEINDEX_ENGINE_PATH", "./engines/PageIndex")).resolve()
AGENT_SKILLS_DIR = (ROOT / os.getenv("AGENT_SKILLS_DIR", "./agents/skills")).resolve()


def activate_local_engine_paths() -> None:
    engine_specs = [
        (PAGEINDEX_ENGINE_PATH, "pageindex"),
        (GPT_RESEARCHER_ENGINE_PATH, "gpt_researcher"),
    ]
    for engine_path, package_dir in engine_specs:
        if (engine_path / package_dir).is_dir():
            engine_string = str(engine_path)
            if engine_string not in sys.path:
                sys.path.insert(0, engine_string)


activate_local_engine_paths()

DEFAULT_TITLE = os.getenv("WORKSPACE_TITLE", "Patent Classification into SDGs")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:72b-instruct")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "bge-m3:567m")
DEFAULT_CLOUD_MODEL = os.getenv("DEFAULT_CLOUD_MODEL", "gpt-4o")
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEFAULT_DEEPSEEK_MODEL", "deepseek-chat")
GOOGLE_RETRIEVERS = {"google", "searchapi", "serper", "serpapi"}


def load_agent_skill_pack(skills_dir: Path, max_chars: int = 32000) -> str:
    if not skills_dir.exists():
        return ""
    parts: list[str] = []
    used = 0
    for path in sorted(skills_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not text:
            continue
        header = f"\n\n# Skill File: {path.name}\n"
        block = header + text
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 400:
                parts.append(block[:remaining] + "\n\n[Skill pack truncated to configured limit.]")
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()


def load_skill_file(skills_dir: Path, file_name: str, max_chars: int = 16000) -> str:
    path = skills_dir / file_name
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""
    if not text:
        return ""
    return text[:max_chars]


def list_saved_reports(bib_pdf_dir: Path) -> list[Path]:
    if not bib_pdf_dir.exists():
        return []
    reports = [p for p in bib_pdf_dir.glob("*.md") if p.is_file()]
    reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return reports


def read_text_file(path: Path, limit_chars: int = 250_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(text) > limit_chars:
        return text[:limit_chars] + "\n\n[Report truncated for display.]"
    return text


class StreamlitResearchLogHandler:
    def __init__(self, progress_box: Any, max_steps: int = 12):
        self._progress_box = progress_box
        self._max_steps = max_steps
        self._steps: list[dict[str, str]] = []
        self._lock = Lock()

    def _complete_running(self) -> None:
        for step in self._steps:
            if step["state"] == "running":
                step["state"] = "done"

    def _stage(self, label: str, detail: str = "") -> None:
        detail = re.sub(r"\s+", " ", detail or "").strip()
        if len(detail) > 180:
            detail = detail[:177].rstrip() + "..."
        with self._lock:
            if self._steps and self._steps[-1]["label"] == label and self._steps[-1]["state"] == "running":
                if detail:
                    self._steps[-1]["detail"] = detail
            else:
                self._complete_running()
                self._steps.append({"label": label, "detail": detail, "state": "running"})
                if len(self._steps) > self._max_steps:
                    self._steps = self._steps[-self._max_steps :]
        self._render()

    def _finish(self, label: str = "Research report ready", detail: str = "") -> None:
        detail = re.sub(r"\s+", " ", detail or "").strip()
        with self._lock:
            self._complete_running()
            if label:
                self._steps.append({"label": label, "detail": detail, "state": "done"})
                if len(self._steps) > self._max_steps:
                    self._steps = self._steps[-self._max_steps :]
        self._render()

    def _fail(self, detail: str = "") -> None:
        with self._lock:
            self._complete_running()
            self._steps.append({"label": "Deep Search stopped", "detail": detail, "state": "error"})
        self._render()

    def _render(self) -> None:
        with self._lock:
            steps = list(self._steps)

        rows = []
        for step in steps:
            label = html.escape(step["label"])
            detail = html.escape(step.get("detail", ""))
            state = step["state"]
            indicator = '<span class="rc-spinner"></span>' if state == "running" else '<span class="rc-check">OK</span>'
            if state == "error":
                indicator = '<span class="rc-error">!</span>'
            rows.append(
                f'<div class="rc-step rc-{state}">'
                f'<div class="rc-indicator">{indicator}</div>'
                f'<div class="rc-copy">'
                f'<div class="rc-label">{label}</div>'
                f'<div class="rc-detail">{detail}</div>'
                f'</div>'
                f'</div>'
            )

        if not rows:
            return
        payload = '<div class="rc-progress">' + "\n".join(rows) + "</div>"
        try:
            if hasattr(self._progress_box, "html"):
                self._progress_box.html(payload)
            else:
                self._progress_box.markdown(payload, unsafe_allow_html=True)
        except Exception:
            pass

    async def on_tool_start(self, tool_name: str, **kwargs) -> None:
        self._stage("Using research tool", tool_name)

    async def on_agent_action(self, action: str, **kwargs) -> None:
        if action == "choose_agent":
            self._stage("Selecting research profile", "Choosing the most relevant academic agent.")
        elif action == "agent_selected":
            details = kwargs.get("details") or {}
            agent = details.get("agent", "")
            self._stage("Research profile ready", agent)
        else:
            self._stage("Research action", action)

    async def on_research_step(self, step: str, details: dict) -> None:
        details = details or {}
        if step == "start":
            query = details.get("query", "")
            self._stage("Starting research task", query[:180])
        elif step == "deep_research_initialize":
            breadth = details.get("breadth", "?")
            depth = details.get("depth", "?")
            concurrency = details.get("concurrency", "?")
            self._stage("Preparing deep research", f"Breadth {breadth}, depth {depth}, concurrency {concurrency}.")
        elif step == "deep_research_start":
            self._stage("Launching deep research", "Generating focused academic search paths.")
        elif step == "deep_research_complete":
            visited = details.get("visited_urls", 0)
            context_length = details.get("context_length", 0)
            self._stage("Deep research complete", f"Collected {visited} URLs and {context_length} context items.")
        elif step == "deep_research_costs":
            self._stage("Research cost checked", f"Total cost: {details.get('total_costs', 0.0)}")
        elif step == "cost_update":
            self._stage("Usage updated", f"Total cost: {details.get('total_cost', 0.0)}")
        elif step == "writing_report":
            self._stage("Writing final report", "Synthesizing findings, evidence, and source references.")
        elif step == "report_completed":
            report_length = details.get("report_length", 0)
            self._stage("Report drafted", f"{report_length} characters generated.")
        elif step == "validating_sources":
            self._stage("Validating source links", f"Checking {details.get('candidate_urls', 0)} discovered URLs.")
        elif step == "sources_validated":
            self._stage(
                "Source links verified",
                f"{details.get('verified_urls', 0)} of {details.get('candidate_urls', 0)} discovered URLs are reachable.",
            )
        elif step == "downloading_pdfs":
            self._stage("Downloading verified PDFs", f"{details.get('pdf_urls', 0)} validated PDF links selected.")
        elif step == "planning_search_strategy":
            self._stage("Building research plan", "Generating editable search directions before source collection.")
        elif step == "agent_selection":
            self._stage("Planning search strategy", "Preparing source priorities and search facets.")
        elif step == "conducting_research":
            self._stage("Searching sources", "Running the selected retriever and collecting candidate evidence.")
        elif step == "research_completed":
            self._stage("Evidence collected", "Research context is ready for synthesis.")
        elif step == "planning_images":
            self._stage("Checking visual assets", "Preparing optional report visuals.")
        elif step == "images_pre_generated":
            self._stage("Visual assets prepared")
        else:
            self._stage(step.replace("_", " ").title(), "Working through the next research stage.")

    def from_log_message(self, msg: str) -> None:
        clean = re.sub(r"\s+", " ", msg).strip()
        if not clean:
            return
        lower = clean.lower()
        if "starting the research task" in lower:
            self._stage("Starting research task", clean[:180])
        elif "browsing the web" in lower:
            self._stage("Browsing sources", clean[:180])
        elif "planning the research strategy" in lower:
            self._stage("Planning search strategy", clean[:180])
        elif "i will conduct my research" in lower:
            self._stage("Focused queries generated", clean[:180])
        elif "running research for" in lower:
            self._stage("Running focused sub-query", clean[:180])
        elif "scraping content" in lower:
            self._stage("Reading candidate sources", clean[:180])
        elif "getting relevant content" in lower:
            self._stage("Extracting relevant evidence", clean[:180])
        elif "scraping complete" in lower:
            self._stage("Source reading complete", clean[:180])
        elif "generating" in lower and "report" in lower:
            self._stage("Writing final report", clean[:180])
        elif "research complete" in lower or "research completed" in lower:
            self._finish("Research report ready")


class StreamlitLoggingCaptureHandler(logging.Handler):
    def __init__(self, target: StreamlitResearchLogHandler):
        super().__init__(level=logging.INFO)
        self._target = target

    def emit(self, record: logging.LogRecord) -> None:
        try:
            name = record.name or ""
            # Keep signal high: capture gpt-researcher + the dedicated 'research' logger.
            if not (name.startswith("gpt_researcher") or name == "research"):
                return
            msg = record.getMessage()
            if not msg:
                return
            # Avoid echoing raw JSON event dumps.
            if msg.startswith("research:") or msg.startswith("action:") or msg.startswith("tool:"):
                return
            self._target.from_log_message(msg)
        except Exception:
            return


@dataclass(frozen=True)
class RuntimePaths:
    bib_pdf: Path
    pageindex_cache: Path
    compiled_output: Path
    paper_tex: Path


def resolve_workspace_path(raw_path: str) -> Path:
    path = Path(raw_path.strip() or ".").expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_slug(value: str, fallback: str = "item") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug[:120] or fallback


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{int(time.time() * 1000)}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def latest_pageindex_log_path(pdf_name: str) -> str:
    logs_dir = ROOT / "logs"
    if not logs_dir.exists():
        return ""
    candidates = sorted(logs_dir.glob(f"{pdf_name}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else ""


LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: str) -> str:
    return "".join(LATEX_ESCAPE.get(char, char) for char in value)


def latex_template(workspace_title: str) -> str:
    title = latex_escape(workspace_title)
    title_for_text = workspace_title.replace("``", '"').replace("''", '"')
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}

\title{{{title}}}
\author{{Research Copilot Workspace}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This living manuscript captures a reproducible research synthesis for the workspace titled ``{latex_escape(title_for_text)}.'' The document is designed to be edited continuously as new literature is discovered, structurally indexed, and cross-examined through the local Research Copilot environment.
\end{{abstract}}

\section{{Research Aim}}
The project investigates methods, evidence, and evaluation protocols for {latex_escape(workspace_title.lower())}. The manuscript should preserve traceable claims, clearly separate empirical findings from interpretation, and cite supporting sources as the workspace evidence base grows.

\section{{Evidence Base}}
The evidence base is maintained outside this manuscript in the workspace directories. Primary literature PDFs are staged in \texttt{{bib\_pdf/}}, semantic PageIndex trees are cached in \texttt{{pageindex\_cache/}}, and compiled artifacts are written to \texttt{{compiled\_output/}}.

\section{{Methodological Notes}}
The intended retrieval path is vectorless. Documents are transformed into semantic section trees with stable node identifiers and page ranges. The co-authoring workflow should use those structural references to ground synthesis, comparisons, and claims.

\section{{Draft Synthesis}}
This section is the active synthesis area. As the workspace accumulates indexed primary literature, revise claims with section-level and page-level evidence references.

\section{{Open Research Questions}}
\begin{{enumerate}}[leftmargin=*]
    \item Which document fields provide the most reliable signal for the classification task?
    \item Which methods produce robust labels under sparse supervision?
    \item What evidence exists for cross-domain, cross-jurisdictional, or multilingual generalization?
    \item How should false positives be handled when labels inform high-impact decisions?
\end{{enumerate}}

\section{{Conclusion}}
This document is ready for iterative AI-assisted editing and local compilation.

\end{{document}}
"""


def ensure_bootstrap_files(paths: RuntimePaths, workspace_title: str) -> bool:
    paths.bib_pdf.mkdir(parents=True, exist_ok=True)
    paths.pageindex_cache.mkdir(parents=True, exist_ok=True)
    paths.compiled_output.mkdir(parents=True, exist_ok=True)

    created_paper = False
    if not paths.paper_tex.exists():
        atomic_write_text(paths.paper_tex, latex_template(workspace_title))
        created_paper = True
    return created_paper


def sync_latex_session(paper_path: Path) -> None:
    active_path = str(paper_path)
    if st.session_state.get("active_paper_path") != active_path or "latex_editor" not in st.session_state:
        paper_text = paper_path.read_text(encoding="utf-8") if paper_path.exists() else ""
        st.session_state["active_paper_path"] = active_path
        st.session_state["latex_editor"] = paper_text
        st.session_state["latex_saved_sha"] = sha256_text(paper_text)


def run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


def strip_provider_prefix(model: str, provider: str) -> str:
    clean = model.strip()
    if clean.startswith(f"{provider}/"):
        return clean.split("/", 1)[1]
    if clean.startswith(f"{provider}:"):
        return clean.split(":", 1)[1]
    return clean


def sanitize_retriever(raw_retriever: str) -> str:
    requested = [item.strip() for item in raw_retriever.split(",") if item.strip()]
    allowed = [item for item in requested if item not in GOOGLE_RETRIEVERS]
    return ",".join(allowed or ["duckduckgo"])


def effective_retriever(settings_or_retriever: dict[str, Any] | str, tavily_key: str | None = None) -> str:
    if isinstance(settings_or_retriever, dict):
        raw_retriever = settings_or_retriever.get("retriever", "duckduckgo")
        tavily_key = settings_or_retriever.get("tavily_api_key", "") or os.getenv("TAVILY_API_KEY", "")
    else:
        raw_retriever = settings_or_retriever
    parts = [item.strip() for item in sanitize_retriever(str(raw_retriever)).split(",") if item.strip()]
    if (tavily_key or "").strip() and "tavily" not in parts:
        parts.insert(0, "tavily")
    return ",".join(parts or ["duckduckgo"])


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def build_llm_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    top_p = float(settings.get("llm_top_p", 1.0))
    if 0 < top_p < 1.0:
        kwargs["top_p"] = top_p

    if settings["backend"] == "Local Ollama":
        ollama_options = {
            "num_ctx": int(settings.get("ollama_num_ctx", 0) or 0),
            "top_k": int(settings.get("ollama_top_k", 0) or 0),
            "repeat_penalty": float(settings.get("ollama_repeat_penalty", 0) or 0),
            "num_predict": int(settings.get("ollama_num_predict", 0) or 0),
        }
        for key, value in ollama_options.items():
            if value:
                kwargs[key] = value
    return kwargs


def litellm_model_name(settings: dict[str, Any]) -> str:
    raw_model = settings["model"].strip()
    if settings["backend"] == "Local Ollama":
        return raw_model if raw_model.startswith("ollama/") else f"ollama/{strip_provider_prefix(raw_model, 'ollama')}"
    if settings["cloud_provider"] == "DeepSeek":
        return raw_model if raw_model.startswith("deepseek/") else f"deepseek/{strip_provider_prefix(raw_model, 'deepseek')}"
    return raw_model


def gpt_researcher_model_name(settings: dict[str, Any]) -> str:
    raw_model = settings["model"].strip()
    if settings["backend"] == "Local Ollama":
        return f"ollama:{strip_provider_prefix(raw_model, 'ollama')}"
    if settings["cloud_provider"] == "DeepSeek":
        return f"deepseek:{strip_provider_prefix(raw_model, 'deepseek')}"
    return f"openai:{strip_provider_prefix(raw_model, 'openai')}"


def pageindex_model_name(settings: dict[str, Any]) -> str:
    raw_model = settings["model"].strip()
    if settings["backend"] == "Local Ollama":
        return f"ollama/{strip_provider_prefix(raw_model, 'ollama')}"
    if settings["cloud_provider"] == "DeepSeek":
        return f"deepseek/{strip_provider_prefix(raw_model, 'deepseek')}"
    return strip_provider_prefix(raw_model, "openai")


def strip_ollama_tag(model: str) -> str:
    model = strip_provider_prefix(model.strip(), "ollama")
    if model.startswith("ollama/"):
        return model.split("/", 1)[1]
    return model


def preflight_pageindex_backend(settings: dict[str, Any]) -> None:
    if settings["backend"] == "Cloud API":
        if not settings.get("api_key", "").strip():
            raise RuntimeError(f"Missing {settings['cloud_provider']} API key for PageIndex indexing.")
        return

    import requests

    base_url = settings.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
    model_tag = strip_ollama_tag(settings.get("model", ""))
    if not model_tag:
        raise RuntimeError("No Ollama model tag provided for PageIndex indexing.")

    PAGEINDEX_LOGGER.info("PageIndex preflight: probing Ollama endpoint %s", base_url)
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"Ollama preflight failed at {base_url}. Ensure Ollama is running and reachable before indexing."
        ) from exc

    models = [str(item.get("name", "")).strip() for item in payload.get("models", []) if item.get("name")]
    aliases = {name for name in models}
    aliases.update({name.split(":", 1)[0] for name in models if ":" in name})
    if model_tag not in aliases:
        raise RuntimeError(
            f"Ollama model `{model_tag}` is not installed. Available: {', '.join(models[:12]) or '(none)'}."
        )

    os.environ["OLLAMA_BASE_URL"] = base_url
    os.environ["OLLAMA_API_BASE"] = base_url


def configure_runtime_env(settings: dict[str, Any]) -> None:
    model_for_researcher = gpt_researcher_model_name(settings)
    safe_retriever = effective_retriever(settings)
    llm_kwargs = build_llm_kwargs(settings)
    os.environ["FAST_LLM"] = model_for_researcher
    os.environ["SMART_LLM"] = model_for_researcher
    os.environ["STRATEGIC_LLM"] = model_for_researcher
    os.environ["RETRIEVER"] = safe_retriever
    os.environ["RETRIEVERS"] = safe_retriever
    os.environ["LANGUAGE"] = settings.get("language", "english")
    os.environ["CURATE_SOURCES"] = "true" if settings.get("curate_sources", True) else "false"
    os.environ["IMAGE_GENERATION_ENABLED"] = "false"
    os.environ["TEMPERATURE"] = str(settings.get("llm_temperature", 0.2))
    os.environ["LLM_TIMEOUT_SECONDS"] = str(settings.get("llm_timeout_seconds", 120))
    os.environ["LLM_KWARGS"] = json.dumps(llm_kwargs)
    os.environ["REASONING_EFFORT"] = settings.get("reasoning_effort", "medium")
    os.environ["MAX_SEARCH_RESULTS_PER_QUERY"] = str(settings.get("max_search_results_per_query", 12))
    os.environ["DEEP_RESEARCH_BREADTH"] = str(settings.get("deep_research_breadth", 3))
    os.environ["DEEP_RESEARCH_DEPTH"] = str(settings.get("deep_research_depth", 2))
    os.environ["DEEP_RESEARCH_CONCURRENCY"] = str(settings.get("deep_research_concurrency", 3))
    os.environ["MAX_SCRAPER_WORKERS"] = str(settings.get("max_scraper_workers", 10))
    os.environ["SCRAPER_RATE_LIMIT_DELAY"] = str(settings.get("scraper_rate_limit_delay", 0.1))
    os.environ["BROWSE_CHUNK_MAX_LENGTH"] = str(settings.get("browse_chunk_max_length", 12000))
    os.environ["SUMMARY_TOKEN_LIMIT"] = str(settings.get("summary_token_limit", 900))
    os.environ["TOTAL_WORDS"] = str(settings.get("total_words", 1800))
    os.environ["MAX_ITERATIONS"] = str(settings.get("max_iterations", 3))
    os.environ["MAX_SUBTOPICS"] = str(settings.get("max_subtopics", 4))
    os.environ["ARXIV_MAX_RESULTS"] = str(settings.get("arxiv_max_results", settings.get("max_search_results_per_query", 12)))
    os.environ["ARXIV_PAGE_SIZE"] = str(settings.get("arxiv_page_size", 25))
    os.environ["ARXIV_DELAY_SECONDS"] = str(settings.get("arxiv_delay_seconds", 4.0))
    os.environ["ARXIV_NUM_RETRIES"] = str(settings.get("arxiv_num_retries", 4))
    os.environ["PAGEINDEX_LLM_TIMEOUT_SECONDS"] = str(settings.get("pageindex_llm_timeout_seconds", 180))
    os.environ["PAGEINDEX_LLM_MAX_RETRIES"] = str(settings.get("pageindex_llm_max_retries", 3))
    os.environ["PAGEINDEX_TEMPERATURE"] = str(settings.get("pageindex_temperature", 0.0))
    os.environ["PAGEINDEX_TOP_P"] = str(settings.get("pageindex_top_p", 1.0))

    embedding = settings.get("embedding_model", "").strip()
    if embedding:
        os.environ["EMBEDDING"] = embedding

    tavily_key = settings.get("tavily_api_key", "").strip()
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    if settings["backend"] == "Local Ollama":
        base_url = settings["ollama_base_url"].strip().rstrip("/")
        os.environ["OLLAMA_BASE_URL"] = base_url
        os.environ["OLLAMA_API_BASE"] = base_url
        os.environ["PAGEINDEX_OLLAMA_API_BASE"] = base_url
        os.environ["PAGEINDEX_OLLAMA_NUM_CTX"] = str(settings.get("ollama_num_ctx", 0))
        os.environ["PAGEINDEX_OLLAMA_TOP_K"] = str(settings.get("ollama_top_k", 0))
        os.environ["PAGEINDEX_OLLAMA_REPEAT_PENALTY"] = str(settings.get("ollama_repeat_penalty", 0.0))
        os.environ["PAGEINDEX_OLLAMA_NUM_PREDICT"] = str(settings.get("ollama_num_predict", 0))
        if embedding and embedding.startswith("ollama:"):
            os.environ["EMBEDDING"] = embedding
    elif settings["cloud_provider"] == "DeepSeek":
        api_key = settings.get("api_key", "").strip()
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
    else:
        api_key = settings.get("api_key", "").strip()
        base_url = settings.get("openai_base_url", "").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url


def strip_reasoning_prefix(text: str) -> str:
    if not text:
        return ""
    clean = text.strip()
    # Remove explicit reasoning blocks if present.
    clean = re.sub(r"<think>[\s\S]*?</think>\s*", "", clean, flags=re.IGNORECASE)
    # If model started a reasoning block but never emitted a final answer, treat as empty.
    if clean.lower().startswith("<think>"):
        return ""
    return clean.strip()


def extract_llm_response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except Exception:
        content = ""
    if content is None:
        content = ""
    cleaned = strip_reasoning_prefix(str(content))
    if cleaned:
        return cleaned

    # Fallback to dict shape when SDK object shape differs.
    try:
        payload = response.model_dump()
    except Exception:
        try:
            payload = dict(response)
        except Exception:
            payload = {}
    choice = (payload.get("choices") or [{}])[0] if isinstance(payload, dict) else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    raw_text = str(message.get("content") or message.get("reasoning_content") or "").strip()
    return strip_reasoning_prefix(raw_text)


def ollama_direct_chat_completion(
    messages: list[dict[str, str]],
    settings: dict[str, Any],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    import requests

    base_url = settings["ollama_base_url"].strip().rstrip("/")
    endpoint = f"{base_url}/api/chat"
    model_tag = strip_ollama_tag(settings.get("model", ""))
    top_p = float(settings.get("llm_top_p", 1.0))
    timeout_seconds = int(settings.get("llm_timeout_seconds", 120))

    options: dict[str, Any] = {
        "temperature": temperature,
        "num_predict": max(64, min(max_tokens, int(settings.get("ollama_num_predict", max_tokens) or max_tokens))),
    }
    if 0 < top_p < 1.0:
        options["top_p"] = top_p
    num_ctx = int(settings.get("ollama_num_ctx", 0) or 0)
    top_k = int(settings.get("ollama_top_k", 0) or 0)
    repeat_penalty = float(settings.get("ollama_repeat_penalty", 0.0) or 0.0)
    if num_ctx > 0:
        options["num_ctx"] = num_ctx
    if top_k > 0:
        options["top_k"] = top_k
    if repeat_penalty > 0:
        options["repeat_penalty"] = repeat_penalty

    payload = {
        "model": model_tag,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    response = requests.post(endpoint, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    data = response.json()
    content = ((data.get("message") or {}).get("content") or "").strip()
    return strip_reasoning_prefix(content)


def llm_complete(
    messages: list[dict[str, str]],
    settings: dict[str, Any],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    configure_runtime_env(settings)
    if settings["backend"] == "Cloud API" and not settings.get("api_key", "").strip():
        provider = settings.get("cloud_provider", "cloud provider")
        raise RuntimeError(f"Add a {provider} API key in the sidebar before calling the cloud model.")

    from litellm import completion

    effective_temperature = float(settings.get("llm_temperature", 0.2) if temperature is None else temperature)
    effective_max_tokens = int(settings.get("llm_max_tokens", 5000) if max_tokens is None else max_tokens)
    timeout_seconds = int(settings.get("llm_timeout_seconds", 120))
    local_model = strip_ollama_tag(settings.get("model", "")).lower() if settings.get("backend") == "Local Ollama" else ""
    is_local_reasoning_model = any(name in local_model for name in ("deepseek-r1", "magistral"))
    if is_local_reasoning_model:
        effective_max_tokens = min(effective_max_tokens, 1800)
        timeout_seconds = max(timeout_seconds, 300)
    kwargs: dict[str, Any] = {
        "model": litellm_model_name(settings),
        "messages": messages,
        "temperature": effective_temperature,
        "max_tokens": effective_max_tokens,
        "timeout": timeout_seconds,
    }
    top_p = float(settings.get("llm_top_p", 1.0))
    if 0 < top_p < 1.0:
        kwargs["top_p"] = top_p
    if settings["backend"] == "Local Ollama":
        kwargs["api_base"] = settings["ollama_base_url"].strip().rstrip("/")
        for key, value in build_llm_kwargs(settings).items():
            kwargs[key] = value
    elif settings["cloud_provider"] == "OpenAI":
        kwargs["api_key"] = settings.get("api_key", "").strip() or os.getenv("OPENAI_API_KEY")
        if settings.get("openai_base_url", "").strip():
            kwargs["api_base"] = settings["openai_base_url"].strip()
    elif settings["cloud_provider"] == "DeepSeek":
        kwargs["api_key"] = settings.get("api_key", "").strip() or os.getenv("DEEPSEEK_API_KEY")

    response = None
    litellm_error: Exception | None = None
    try:
        response = completion(**kwargs)
    except Exception as exc:
        litellm_error = exc

    if response is not None:
        text = extract_llm_response_text(response)
        if text:
            return text

    # Some local reasoning models can timeout or return empty content through OpenAI-compatible wrappers.
    # Fallback to direct Ollama /api/chat with strict "final answer only" guard and short bounded retries.
    if settings["backend"] == "Local Ollama":
        guard_messages = [
            {
                "role": "system",
                "content": "Return only the final answer. Do not include chain-of-thought, reasoning traces, or <think> tags.",
            },
            *messages,
        ]
        fallback_attempts = 2 if is_local_reasoning_model else 1
        for attempt in range(1, fallback_attempts + 1):
            try:
                fallback_text = ollama_direct_chat_completion(
                    guard_messages,
                    settings,
                    temperature=effective_temperature,
                    max_tokens=min(effective_max_tokens, 1200 if is_local_reasoning_model else 1800),
                )
                if fallback_text:
                    return fallback_text
            except Exception:
                if attempt == fallback_attempts:
                    break

    if litellm_error:
        raise RuntimeError(str(litellm_error)) from litellm_error
    raise RuntimeError("Model returned an empty completion.")


def discover_pdfs(bib_pdf_dir: Path) -> list[Path]:
    if not bib_pdf_dir.exists():
        return []
    return sorted(path for path in bib_pdf_dir.glob("*.pdf") if path.is_file())


def collect_strings(payload: Any) -> Iterable[str]:
    if payload is None:
        return
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from collect_strings(value)
    elif isinstance(payload, (list, tuple, set)):
        for item in payload:
            yield from collect_strings(item)


def extract_urls(*payloads: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for payload in payloads:
        for text in collect_strings(payload):
            for url in re.findall(r"https?://[^\s<>\]\)\"']+", text):
                clean = normalize_url(url)
                if clean not in seen:
                    seen.add(clean)
                    ordered.append(clean)
    return ordered


def normalize_url(url: str) -> str:
    clean = html.unescape(str(url or "")).strip()
    clean = clean.rstrip(".,;:)]}'\"")
    clean = clean.replace("\\", "")
    return clean


def is_candidate_source_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.netloc.lower()
    blocked_hosts = {"example.com", "localhost", "127.0.0.1", "0.0.0.0"}
    if host in blocked_hosts or host.endswith(".local"):
        return False
    if host == "export.arxiv.org" and parsed.path.startswith("/api/"):
        return False
    return True


def validate_source_urls(urls: list[str], limit: int = 60, timeout: int = 8) -> list[dict[str, Any]]:
    import requests

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "ResearchCopilot/1.0 (+https://github.com/assafelovic/gpt-researcher)",
            "Accept": "text/html,application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    for raw_url in urls:
        url = normalize_url(raw_url)
        if not url or url in seen or not is_candidate_source_url(url):
            continue
        seen.add(url)
        try:
            response = session.head(url, timeout=timeout, allow_redirects=True)
            if response.status_code in {405, 403} or response.status_code >= 500:
                response.close()
                response = session.get(url, timeout=timeout, allow_redirects=True, stream=True)
            status = int(response.status_code)
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            final_url = normalize_url(response.url or url)
            response.close()
            if 200 <= status < 400 and is_candidate_source_url(final_url):
                validated.append({"url": final_url, "status": status, "content_type": content_type})
        except Exception:
            continue
        if len(validated) >= limit:
            break
    return validated


def looks_like_pdf_source(source: dict[str, Any]) -> bool:
    url = str(source.get("url", "")).lower()
    content_type = str(source.get("content_type", "")).lower()
    return url.endswith(".pdf") or "/pdf/" in url or "application/pdf" in content_type


def append_verified_sources(report: str, validated_sources: list[dict[str, Any]], discovered_count: int) -> str:
    if "## Verified Source Links" in report:
        return report
    lines = [
        "## Verified Source Links",
        "",
        "These links were discovered during the run and responded successfully during the local validation pass.",
        "",
    ]
    if validated_sources:
        for index, source in enumerate(validated_sources, start=1):
            url = source["url"]
            content_type = source.get("content_type") or "unknown"
            lines.append(f"{index}. [{url}]({url}) — HTTP {source.get('status')}, `{content_type}`")
    else:
        lines.append(
            "No discovered source URL passed validation. Re-run with a broader retriever such as `duckduckgo,arxiv` "
            "or add a Tavily key for stronger web discovery."
        )
    lines.extend(
        [
            "",
            f"Validation summary: {len(validated_sources)} reachable links from {discovered_count} discovered candidate URLs.",
        ]
    )
    return report.rstrip() + "\n\n" + "\n".join(lines) + "\n"


def build_research_role_prompt() -> str:
    skill_pack = load_agent_skill_pack(AGENT_SKILLS_DIR)
    return (
        f"You are a senior scientific research agent.\n\n{skill_pack}"
        if skill_pack
        else "You are a senior scientific research agent. Produce evidence-grounded research reports with citations."
    )


def build_rag_role_prompt() -> str:
    rag_skill = load_skill_file(AGENT_SKILLS_DIR, "vectorless_tree_reasoning_chat.md")
    if not rag_skill:
        return (
            "You are a vectorless PageIndex research copilot. Reason over semantic tree structure and page-linked evidence. "
            "Cite all document-grounded claims with [filename :: node_id :: pp. start-end :: title]."
        )
    return f"You are a vectorless PageIndex research copilot.\n\n{rag_skill}"


PLAN_HELP_TEXT = (
    "Each row is one search direction. Query = the exact search string sent to retrievers; keep it concise, ideally "
    "under 400 characters for Tavily. Goal = why the query exists and what evidence it should retrieve. "
    "Bulk syntax, if you copy a plan elsewhere: query :: goal. Example: "
    '"patent classification" "sustainable development goals" dataset :: Find datasets and label sources.'
)


def infer_plan_goal(query_text: str) -> str:
    lower = query_text.lower()
    if any(term in lower for term in ("dataset", "benchmark", "corpus")):
        return "Find reusable datasets, benchmarks, labels, splits, and evaluation artifacts."
    if any(term in lower for term in ("github", "code", "repository", "reproduc")):
        return "Find reproducible implementations, code repositories, licenses, and released artifacts."
    if any(term in lower for term in ("pdf", "peer", "journal", "transactions", "acm", "ieee", "springer", "elsevier")):
        return "Find primary scholarly papers and accessible full-text sources."
    if any(term in lower for term in ("weak supervision", "zero-shot", "llm", "bert", "transformer", "ontology")):
        return "Find method papers and compare modeling approaches for the classification task."
    return "Collect high-signal evidence relevant to the research question."


def normalize_plan_items(plan_items: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in plan_items:
        if isinstance(item, dict):
            query_text = str(item.get("query", "")).strip()
            goal = str(item.get("researchGoal", "") or item.get("goal", "")).strip()
        else:
            query_text = str(item).strip()
            goal = ""
        if not query_text:
            continue
        normalized.append({"query": query_text, "researchGoal": goal or infer_plan_goal(query_text)})
    return normalized


def parse_plan_editor(text: str) -> list[dict[str, str]]:
    planned: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        if "::" in line:
            query_text, goal = [part.strip() for part in line.split("::", 1)]
        else:
            query_text, goal = line, ""
        if query_text:
            planned.append({"query": query_text, "researchGoal": goal or infer_plan_goal(query_text)})
    return planned


def format_plan_for_editor(plan_items: list[Any]) -> str:
    lines = []
    for item in plan_items:
        if isinstance(item, dict):
            query_text = str(item.get("query", "")).strip()
            goal = str(item.get("researchGoal", "")).strip()
        else:
            query_text = str(item).strip()
            goal = ""
        if not query_text:
            continue
        line = f"- {query_text}"
        if goal:
            line += f" :: {goal}"
        lines.append(line)
    return "\n".join(lines)


def clear_research_plan_state() -> None:
    st.session_state["research_plan_text"] = ""
    st.session_state["research_plan_editor"] = ""
    st.session_state["research_plan_items"] = []
    st.session_state["research_plan_source"] = {}


def reset_plan_widget_keys() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith(("plan_query_", "plan_goal_")):
            del st.session_state[key]


def set_research_plan_items(plan_items: list[Any], plan_key: dict[str, Any]) -> None:
    normalized = normalize_plan_items(plan_items)
    st.session_state["research_plan_items"] = normalized
    st.session_state["research_plan_text"] = format_plan_for_editor(normalized)
    st.session_state["research_plan_source"] = plan_key
    reset_plan_widget_keys()


def append_blank_plan_item() -> None:
    items = list(st.session_state.get("research_plan_items", []))
    items.append({"query": "", "researchGoal": ""})
    st.session_state["research_plan_items"] = items


def sync_plan_from_row_widgets() -> list[dict[str, str]]:
    source_items = list(st.session_state.get("research_plan_items", []))
    synced_all: list[dict[str, str]] = []
    runnable: list[dict[str, str]] = []
    for index, item in enumerate(source_items):
        query_text = str(st.session_state.get(f"plan_query_{index}", item.get("query", ""))).strip()
        goal = str(st.session_state.get(f"plan_goal_{index}", item.get("researchGoal", ""))).strip()
        synced_all.append({"query": query_text, "researchGoal": goal or (infer_plan_goal(query_text) if query_text else "")})
        if query_text:
            runnable.append({"query": query_text, "researchGoal": goal or infer_plan_goal(query_text)})
    st.session_state["research_plan_items"] = synced_all
    st.session_state["research_plan_text"] = format_plan_for_editor(runnable)
    return runnable


def create_researcher(
    query: str,
    report_type: str,
    settings: dict[str, Any],
    log_handler: Any | None = None,
    planned_queries: list[dict[str, str]] | None = None,
):
    from gpt_researcher import GPTResearcher

    return GPTResearcher(
        query=query,
        report_type=report_type,
        config_path=None,
        headers={"retrievers": effective_retriever(settings)},
        role=build_research_role_prompt(),
        log_handler=log_handler,
        preplanned_queries=planned_queries or [],
    )


async def generate_research_plan(
    query: str,
    report_type: str,
    settings: dict[str, Any],
    log_handler: Any | None = None,
) -> list[dict[str, str]]:
    require_runtime_modules(settings)
    configure_runtime_env(settings)
    if settings["backend"] == "Cloud API" and not settings.get("api_key", "").strip():
        raise RuntimeError(f"Add a {settings['cloud_provider']} API key in the sidebar before planning research.")

    researcher = create_researcher(query, report_type, settings, log_handler=log_handler)
    if log_handler:
        await log_handler.on_research_step("planning_search_strategy", {"query": query})
    if report_type == "deep" and researcher.deep_researcher:
        plan = await researcher.deep_researcher.generate_search_queries(
            query,
            num_queries=int(settings.get("deep_research_breadth", 3)),
        )
    else:
        plan = await researcher.research_conductor.plan_research(query, researcher.query_domains)
    normalized = parse_plan_editor(format_plan_for_editor(plan))
    if not normalized:
        normalized = [{"query": query, "researchGoal": "Fallback to the original user query."}]
    return normalized


def pdf_name_from_url(url: str, index: int) -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    if not name.lower().endswith(".pdf"):
        name = f"acquired_literature_{index:03d}.pdf"
    return safe_slug(name, f"acquired_literature_{index:03d}.pdf")


def download_pdf_urls(urls: list[str], target_dir: Path, limit: int = 12) -> list[Path]:
    import requests

    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "ResearchCopilot/1.0 (+local workspace)"})

    for index, url in enumerate(urls[:limit], start=1):
        try:
            response = session.get(url, timeout=25, allow_redirects=True, stream=True)
            content_type = response.headers.get("content-type", "").lower()
            looks_like_pdf = ".pdf" in url.lower() or "application/pdf" in content_type
            if response.status_code >= 400 or not looks_like_pdf:
                response.close()
                continue

            candidate = target_dir / pdf_name_from_url(response.url or url, index)
            if candidate.exists():
                candidate = target_dir / f"{candidate.stem}_{hashlib.sha1(url.encode()).hexdigest()[:8]}.pdf"
            temp_path = candidate.with_suffix(".pdf.download")
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        handle.write(chunk)
            temp_path.replace(candidate)
            downloaded.append(candidate)
        except Exception:
            continue
    return downloaded


async def run_gpt_researcher(
    query: str,
    report_type: str,
    settings: dict[str, Any],
    paths: RuntimePaths,
    log_handler: Any | None = None,
    planned_queries: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    require_runtime_modules(settings)
    configure_runtime_env(settings)
    if settings["backend"] == "Cloud API" and not settings.get("api_key", "").strip():
        raise RuntimeError(f"Add a {settings['cloud_provider']} API key in the sidebar before running Deep Search.")

    researcher = create_researcher(
        query=query,
        report_type=report_type,
        settings=settings,
        log_handler=log_handler,
        planned_queries=planned_queries,
    )
    research_result = await researcher.conduct_research()
    report = await researcher.write_report()
    source_urls = []
    try:
        source_urls = researcher.get_source_urls()
    except Exception:
        source_urls = list(getattr(researcher, "visited_urls", []) or [])
    try:
        research_sources = researcher.get_research_sources()
    except Exception:
        research_sources = getattr(researcher, "research_sources", []) or []

    urls = extract_urls(report, research_result, source_urls, research_sources)
    if log_handler:
        await log_handler.on_research_step("validating_sources", {"candidate_urls": len(urls)})
    validated_sources = validate_source_urls(urls, limit=int(settings.get("source_validation_limit", 80)))
    if log_handler:
        await log_handler.on_research_step(
            "sources_validated", {"candidate_urls": len(urls), "verified_urls": len(validated_sources)}
        )
    report = append_verified_sources(report, validated_sources, len(urls))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = paths.bib_pdf / f"deep_search_report_{timestamp}.md"
    atomic_write_text(report_path, report)

    pdf_urls = [source["url"] for source in validated_sources if looks_like_pdf_source(source)]
    if log_handler:
        await log_handler.on_research_step("downloading_pdfs", {"pdf_urls": len(pdf_urls)})
    downloaded = download_pdf_urls(pdf_urls, paths.bib_pdf, limit=int(settings.get("pdf_download_limit", 15)))
    return {
        "report": report,
        "report_path": report_path,
        "downloaded": downloaded,
        "url_count": len(urls),
        "verified_url_count": len(validated_sources),
    }


def index_single_pdf(pdf_path: Path, settings: dict[str, Any], paths: RuntimePaths) -> dict[str, Any]:
    started = time.perf_counter()
    PAGEINDEX_LOGGER.info("Index request received for `%s`", pdf_path)
    require_runtime_modules(settings)
    configure_runtime_env(settings)
    preflight_pageindex_backend(settings)
    pdf_hash = sha256_file(pdf_path)
    output_path = paths.pageindex_cache / f"{safe_slug(pdf_path.stem)}__{pdf_hash[:12]}_structure.json"
    if output_path.exists():
        PAGEINDEX_LOGGER.info("Cache hit for `%s` -> `%s`", pdf_path.name, output_path.name)
        return {"status": "cached", "pdf": pdf_path, "output": output_path, "log": latest_pageindex_log_path(pdf_path.name)}

    try:
        from pageindex import page_index_main
        from pageindex.utils import ConfigLoader
    except Exception as exc:
        raise RuntimeError(
            "PageIndex is not importable. Install the environment with `pip install -r requirements.txt`."
        ) from exc

    user_opt = {
        "model": pageindex_model_name(settings),
        "toc_check_page_num": int(settings["pageindex_toc_pages"]),
        "max_page_num_each_node": int(settings["pageindex_max_pages_per_node"]),
        "max_token_num_each_node": int(settings["pageindex_max_tokens_per_node"]),
        "if_add_node_id": "yes",
        "if_add_node_summary": "yes",
        "if_add_doc_description": "yes",
        "if_add_node_text": "no",
    }
    PAGEINDEX_LOGGER.info(
        "PageIndex options for `%s`: model=%s toc_pages=%s max_pages_per_node=%s max_tokens_per_node=%s timeout=%ss retries=%s temperature=%s top_p=%s",
        pdf_path.name,
        user_opt["model"],
        user_opt["toc_check_page_num"],
        user_opt["max_page_num_each_node"],
        user_opt["max_token_num_each_node"],
        settings.get("pageindex_llm_timeout_seconds", 180),
        settings.get("pageindex_llm_max_retries", 3),
        settings.get("pageindex_temperature", 0.0),
        settings.get("pageindex_top_p", 1.0),
    )
    opt = ConfigLoader().load({key: value for key, value in user_opt.items() if value is not None})
    try:
        PAGEINDEX_LOGGER.info("Calling PageIndex engine for `%s`", pdf_path.name)
        tree = page_index_main(str(pdf_path), opt)
    except Exception as exc:
        PAGEINDEX_LOGGER.error("PageIndex failed on `%s`: %s", pdf_path.name, exc)
        PAGEINDEX_LOGGER.error(traceback.format_exc())
        raise

    payload = {
        "schema_version": 1,
        "source_pdf": str(pdf_path),
        "source_pdf_name": pdf_path.name,
        "source_sha256": pdf_hash,
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
        "pageindex_model": user_opt["model"],
        "tree": tree,
    }
    atomic_write_json(output_path, payload)
    elapsed = time.perf_counter() - started
    PAGEINDEX_LOGGER.info("Indexed `%s` in %.2fs -> `%s`", pdf_path.name, elapsed, output_path.name)
    return {"status": "indexed", "pdf": pdf_path, "output": output_path, "log": latest_pageindex_log_path(pdf_path.name)}


def write_workspace_manifest(paths: RuntimePaths) -> None:
    entries = []
    for cache_file in sorted(paths.pageindex_cache.glob("*_structure.json")):
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            entries.append(
                {
                    "cache_file": str(cache_file),
                    "source_pdf": payload.get("source_pdf"),
                    "source_pdf_name": payload.get("source_pdf_name"),
                    "source_sha256": payload.get("source_sha256"),
                    "indexed_at": payload.get("indexed_at"),
                    "pageindex_model": payload.get("pageindex_model"),
                }
            )
        except Exception:
            continue
    atomic_write_json(
        paths.pageindex_cache / "workspace_index.json",
        {"generated_at": datetime.now().isoformat(timespec="seconds"), "entries": entries},
    )


def reindex_workspace(settings: dict[str, Any], paths: RuntimePaths) -> list[dict[str, Any]]:
    pdfs = discover_pdfs(paths.bib_pdf)
    PAGEINDEX_LOGGER.info(
        "Workspace reindex started: %s PDF(s), backend=%s model=%s cache_dir=%s",
        len(pdfs),
        settings.get("backend"),
        pageindex_model_name(settings),
        paths.pageindex_cache,
    )
    results: list[dict[str, Any]] = []
    for pdf_path in pdfs:
        try:
            results.append(index_single_pdf(pdf_path, settings, paths))
        except Exception as exc:
            results.append(
                {
                    "status": "error",
                    "pdf": pdf_path,
                    "output": None,
                    "error": str(exc),
                    "log": latest_pageindex_log_path(pdf_path.name),
                }
            )
    write_workspace_manifest(paths)
    summary = {
        "indexed": sum(1 for item in results if item["status"] == "indexed"),
        "cached": sum(1 for item in results if item["status"] == "cached"),
        "errors": sum(1 for item in results if item["status"] == "error"),
    }
    PAGEINDEX_LOGGER.info("Workspace reindex completed: %s", summary)
    return results


def load_index_payloads(paths: RuntimePaths) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not paths.pageindex_cache.exists():
        return payloads
    for cache_file in sorted(paths.pageindex_cache.glob("*_structure.json")):
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if "tree" not in payload:
                payload = {"schema_version": 0, "source_pdf_name": cache_file.stem, "source_pdf": "", "tree": payload}
            payload["_cache_file"] = str(cache_file)
            payloads.append(payload)
        except Exception:
            continue
    return payloads


def child_nodes(node: dict[str, Any]) -> list[Any]:
    for key in ("nodes", "children", "subsections"):
        value = node.get(key)
        if isinstance(value, list):
            return value
    return []


def flatten_nodes(tree: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if any(key in value for key in ("title", "node_id", "start_index", "summary")):
                nodes.append(value)
            for child in child_nodes(value):
                visit(child)
            if "structure" in value:
                visit(value["structure"])
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(tree)
    return nodes


def find_node(tree: Any, node_id: str | None, title: str | None = None) -> dict[str, Any] | None:
    candidates = flatten_nodes(tree)
    if node_id:
        for node in candidates:
            if str(node.get("node_id", "")).strip() == str(node_id).strip():
                return node
    if title:
        normalized = title.lower().strip()
        for node in candidates:
            if str(node.get("title", "")).lower().strip() == normalized:
                return node
    return None


def coerce_page(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def format_node_line(node: dict[str, Any], depth: int = 0) -> str:
    title = str(node.get("title", "Untitled section"))
    node_id = str(node.get("node_id", "no-node-id"))
    start = node.get("start_index", "?")
    end = node.get("end_index", "?")
    summary = str(node.get("summary", "")).replace("\n", " ").strip()
    indent = "  " * depth
    return f"{indent}- node_id={node_id}; pages={start}-{end}; title={title}; summary={summary}"


def format_tree(tree: Any, depth: int = 0, max_depth: int = 6) -> list[str]:
    lines: list[str] = []
    if depth > max_depth:
        return lines
    if isinstance(tree, dict):
        if any(key in tree for key in ("title", "node_id", "start_index", "summary")):
            lines.append(format_node_line(tree, depth))
            next_depth = depth + 1
        else:
            next_depth = depth
        if "structure" in tree:
            lines.extend(format_tree(tree["structure"], next_depth, max_depth))
        for child in child_nodes(tree):
            lines.extend(format_tree(child, next_depth, max_depth))
    elif isinstance(tree, list):
        for item in tree:
            lines.extend(format_tree(item, depth, max_depth))
    return lines


def build_tree_context(payloads: list[dict[str, Any]], char_budget: int) -> str:
    sections: list[str] = []
    used_chars = 0
    for index, payload in enumerate(payloads, start=1):
        doc_key = f"DOC_{index}"
        header = (
            f"\n### {doc_key}: {payload.get('source_pdf_name') or Path(payload.get('source_pdf', '')).name}\n"
            f"source_pdf={payload.get('source_pdf', '')}\n"
        )
        lines = format_tree(payload.get("tree"))
        block = header + "\n".join(lines)
        if used_chars + len(block) > char_budget:
            remaining = max(0, char_budget - used_chars)
            if remaining > 1000:
                sections.append(block[:remaining] + "\n[Tree context truncated at configured budget.]")
            break
        sections.append(block)
        used_chars += len(block)
    return "\n".join(sections).strip()


def parse_json_response(text: str) -> dict[str, Any]:
    clean = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    object_match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))
    candidates.append(clean)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return {}


def extract_pdf_pages(pdf_path: Path, start_page: int | None, end_page: int | None, max_chars: int = 7000) -> str:
    if not pdf_path.exists():
        return ""
    start = max(1, start_page or 1)
    end = max(start, end_page or start)
    text_parts: list[str] = []
    try:
        import fitz

        document = fitz.open(pdf_path)
        page_count = document.page_count
        end = min(end, page_count)
        for page_number in range(start, end + 1):
            page_text = document.load_page(page_number - 1).get_text("text")
            text_parts.append(f"\n[page {page_number}]\n{page_text}")
            if sum(len(part) for part in text_parts) >= max_chars:
                break
        document.close()
    except Exception:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(pdf_path))
            end = min(end, len(reader.pages))
            for page_number in range(start, end + 1):
                page_text = reader.pages[page_number - 1].extract_text() or ""
                text_parts.append(f"\n[page {page_number}]\n{page_text}")
                if sum(len(part) for part in text_parts) >= max_chars:
                    break
        except Exception:
            return ""
    joined = "\n".join(text_parts)
    return joined[:max_chars]


def build_evidence_blocks(
    selected_sections: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    max_sections: int = 8,
) -> str:
    blocks: list[str] = []
    for section in selected_sections[:max_sections]:
        doc_key = str(section.get("doc_key", "")).strip()
        match = re.search(r"\d+", doc_key)
        if not match:
            continue
        doc_index = int(match.group(0)) - 1
        if doc_index < 0 or doc_index >= len(payloads):
            continue
        payload = payloads[doc_index]
        node = find_node(payload.get("tree"), section.get("node_id"), section.get("title"))
        if not node:
            continue
        pdf_path = Path(payload.get("source_pdf", ""))
        start = coerce_page(node.get("start_index"))
        end = coerce_page(node.get("end_index"))
        title = str(node.get("title", "Untitled section"))
        node_id = str(node.get("node_id", "no-node-id"))
        page_text = extract_pdf_pages(pdf_path, start, end)
        summary = str(node.get("summary", "")).strip()
        blocks.append(
            f"### {doc_key} | {payload.get('source_pdf_name')} | node_id={node_id} | pages={start}-{end} | {title}\n"
            f"PageIndex summary: {summary}\n"
            f"Extracted page-range evidence:\n{page_text}"
        )
    return "\n\n".join(blocks)


def answer_workspace_question(
    question: str,
    settings: dict[str, Any],
    paths: RuntimePaths,
    chat_history: list[dict[str, str]],
) -> str:
    payloads = load_index_payloads(paths)
    if not payloads:
        return "No PageIndex cache is loaded yet. Add PDFs to the staging room and run PageIndex indexing first."

    rag_role_prompt = build_rag_role_prompt()
    tree_context = build_tree_context(payloads, int(settings["tree_context_budget"]))
    selector_messages = [
        {
            "role": "system",
            "content": (
                f"{rag_role_prompt}\n\n"
                "Task phase: section selection only.\n"
                "Select the most relevant document sections by reasoning over the semantic tree only. "
                "Return strict JSON with this shape: "
                '{"sections":[{"doc_key":"DOC_1","node_id":"0001","title":"section title","reason":"brief reason"}]}. '
                "Select no more than 8 sections."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nPageIndex tree context:\n{tree_context}",
        },
    ]
    selector_text = llm_complete(selector_messages, settings, temperature=0.0, max_tokens=1400)
    selected = parse_json_response(selector_text).get("sections", [])
    if not isinstance(selected, list):
        selected = []
    evidence = build_evidence_blocks(selected, payloads)

    recent_history = "\n".join(
        f"{message['role']}: {message['content']}" for message in chat_history[-6:] if message.get("content")
    )
    final_messages = [
        {
            "role": "system",
            "content": (
                f"{rag_role_prompt}\n\n"
                "Do not use vector-search language, embeddings, nearest neighbors, or similarity scores. "
                "Use only the supplied semantic tree and extracted page-range evidence. "
                "Cite every document-grounded claim with [filename :: node_id :: pp. start-end :: title]. "
                "If the evidence is insufficient, state exactly what is missing."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\nRecent conversation:\n{recent_history}\n\n"
                f"PageIndex tree context:\n{tree_context}\n\nSelected page-range evidence:\n{evidence or 'No page text extracted; answer only from tree summaries.'}"
            ),
        },
    ]
    return llm_complete(final_messages, settings)


def clean_latex_response(text: str) -> str:
    clean = text.strip()
    clean = re.sub(r"^```(?:latex|tex)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    match = re.search(r"\\documentclass[\s\S]*?\\end\{document\}", clean)
    if match:
        return match.group(0).strip() + "\n"
    return clean.strip() + "\n"


def modify_latex_source(current_source: str, instruction: str, settings: dict[str, Any]) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful LaTeX co-author. Return only the complete raw LaTeX source. "
                "Do not include markdown fences, explanations, or commentary. Preserve packages and compilability."
            ),
        },
        {
            "role": "user",
            "content": f"Instruction:\n{instruction}\n\nCurrent LaTeX source:\n{current_source}",
        },
    ]
    return clean_latex_response(llm_complete(messages, settings))


def compile_latex(paths: RuntimePaths) -> dict[str, Any]:
    if shutil.which("pdflatex") is None:
        return {"ok": False, "log": "pdflatex was not found on PATH. Install a TeX distribution such as TeX Live."}

    paths.compiled_output.mkdir(parents=True, exist_ok=True)
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        f"-output-directory={str(paths.compiled_output)}",
        paths.paper_tex.name,
    ]
    combined_log: list[str] = []
    ok = True
    for run_number in range(1, 3):
        try:
            result = subprocess.run(
                command,
                cwd=str(paths.paper_tex.parent),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            combined_log.append(f"===== pdflatex run {run_number} =====\n{result.stdout}\n{result.stderr}")
            if result.returncode != 0:
                ok = False
        except subprocess.TimeoutExpired as exc:
            ok = False
            combined_log.append(f"===== pdflatex run {run_number} timed out =====\n{exc}")
            break

    pdf_path = paths.compiled_output / f"{paths.paper_tex.stem}.pdf"
    if not pdf_path.exists():
        ok = False

    log_path = paths.compiled_output / f"{paths.paper_tex.stem}.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        important = "\n".join(line for line in log_text.splitlines() if line.startswith("!") or "Error" in line)
        if important:
            combined_log.append(f"===== TeX error highlights =====\n{important}")

    return {"ok": ok, "pdf_path": pdf_path, "log": "\n\n".join(combined_log)}


def save_editor_if_changed(paths: RuntimePaths) -> None:
    current_text = st.session_state.get("latex_editor", "")
    current_sha = sha256_text(current_text)
    if current_sha != st.session_state.get("latex_saved_sha"):
        atomic_write_text(paths.paper_tex, current_text)
        st.session_state["latex_saved_sha"] = current_sha


def safe_toast(message: str) -> None:
    try:
        st.toast(message)
    except Exception:
        st.success(message)


def missing_runtime_modules(settings: dict[str, Any]) -> list[str]:
    required_modules = [
        "gpt_researcher",
        "pageindex",
        "litellm",
        "openai",
        "dotenv",
        "fitz",
        "PyPDF2",
        "requests",
        "langchain_mcp_adapters",
    ]
    safe_retriever = sanitize_retriever(settings.get("retriever", "duckduckgo"))
    if "duckduckgo" in safe_retriever.split(","):
        required_modules.extend(["ddgs", "duckduckgo_search"])
    if settings.get("backend") == "Local Ollama":
        required_modules.append("langchain_ollama")
    if settings.get("backend") == "Cloud API" and settings.get("cloud_provider") == "DeepSeek":
        required_modules.append("langchain_deepseek")

    missing: list[str] = []
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return sorted(set(missing))


def require_runtime_modules(settings: dict[str, Any]) -> None:
    missing = missing_runtime_modules(settings)
    if missing:
        module_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing runtime module(s): {module_list}. "
            "Install the updated environment with `python -m pip install -r requirements.txt`, then restart Streamlit."
        )


st.set_page_config(page_title="Research Copilot", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
    /* Metrics: use theme variables so text is readable in dark/light mode */
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.28);
        padding: 0.8rem;
        border-radius: 8px;
    }
    div[data-testid="stMetric"] * {
        color: var(--text-color) !important;
    }
    .rc-progress {
        border: 1px solid rgba(128, 128, 128, 0.24);
        border-radius: 8px;
        padding: 0.6rem 0.7rem;
        margin-top: 0.75rem;
        background: var(--secondary-background-color);
    }
    .rc-step {
        display: grid;
        grid-template-columns: 2.1rem minmax(0, 1fr);
        gap: 0.65rem;
        align-items: start;
        padding: 0.52rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.16);
    }
    .rc-step:last-child { border-bottom: 0; }
    .rc-indicator {
        min-height: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .rc-spinner {
        width: 1rem;
        height: 1rem;
        border-radius: 50%;
        border: 2px solid rgba(128, 128, 128, 0.28);
        border-top-color: #ff4b4b;
        animation: rc-spin 0.8s linear infinite;
    }
    .rc-check, .rc-error {
        width: 1.35rem;
        height: 1.35rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.64rem;
        line-height: 1;
        font-weight: 800;
    }
    .rc-check {
        color: #063b1f;
        background: #2fd27f;
    }
    .rc-error {
        color: #fff;
        background: #ff4b4b;
    }
    .rc-label {
        color: var(--text-color);
        font-size: 0.93rem;
        font-weight: 700;
        line-height: 1.25;
    }
    .rc-detail {
        color: rgba(128, 128, 128, 0.95);
        font-size: 0.78rem;
        line-height: 1.35;
        margin-top: 0.12rem;
        overflow-wrap: anywhere;
    }
    @keyframes rc-spin { to { transform: rotate(360deg); } }
    textarea {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;}
    .stTabs [data-baseweb="tab-list"] {gap: 0.4rem;}
    .stTabs [data-baseweb="tab"] {height: 2.6rem; padding-left: 1rem; padding-right: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.title("Research Copilot")
    workspace_title = st.text_input("Workspace title", value=st.session_state.get("workspace_title", DEFAULT_TITLE))
    st.session_state["workspace_title"] = workspace_title

    backend = st.selectbox("LLM backend", ["Local Ollama", "Cloud API"], index=0)
    cloud_provider = "OpenAI"
    api_key = ""
    openai_base_url = os.getenv("OPENAI_BASE_URL", "")
    ollama_base_url = DEFAULT_OLLAMA_BASE_URL

    if backend == "Cloud API":
        configured_provider = os.getenv("DEFAULT_CLOUD_PROVIDER", "openai").lower()
        provider_index = 1 if configured_provider == "deepseek" else 0
        cloud_provider = st.selectbox("Cloud provider", ["OpenAI", "DeepSeek"], index=provider_index)
        key_name = "OPENAI_API_KEY" if cloud_provider == "OpenAI" else "DEEPSEEK_API_KEY"
        api_key = st.text_input(f"{cloud_provider} API key", value=os.getenv(key_name, ""), type="password")
        default_cloud_model = DEFAULT_CLOUD_MODEL if cloud_provider == "OpenAI" else DEFAULT_DEEPSEEK_MODEL
        active_model = st.text_input("Cloud model", value=default_cloud_model)
        if cloud_provider == "OpenAI":
            openai_base_url = st.text_input("OpenAI-compatible base URL", value=openai_base_url)
    else:
        ollama_base_url = st.text_input("Ollama base URL", value=DEFAULT_OLLAMA_BASE_URL)
        active_model = st.text_input("Local model tag", value=DEFAULT_OLLAMA_MODEL)

    with st.expander("Discovery and indexing", expanded=False):
        default_retriever = sanitize_retriever(os.getenv("RETRIEVER", "duckduckgo,arxiv"))
        retrievers = ["duckduckgo", "arxiv", "semantic_scholar", "pubmed_central", "tavily", "searx", "bing", "exa"]
        default_retrievers = [item for item in default_retriever.split(",") if item in retrievers] or ["duckduckgo"]
        selected_retrievers = st.multiselect("Research retrievers", retrievers, default=default_retrievers)
        retriever = ",".join(selected_retrievers or ["duckduckgo"])
        tavily_api_key = st.text_input("Tavily API key", value=os.getenv("TAVILY_API_KEY", ""), type="password")
        default_embedding = (
            f"ollama:{DEFAULT_EMBEDDING_MODEL}"
            if backend == "Local Ollama" or cloud_provider == "DeepSeek"
            else "openai:text-embedding-3-small"
        )
        embedding_model = st.text_input(
            "Research embedding model",
            value=os.getenv("EMBEDDING", default_embedding),
        )
        max_search_results_per_query = st.number_input(
            "Max results per query",
            min_value=3,
            max_value=100,
            value=env_int("MAX_SEARCH_RESULTS_PER_QUERY", 12),
            help="GPT Researcher default is 5. Raise this for broader discovery; reduce it when APIs rate-limit.",
        )
        source_validation_limit = st.number_input(
            "Reachable source validation limit",
            min_value=10,
            max_value=250,
            value=env_int("SOURCE_VALIDATION_LIMIT", 80),
        )
        pdf_download_limit = st.number_input(
            "Verified PDF download limit",
            min_value=0,
            max_value=60,
            value=env_int("PDF_DOWNLOAD_LIMIT", 15),
        )
        deep_research_breadth = st.number_input("Deep research breadth", min_value=1, max_value=8, value=env_int("DEEP_RESEARCH_BREADTH", 3))
        deep_research_depth = st.number_input("Deep research depth", min_value=1, max_value=5, value=env_int("DEEP_RESEARCH_DEPTH", 2))
        deep_research_concurrency = st.number_input(
            "Deep research concurrency",
            min_value=1,
            max_value=8,
            value=env_int("DEEP_RESEARCH_CONCURRENCY", 3),
            help="Keep this modest when arXiv is enabled; arXiv itself is serialized to respect its API terms.",
        )
        max_scraper_workers = st.number_input("Max scraper workers", min_value=1, max_value=40, value=env_int("MAX_SCRAPER_WORKERS", 10))
        scraper_rate_limit_delay = st.number_input(
            "Scraper delay seconds",
            min_value=0.0,
            max_value=5.0,
            step=0.1,
            value=env_float("SCRAPER_RATE_LIMIT_DELAY", 0.1),
        )
        browse_chunk_max_length = st.number_input(
            "Browse chunk max length",
            min_value=2000,
            max_value=50000,
            step=1000,
            value=env_int("BROWSE_CHUNK_MAX_LENGTH", 12000),
        )
        summary_token_limit = st.number_input(
            "Summary token limit",
            min_value=300,
            max_value=4000,
            step=100,
            value=env_int("SUMMARY_TOKEN_LIMIT", 900),
        )
        total_words = st.number_input("Target report words", min_value=600, max_value=8000, step=100, value=env_int("TOTAL_WORDS", 1800))
        max_iterations = st.number_input("Research iterations", min_value=1, max_value=8, value=env_int("MAX_ITERATIONS", 3))
        max_subtopics = st.number_input("Max subtopics", min_value=1, max_value=10, value=env_int("MAX_SUBTOPICS", 4))
        st.markdown("**arXiv API discipline**")
        arxiv_max_results = st.number_input(
            "arXiv max results per query",
            min_value=3,
            max_value=100,
            value=env_int("ARXIV_MAX_RESULTS", min(25, int(max_search_results_per_query))),
        )
        arxiv_page_size = st.number_input("arXiv page size", min_value=3, max_value=100, value=env_int("ARXIV_PAGE_SIZE", 25))
        arxiv_delay_seconds = st.number_input(
            "arXiv delay seconds",
            min_value=3.0,
            max_value=30.0,
            step=0.5,
            value=max(3.0, env_float("ARXIV_DELAY_SECONDS", 4.0)),
        )
        arxiv_num_retries = st.number_input("arXiv retries", min_value=0, max_value=8, value=env_int("ARXIV_NUM_RETRIES", 4))
        pageindex_toc_pages = st.number_input("PageIndex ToC scan pages", min_value=1, max_value=80, value=int(os.getenv("PAGEINDEX_TOC_CHECK_PAGES", "20")))
        pageindex_max_pages_per_node = st.number_input(
            "Max pages per node", min_value=1, max_value=80, value=int(os.getenv("PAGEINDEX_MAX_PAGES_PER_NODE", "12"))
        )
        pageindex_max_tokens_per_node = st.number_input(
            "Max tokens per node", min_value=1000, max_value=100000, step=1000, value=int(os.getenv("PAGEINDEX_MAX_TOKENS_PER_NODE", "20000"))
        )
        pageindex_llm_timeout_seconds = st.number_input(
            "PageIndex LLM timeout (seconds)",
            min_value=20,
            max_value=1200,
            step=10,
            value=env_int("PAGEINDEX_LLM_TIMEOUT_SECONDS", 180),
        )
        pageindex_llm_max_retries = st.number_input(
            "PageIndex LLM retries",
            min_value=1,
            max_value=10,
            value=env_int("PAGEINDEX_LLM_MAX_RETRIES", 3),
        )
        pageindex_temperature = st.number_input(
            "PageIndex temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            value=env_float("PAGEINDEX_TEMPERATURE", 0.0),
        )
        pageindex_top_p = st.number_input(
            "PageIndex top-p",
            min_value=0.05,
            max_value=1.0,
            step=0.05,
            value=env_float("PAGEINDEX_TOP_P", 1.0),
        )
        tree_context_budget = st.number_input(
            "Tree context char budget", min_value=8000, max_value=250000, step=1000, value=int(os.getenv("TREE_CONTEXT_CHAR_BUDGET", "45000"))
        )
        st.caption(f"Agent skills: {AGENT_SKILLS_DIR}")

    with st.expander("LLM generation controls", expanded=False):
        llm_temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=env_float("TEMPERATURE", 0.2), step=0.05)
        llm_top_p = st.slider("Top-p", min_value=0.05, max_value=1.0, value=env_float("LLM_TOP_P", 0.9), step=0.05)
        llm_max_tokens = st.number_input("Max output tokens", min_value=512, max_value=32000, step=256, value=env_int("LLM_MAX_TOKENS", 6000))
        llm_timeout_seconds = st.number_input("LLM timeout (seconds)", min_value=15, max_value=900, step=5, value=env_int("LLM_TIMEOUT_SECONDS", 120))
        reasoning_effort = st.selectbox(
            "Reasoning effort",
            ["low", "medium", "high"],
            index=["low", "medium", "high"].index(os.getenv("REASONING_EFFORT", "medium"))
            if os.getenv("REASONING_EFFORT", "medium") in ["low", "medium", "high"]
            else 1,
        )
        ollama_num_ctx = 0
        ollama_top_k = 0
        ollama_repeat_penalty = 0.0
        ollama_num_predict = 0
        if backend == "Local Ollama":
            ollama_num_ctx = st.number_input("Ollama context window", min_value=2048, max_value=262144, step=1024, value=env_int("OLLAMA_NUM_CTX", 32768))
            ollama_top_k = st.number_input("Ollama top-k", min_value=0, max_value=200, value=env_int("OLLAMA_TOP_K", 40))
            ollama_repeat_penalty = st.number_input(
                "Ollama repeat penalty",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                value=env_float("OLLAMA_REPEAT_PENALTY", 1.1),
            )
            ollama_num_predict = st.number_input("Ollama num_predict", min_value=0, max_value=32000, step=256, value=env_int("OLLAMA_NUM_PREDICT", 6000))
        else:
            st.caption("OpenAI-compatible cloud calls use temperature, top-p, max tokens, and reasoning effort where the model supports them.")

    with st.expander("Workspace paths", expanded=False):
        bib_pdf_raw = st.text_input("PDF staging directory", value=os.getenv("BIB_PDF_DIR", "./bib_pdf"))
        pageindex_cache_raw = st.text_input("PageIndex cache directory", value=os.getenv("PAGEINDEX_CACHE_DIR", "./pageindex_cache"))
        compiled_output_raw = st.text_input("Compile output directory", value=os.getenv("COMPILED_OUTPUT_DIR", "./compiled_output"))
        paper_tex_raw = st.text_input("LaTeX source path", value=os.getenv("PAPER_TEX_PATH", "./paper.tex"))

settings = {
    "backend": backend,
    "cloud_provider": cloud_provider,
    "api_key": api_key,
    "openai_base_url": openai_base_url,
    "ollama_base_url": ollama_base_url,
    "model": active_model,
    "retriever": sanitize_retriever(retriever),
    "tavily_api_key": tavily_api_key,
    "embedding_model": embedding_model,
    "llm_temperature": float(llm_temperature),
    "llm_top_p": float(llm_top_p),
    "llm_max_tokens": int(llm_max_tokens),
    "llm_timeout_seconds": int(llm_timeout_seconds),
    "reasoning_effort": reasoning_effort,
    "ollama_num_ctx": int(ollama_num_ctx),
    "ollama_top_k": int(ollama_top_k),
    "ollama_repeat_penalty": float(ollama_repeat_penalty),
    "ollama_num_predict": int(ollama_num_predict),
    "max_search_results_per_query": int(max_search_results_per_query),
    "source_validation_limit": int(source_validation_limit),
    "pdf_download_limit": int(pdf_download_limit),
    "deep_research_breadth": int(deep_research_breadth),
    "deep_research_depth": int(deep_research_depth),
    "deep_research_concurrency": int(deep_research_concurrency),
    "max_scraper_workers": int(max_scraper_workers),
    "scraper_rate_limit_delay": float(scraper_rate_limit_delay),
    "browse_chunk_max_length": int(browse_chunk_max_length),
    "summary_token_limit": int(summary_token_limit),
    "total_words": int(total_words),
    "max_iterations": int(max_iterations),
    "max_subtopics": int(max_subtopics),
    "arxiv_max_results": int(arxiv_max_results),
    "arxiv_page_size": int(arxiv_page_size),
    "arxiv_delay_seconds": float(arxiv_delay_seconds),
    "arxiv_num_retries": int(arxiv_num_retries),
    "language": os.getenv("LANGUAGE", "english"),
    "curate_sources": os.getenv("CURATE_SOURCES", "true").lower() == "true",
    "pageindex_toc_pages": int(pageindex_toc_pages),
    "pageindex_max_pages_per_node": int(pageindex_max_pages_per_node),
    "pageindex_max_tokens_per_node": int(pageindex_max_tokens_per_node),
    "pageindex_llm_timeout_seconds": int(pageindex_llm_timeout_seconds),
    "pageindex_llm_max_retries": int(pageindex_llm_max_retries),
    "pageindex_temperature": float(pageindex_temperature),
    "pageindex_top_p": float(pageindex_top_p),
    "tree_context_budget": int(tree_context_budget),
}

paths = RuntimePaths(
    bib_pdf=resolve_workspace_path(bib_pdf_raw),
    pageindex_cache=resolve_workspace_path(pageindex_cache_raw),
    compiled_output=resolve_workspace_path(compiled_output_raw),
    paper_tex=resolve_workspace_path(paper_tex_raw),
)
created_paper = ensure_bootstrap_files(paths, workspace_title)
sync_latex_session(paths.paper_tex)
missing_modules = missing_runtime_modules(settings)

st.title(workspace_title)
st.caption(f"Active model: {settings['backend']} / {settings['model']} | Paper: {paths.paper_tex.relative_to(ROOT) if paths.paper_tex.is_relative_to(ROOT) else paths.paper_tex}")
st.caption(f"Engines: GPT Researcher `{GPT_RESEARCHER_ENGINE_PATH}` | PageIndex `{PAGEINDEX_ENGINE_PATH}` | Retriever `{effective_retriever(settings)}`")
if created_paper:
    safe_toast(f"Initialized {paths.paper_tex.name}")
if missing_modules:
    st.error(
        "Missing runtime modules: "
        + ", ".join(missing_modules)
        + ". Install the updated requirements and restart Streamlit."
    )

tab_search, tab_chat, tab_latex = st.tabs(
    ["Deep Search & PageIndex Curation", "Vectorless Tree-Reasoning RAG Chat", "Live LaTeX Studio & Local Compiling"]
)


with tab_search:
    left, right = st.columns([0.95, 1.05], gap="large")
    if "selected_report_path" not in st.session_state:
        st.session_state["selected_report_path"] = ""
    if "research_plan_text" not in st.session_state:
        st.session_state["research_plan_text"] = ""
    if "research_plan_editor" not in st.session_state:
        st.session_state["research_plan_editor"] = ""
    if "research_plan_items" not in st.session_state:
        st.session_state["research_plan_items"] = []
    if "research_plan_source" not in st.session_state:
        st.session_state["research_plan_source"] = {}
    with left:
        st.subheader("Agentic Discovery")
        default_query = (
            f"Find primary literature, datasets, benchmark studies, and repositories relevant to {workspace_title}. "
            "Prioritize PDFs, scholarly sources, and reproducible methods."
        )
        query = st.text_area("Deep search query", value=default_query, height=170)
        report_type = st.selectbox(
            "Report type",
            ["research_report", "detailed_report", "deep", "custom_report"],
            index=["research_report", "detailed_report", "deep", "custom_report"].index(os.getenv("GPT_RESEARCHER_REPORT_TYPE", "research_report"))
            if os.getenv("GPT_RESEARCHER_REPORT_TYPE", "research_report") in ["research_report", "detailed_report", "deep", "custom_report"]
            else 0,
        )
        plan_key = {
            "query": query.strip(),
            "report_type": report_type,
            "retriever": effective_retriever(settings),
            "max_results": settings["max_search_results_per_query"],
            "breadth": settings["deep_research_breadth"],
        }
        if st.button("Generate Research Plan", type="primary", width="stretch"):
            if not query.strip():
                st.warning("Enter a query before planning the researcher.")
            else:
                progress_box = st.empty()
                capture = None
                handler = StreamlitResearchLogHandler(progress_box=progress_box)
                try:
                    handler._stage("Preparing planner", "Initializing model, retriever, and scientific search skills.")
                    capture = StreamlitLoggingCaptureHandler(handler)
                    logging.getLogger("research").addHandler(capture)
                    logging.getLogger("gpt_researcher").addHandler(capture)
                    plan_items = run_async(generate_research_plan(query.strip(), report_type, settings, log_handler=handler))
                    if capture:
                        logging.getLogger("research").removeHandler(capture)
                        logging.getLogger("gpt_researcher").removeHandler(capture)
                    set_research_plan_items(plan_items, plan_key)
                    handler._finish("Research plan ready", f"{len(plan_items)} planned search directions.")
                    safe_toast("Research plan generated")
                except Exception as exc:
                    try:
                        if capture:
                            logging.getLogger("research").removeHandler(capture)
                            logging.getLogger("gpt_researcher").removeHandler(capture)
                    except Exception:
                        pass
                    handler._fail(str(exc)[:180])
                    st.error("Research planning failed.")
                    st.code(str(exc), language="text")

        if st.session_state.get("research_plan_items"):
            if st.session_state.get("research_plan_source") != plan_key:
                st.warning("This plan was generated for different query/settings. Regenerate it or review edits carefully.")
            title_cols = st.columns([0.88, 0.12])
            title_cols[0].markdown("#### Planned Research Queries")
            title_cols[1].button("i", help=PLAN_HELP_TEXT, width="stretch")
            st.caption("Edit rows directly. Changes are applied when a field loses focus or the page reruns.")

            for index, item in enumerate(st.session_state.get("research_plan_items", [])):
                query_key = f"plan_query_{index}"
                goal_key = f"plan_goal_{index}"
                if query_key not in st.session_state:
                    st.session_state[query_key] = item.get("query", "")
                if goal_key not in st.session_state:
                    st.session_state[goal_key] = item.get("researchGoal", "") or infer_plan_goal(item.get("query", ""))
                with st.container(border=True):
                    st.markdown(f"**Search direction {index + 1}**")
                    st.text_input(
                        "Query",
                        key=query_key,
                        placeholder='"patent classification" "sustainable development goals" dataset',
                        help="Exact retriever query. Keep concise; Tavily rejects queries over 400 characters.",
                    )
                    st.text_input(
                        "Goal",
                        key=goal_key,
                        placeholder="Find datasets, benchmarks, or primary papers for this query.",
                        help="Purpose of this query. If left blank, the app fills a sensible default from the query terms.",
                    )
            planned_queries = sync_plan_from_row_widgets()
            run_col, clear_col = st.columns([0.72, 0.28])
            with run_col:
                if st.button("Start Research with Approved Plan", type="primary", width="stretch"):
                    if not planned_queries:
                        st.warning("Keep at least one planned query before starting research.")
                    else:
                        progress_box = st.empty()
                        capture = None
                        handler = StreamlitResearchLogHandler(progress_box=progress_box)
                        try:
                            handler._stage("Preparing Deep Search", "Starting from the user-approved research plan.")
                            capture = StreamlitLoggingCaptureHandler(handler)
                            logging.getLogger("research").addHandler(capture)
                            logging.getLogger("gpt_researcher").addHandler(capture)
                            result = run_async(
                                run_gpt_researcher(
                                    query.strip(),
                                    report_type,
                                    settings,
                                    paths,
                                    log_handler=handler,
                                    planned_queries=planned_queries,
                                )
                            )
                            if capture:
                                logging.getLogger("research").removeHandler(capture)
                                logging.getLogger("gpt_researcher").removeHandler(capture)
                            handler._stage("Saving report and acquired PDFs", "Writing markdown report and downloading discovered PDFs when available.")
                            handler._finish("Deep Search complete", f"Saved report: {Path(result['report_path']).name}")
                            safe_toast("Deep literature search complete")
                            st.success(f"Report saved to {result['report_path']}")
                            st.session_state["selected_report_path"] = str(result["report_path"])
                            st.session_state["last_deep_search_report"] = result.get("report", "")
                            st.write(f"Discovered URLs scanned for PDFs: {result['url_count']}")
                            st.write(f"Verified reachable source links: {result['verified_url_count']}")
                            if result["downloaded"]:
                                st.write("Downloaded PDFs")
                                st.dataframe(
                                    [{"file": path.name, "path": str(path)} for path in result["downloaded"]],
                                    width="stretch",
                                )
                        except Exception as exc:
                            try:
                                if capture:
                                    logging.getLogger("research").removeHandler(capture)
                                    logging.getLogger("gpt_researcher").removeHandler(capture)
                            except Exception:
                                pass
                            handler._fail(str(exc)[:180])
                            st.error("Deep Search failed.")
                            st.code(str(exc), language="text")
            with clear_col:
                st.button("Clear Plan", width="stretch", on_click=clear_research_plan_state)
            st.button("Add Query Row", width="stretch", on_click=append_blank_plan_item)

    with right:
        st.subheader("Structural Staging Room")
        pdfs = discover_pdfs(paths.bib_pdf)
        cache_payloads = load_index_payloads(paths)
        metrics = st.columns(3)
        metrics[0].metric("Staged PDFs", len(pdfs))
        metrics[1].metric("Indexed trees", len(cache_payloads))
        metrics[2].metric("Reports", len(list(paths.bib_pdf.glob("*.md"))) if paths.bib_pdf.exists() else 0)

        if pdfs:
            st.dataframe(
                [
                    {
                        "file": pdf.name,
                        "MB": round(pdf.stat().st_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                    }
                    for pdf in pdfs
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(f"No PDFs found in {paths.bib_pdf}")

        if st.button("🔄 Re-Index Workspace Base via PageIndex", type="primary", width="stretch"):
            if not pdfs:
                st.warning("Add PDFs to the staging directory before indexing.")
            else:
                with st.spinner("PageIndex is building semantic section trees..."):
                    results = reindex_workspace(settings, paths)
                    indexed = sum(1 for item in results if item["status"] == "indexed")
                    cached = sum(1 for item in results if item["status"] == "cached")
                    errors = [item for item in results if item["status"] == "error"]
                    if errors:
                        st.error(f"PageIndex finished with {len(errors)} error(s). Check terminal logs for detailed traces.")
                    else:
                        safe_toast("PageIndex tree indices locked and loaded")
                        st.success(f"Indexed {indexed} PDF(s); reused {cached} cached tree(s).")
                    st.caption("Detailed PageIndex execution logs are printed to terminal and persisted under `./logs/`.")
                    st.dataframe(
                        [
                            {
                                "status": item["status"],
                                "pdf": item["pdf"].name,
                                "cache": str(item["output"]) if item.get("output") else "",
                                "error": item.get("error", ""),
                                "log": item.get("log", ""),
                            }
                            for item in results
                        ],
                        width="stretch",
                        hide_index=True,
                    )

    # Full-width report viewer (persists across tab switches)
    st.divider()
    st.subheader("Research Reports")
    report_files = list_saved_reports(paths.bib_pdf)
    report_labels = ["(none)"] + [f"{p.name}" for p in report_files]
    default_selected = st.session_state.get("selected_report_path", "")
    default_name = Path(default_selected).name if default_selected else ""
    default_index = 0
    if default_name:
        for i, p in enumerate(report_files, start=1):
            if p.name == default_name:
                default_index = i
                break
    selected_label = st.selectbox("Select report", report_labels, index=default_index)
    selected_path = ""
    if selected_label != "(none)":
        selected_path = str(paths.bib_pdf / selected_label)
        st.session_state["selected_report_path"] = selected_path

    if selected_path:
        report_text = read_text_file(Path(selected_path))
        st.markdown(report_text)
    else:
        st.info("Run Deep Search to generate a report, or select an existing markdown report from the list.")


with tab_chat:
    st.subheader("Workspace Cross-Examination")
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "pending_chat_question" not in st.session_state:
        st.session_state["pending_chat_question"] = None
    if "rag_chat_draft" not in st.session_state:
        st.session_state["rag_chat_draft"] = ""

    toolbar_cols = st.columns([0.24, 0.76])
    if toolbar_cols[0].button("Clear conversation", width="stretch"):
        st.session_state["chat_messages"] = []
        st.session_state["pending_chat_question"] = None
        st.session_state["rag_chat_draft"] = ""
        st.rerun()

    with st.container(border=True):
        st.markdown("**Ask the indexed workspace**")
        with st.form("rag_chat_composer", clear_on_submit=True):
            st.text_area(
                "Question",
                key="rag_chat_draft",
                height=130,
                placeholder="Ask a high-precision question. Example: Compare weak supervision vs zero-shot methods and cite exact sections/pages.",
                label_visibility="collapsed",
            )
            submit_cols = st.columns([0.22, 0.78])
            submitted = submit_cols[0].form_submit_button("Ask", type="primary", width="stretch")
            if submitted:
                candidate = st.session_state.get("rag_chat_draft", "").strip()
                if candidate:
                    st.session_state["pending_chat_question"] = candidate
                    st.rerun()

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending_question = st.session_state.get("pending_chat_question")
    if pending_question:
        st.session_state["pending_chat_question"] = None
        st.session_state["chat_messages"].append({"role": "user", "content": pending_question})
        with st.chat_message("user"):
            st.markdown(pending_question)
        with st.chat_message("assistant"):
            with st.spinner("Reasoning over PageIndex section trees..."):
                try:
                    answer = answer_workspace_question(pending_question, settings, paths, st.session_state["chat_messages"])
                    st.markdown(answer)
                except Exception as exc:
                    answer = f"RAG chat failed:\n\n```text\n{exc}\n```"
                    st.markdown(answer)
        st.session_state["chat_messages"].append({"role": "assistant", "content": answer})


with tab_latex:
    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        st.subheader("AI Co-Author")
        modify_instruction = st.text_area(
            "Ask AI to modify your active LaTeX source directly",
            height=115,
        )
        if st.button("Apply AI Edit To Source", type="primary", width="stretch"):
            save_editor_if_changed(paths)
            if not modify_instruction.strip():
                st.warning("Enter an editing instruction first.")
            else:
                with st.spinner("The co-author is rewriting the LaTeX source..."):
                    try:
                        updated = modify_latex_source(st.session_state["latex_editor"], modify_instruction.strip(), settings)
                        atomic_write_text(paths.paper_tex, updated)
                        st.session_state["latex_editor"] = updated
                        st.session_state["latex_saved_sha"] = sha256_text(updated)
                        safe_toast("LaTeX source updated")
                        st.rerun()
                    except Exception as exc:
                        st.error("AI edit failed.")
                        st.code(str(exc), language="text")

        st.text_area("Active LaTeX source", key="latex_editor", height=720)
        save_editor_if_changed(paths)

    with right:
        st.subheader("Local Compilation")
        st.write(f"Source: `{paths.paper_tex}`")
        st.write(f"Output: `{paths.compiled_output}`")
        if st.button("🚀 Compile Document", type="primary", width="stretch"):
            save_editor_if_changed(paths)
            with st.spinner("Running pdflatex twice..."):
                result = compile_latex(paths)
            if result["ok"]:
                st.success(f"Compilation succeeded: {result['pdf_path']}")
            else:
                st.error("Compilation failed.")
                st.code(result.get("log", ""), language="text")

        pdf_path = paths.compiled_output / f"{paths.paper_tex.stem}.pdf"
        if pdf_path.exists():
            st.download_button(
                "Open compiled PDF bytes",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                width="stretch",
            )

        manifest_path = paths.pageindex_cache / "workspace_index.json"
        if manifest_path.exists():
            with st.expander("Current PageIndex manifest", expanded=False):
                st.json(json.loads(manifest_path.read_text(encoding="utf-8")))
