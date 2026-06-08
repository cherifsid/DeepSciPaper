from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
import streamlit.components.v1 as components
from dotenv import load_dotenv

from case_management import (
    append_chat_message as persist_chat_message,
    case_paths,
    clear_chat_messages as clear_case_chat_history,
    create_case as create_managed_case,
    db_path as case_db_path,
    ensure_legacy_case,
    get_case_by_slug,
    init_db as init_case_db,
    latest_sync_event,
    list_cases as list_managed_cases,
    list_chat_messages as load_case_chat_messages,
    minio_is_configured,
    sync_case_to_minio,
    update_case_metadata,
)


load_dotenv()

ROOT = Path(__file__).resolve().parent
GPT_RESEARCHER_ENGINE_PATH = (ROOT / os.getenv("GPT_RESEARCHER_ENGINE_PATH", "./engines/gpt-researcher")).resolve()
AGENT_SKILLS_DIR = (ROOT / os.getenv("AGENT_SKILLS_DIR", "./agents/skills")).resolve()


def activate_local_engine_paths() -> None:
    engine_specs = [
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
FIXED_MULTIMODAL_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
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
This living manuscript captures a reproducible research synthesis for the workspace titled ``{latex_escape(title_for_text)}.'' The document is designed to be edited continuously as new literature is discovered, parsed into multimodal artifacts, and cross-examined through the local Research Copilot environment.
\end{{abstract}}

\section{{Research Aim}}
The project investigates methods, evidence, and evaluation protocols for {latex_escape(workspace_title.lower())}. The manuscript should preserve traceable claims, clearly separate empirical findings from interpretation, and cite supporting sources as the workspace evidence base grows.

\section{{Evidence Base}}
The evidence base is maintained outside this manuscript in the workspace directories. Primary literature PDFs are staged in \texttt{{bib\_pdf/}}, multimodal MinerU artifacts and local Qdrant indexes are stored in \texttt{{multimodal\_store/}}, and compiled artifacts are written to \texttt{{compiled\_output/}}.

\section{{Methodological Notes}}
The intended retrieval path is agentic and multimodal. Documents are parsed into Markdown, raw JSON, tables, figures, captions, and local embedding records. The co-authoring workflow should use those artifact references to ground synthesis, comparisons, and claims.

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


def runtime_preflight(
    settings: dict[str, Any],
    paths: RuntimePaths,
    *,
    multimodal_store_path: Path | None = None,
    graph_store_path: Path | None = None,
    require_pdfs: bool = False,
    require_multimodal_index: bool = False,
    require_graph_index: bool = False,
    require_paper: bool = False,
) -> list[str]:
    issues: list[str] = []
    if not settings.get("model", "").strip():
        issues.append("Select a model name in the sidebar.")
    if settings.get("backend") == "Local Ollama":
        if not settings.get("ollama_base_url", "").strip():
            issues.append("Set the Ollama base URL in the sidebar.")
    else:
        if not settings.get("api_key", "").strip():
            provider = settings.get("cloud_provider", "cloud provider")
            issues.append(f"Add a {provider} API key in the sidebar.")

    if require_pdfs and not discover_pdfs(paths.bib_pdf):
        issues.append(f"No PDFs were found in `{paths.bib_pdf}`.")
    if require_multimodal_index and multimodal_store_path is not None:
        has_records = any(multimodal_store_path.glob("indexes/*_records.json"))
        if not has_records:
            issues.append("The multimodal index is empty. Click Index before asking Standard RAG questions.")
    if require_graph_index and graph_store_path is not None:
        has_graphs = any(graph_store_path.glob("graphs/*/graph.json"))
        if not has_graphs:
            issues.append("The graph RAG store is empty. Run Index before using Research Mode.")
    if require_paper and not paths.paper_tex.exists():
        issues.append(f"The LaTeX source file is missing: `{paths.paper_tex}`.")

    return issues


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


def strip_ollama_tag(model: str) -> str:
    model = strip_provider_prefix(model.strip(), "ollama")
    if model.startswith("ollama/"):
        return model.split("/", 1)[1]
    return model


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
    model_name = settings.get("model", "").strip().lower()
    supports_reasoning_effort = any(
        token in model_name
        for token in ("gpt-5", "o1", "o3", "o4")
    )
    if settings.get("backend") == "Cloud API" and settings.get("cloud_provider") == "OpenAI" and supports_reasoning_effort:
        kwargs["reasoning_effort"] = settings.get("reasoning_effort", "medium")
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
        error_text = str(litellm_error)
        if "reasoning_effort" in kwargs and ("unexpected keyword" in error_text.lower() or "unsupported" in error_text.lower()):
            kwargs.pop("reasoning_effort", None)
            try:
                response = completion(**kwargs)
                text = extract_llm_response_text(response)
                if text:
                    return text
            except Exception as retry_exc:
                litellm_error = retry_exc
        raise RuntimeError(str(litellm_error)) from litellm_error
    raise RuntimeError("Model returned an empty completion.")


def discover_pdfs(bib_pdf_dir: Path) -> list[Path]:
    if not bib_pdf_dir.exists():
        return []
    return sorted(path for path in bib_pdf_dir.glob("*.pdf") if path.is_file())


def summarize_multimodal_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in results:
        rows.append(
            {
                "status": item.get("status"),
                "pdf": Path(item.get("source_pdf", "")).name,
                "parser": item.get("parser", ""),
                "chunks": item.get("text_chunks", 0),
                "tables": item.get("tables", 0),
                "images": item.get("images", 0),
                "vectors": item.get("vector_records", 0),
                "error": item.get("error", ""),
            }
        )
    return rows


def render_artifacts(artifact_references: list[dict[str, Any]]) -> None:
    if not artifact_references:
        return
    with st.expander("Image and table artifacts", expanded=False):
        st.dataframe(artifact_references, width="stretch", hide_index=True)
        image_refs = [
            item
            for item in artifact_references
            if (item.get("modality") == "image" or item.get("node_type") == "figure") and item.get("asset_path")
        ]
        if image_refs:
            image_cols = st.columns(2)
            for index, item in enumerate(image_refs[:8]):
                path = Path(str(item.get("asset_path", "")))
                if path.exists():
                    image_cols[index % 2].image(
                        str(path),
                        caption=f"{item.get('source_pdf_name', '')} | {item.get('title', '')}",
                        width="stretch",
                    )


def clean_markdown_for_display(text: str) -> str:
    clean = html.unescape(text or "")
    clean = re.sub(r"!\[[^\]]*]\(([^)]+)\)", "", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def render_markdown_block(text: str) -> None:
    st.markdown(clean_markdown_for_display(text), unsafe_allow_html=True)


def pdf_embed_html(pdf_path: Path, height: int = 440) -> str:
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    return (
        f'<div style="height:{height}px; overflow:auto; border:1px solid rgba(128,128,128,0.2); border-radius:10px; background:#fff;">'
        f'<object data="data:application/pdf;base64,{encoded}" type="application/pdf" '
        f'width="100%" height="{max(height - 8, 240)}" style="display:block;">'
        f'<embed src="data:application/pdf;base64,{encoded}" type="application/pdf" width="100%" height="{max(height - 8, 240)}"></embed>'
        "</object>"
        "</div>"
    )


def render_pdf_preview(pdf_path: Path, max_pages: int = 2) -> tuple[list[bytes], int]:
    import fitz

    previews: list[bytes] = []
    with fitz.open(pdf_path) as doc:
        page_total = len(doc)
        for page_index in range(min(max_pages, page_total)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
            previews.append(pix.tobytes("png"))
    return previews, page_total


def request_workspace_tab_switch(tab_name: str) -> None:
    st.session_state["workspace_tab_switch"] = tab_name


def maybe_switch_workspace_tab() -> None:
    target = st.session_state.pop("workspace_tab_switch", "")
    if not target:
        return
    escaped = json.dumps(target)
    components.html(
        f"""
        <script>
        const target = {escaped};
        const doc = window.parent.document;
        const clickTab = () => {{
          const buttons = Array.from(doc.querySelectorAll('button[role="tab"]'));
          const match = buttons.find((btn) => (btn.innerText || '').trim() === target);
          if (match) {{
            match.click();
          }}
        }};
        clickTab();
        setTimeout(clickTab, 120);
        setTimeout(clickTab, 360);
        </script>
        """,
        height=0,
        width=0,
    )


def resolve_case_pdf_path(bib_pdf_dir: Path, source_pdf_name: str) -> Path | None:
    if not source_pdf_name:
        return None
    candidate = bib_pdf_dir / source_pdf_name
    return candidate if candidate.exists() else None


def render_pdf_resource_popover(
    *,
    label: str,
    pdf_path: Path | None,
    section: str = "",
    title: str = "",
    snippet: str = "",
    asset_path: str = "",
    source_url: str = "",
    key_hint: str = "",
) -> None:
    with st.popover(label):
        if title:
            st.markdown(f"**{title}**")
        if section:
            st.caption(f"Section: {section}")
        if source_url:
            st.link_button("Open source link", source_url, width="stretch")
        if snippet:
            st.markdown(clean_markdown_for_display(snippet[:1600]), unsafe_allow_html=True)
        asset = Path(asset_path) if asset_path else None
        if asset and asset.exists() and asset.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            st.image(str(asset), caption=asset.name, width="stretch")
        elif asset and asset.exists():
            st.caption(f"Linked artifact: `{asset.name}`")
        elif asset_path:
            st.caption(f"Linked artifact path: `{asset_path}`")
        if pdf_path and pdf_path.exists():
            try:
                previews, page_total = render_pdf_preview(pdf_path, max_pages=3)
                st.caption(f"PDF preview · {page_total} page(s)")
                for page_number, preview in enumerate(previews, start=1):
                    st.image(preview, caption=f"Page {page_number}", width="stretch")
            except Exception as exc:
                st.info("Inline PDF preview is unavailable for this evidence item. Use the download action below.")
                with st.expander("Preview diagnostic", expanded=False):
                    st.code(str(exc), language="text")
            st.download_button(
                "Download PDF",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                key=f"download_pdf_{key_hint}_{pdf_path.name}",
                width="stretch",
            )
            st.caption(str(pdf_path))
        else:
            st.info("No local PDF preview is available for this evidence item.")


def render_evidence_resources(
    evidence_references: list[dict[str, Any]],
    *,
    bib_pdf_dir: Path,
    key_prefix: str,
    title: str = "Evidence resources",
) -> None:
    if not evidence_references:
        return
    st.markdown(f"#### {title}")
    for item in evidence_references:
        evidence_id = str(item.get("evidence_id") or f"E{item.get('index', '')}").strip()
        source_pdf_name = str(item.get("source_pdf_name", "")).strip()
        pdf_path = resolve_case_pdf_path(bib_pdf_dir, source_pdf_name)
        display_label = evidence_id or source_pdf_name or "Evidence"
        subtitle = item.get("section") or item.get("section_title") or item.get("title") or ""
        row_cols = st.columns([0.24, 0.76], gap="small")
        row_cols[0].markdown(f"**{display_label}**")
        with row_cols[1]:
            render_pdf_resource_popover(
                label=f"View {display_label}",
                pdf_path=pdf_path,
                section=str(item.get("section") or item.get("section_title") or ""),
                title=str(item.get("title") or item.get("paper_title") or source_pdf_name),
                snippet=str(item.get("content") or ""),
                asset_path=str(item.get("asset_path") or ""),
                source_url=str(item.get("source_url") or ""),
                key_hint=f"{key_prefix}_{display_label}",
            )
            if subtitle:
                st.caption(f"{source_pdf_name} | {subtitle}")
            elif source_pdf_name:
                st.caption(source_pdf_name)


def report_resource_links(report_text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for url in extract_urls(report_text):
        clean = normalize_url(url)
        if clean and clean not in seen:
            seen.add(clean)
            urls.append(clean)
    return urls


def render_report_resource_explorer(report_text: str, bib_pdf_dir: Path, key_prefix: str) -> None:
    pdfs = discover_pdfs(bib_pdf_dir)
    urls = report_resource_links(report_text)
    if not pdfs and not urls:
        return
    st.markdown("#### Evidence and source explorer")
    if pdfs:
        st.caption("Local PDFs")
        local_cols = st.columns(3)
        for index, pdf_path in enumerate(pdfs[:18]):
            with local_cols[index % 3]:
                render_pdf_resource_popover(
                    label=f"Open {pdf_path.stem[:28]}",
                    pdf_path=pdf_path,
                    title=pdf_path.name,
                    key_hint=f"{key_prefix}_pdf_{index}",
                )
    if urls:
        st.caption("Verified source links")
        for url in urls[:20]:
            with st.popover(f"Source: {urlparse(url).netloc}"):
                st.link_button("Open source", url, width="stretch")
                st.code(url, language="text")


def normalize_graph_artifacts(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for item in evidence_items:
        node_type = str(item.get("node_type", "")).lower()
        modality = "image" if node_type == "figure" else ("table" if node_type == "table" else node_type)
        if modality not in {"image", "table"}:
            continue
        artifacts.append(
            {
                "index": item.get("evidence_id"),
                "modality": modality,
                "node_type": node_type,
                "source_pdf_name": item.get("source_pdf_name"),
                "asset_path": item.get("asset_path"),
                "score": item.get("score"),
                "title": item.get("section") or item.get("paper_title") or item.get("node_id"),
                "section": item.get("section", ""),
                "content": item.get("content", ""),
            }
        )
    return artifacts


def build_multimodal_evidence_bundle(evidence: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_lines: list[str] = []
    evidence_references: list[dict[str, Any]] = []
    for index, item in enumerate(evidence, start=1):
        score_value = item.get("score", 0.0)
        try:
            score = f"{float(score_value):.3f}"
        except Exception:
            score = str(score_value)
        evidence_lines.append(
            (
                f"[E{index}] modality={item.get('modality')} score={score} "
                f"source={item.get('source_pdf_name')} section={item.get('section_title', '')} "
                f"title={item.get('title')} asset={item.get('asset_path')}\n"
                f"{item.get('display_text', '')}"
            ).strip()
        )
        evidence_references.append(
            {
                "index": index,
                "modality": item.get("modality"),
                "source_pdf_name": item.get("source_pdf_name"),
                "section_title": item.get("section_title", ""),
                "asset_path": item.get("asset_path"),
                "score": item.get("score"),
                "title": item.get("title"),
            }
        )
    artifact_references = [row for row in evidence_references if row.get("modality") in {"table", "image"}]
    return "\n\n".join(evidence_lines), evidence_references, artifact_references


def multimodal_chat_answer(
    question: str,
    settings: dict[str, Any],
    multimodal_store_path: Path,
) -> dict[str, Any]:
    from multimodal_pipeline import search_records, store_paths

    limit = int(settings.get("multimodal_retrieval_limit", 10))
    evidence = search_records(
        question,
        store_paths(multimodal_store_path),
        embed_model=FIXED_MULTIMODAL_EMBED_MODEL,
        ollama_base_url=settings.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
        limit=limit,
    )
    evidence_text, evidence_references, artifact_references = build_multimodal_evidence_bundle(evidence)
    system_prompt = build_rag_role_prompt()
    user_prompt = (
        "Answer the user question using only the retrieved workspace evidence.\n"
        "Required behavior:\n"
        "1. Cite factual claims with evidence IDs like [E1], [E2].\n"
        "2. Mention table/image artifacts only when present in evidence.\n"
        "3. If evidence is insufficient or conflicting, say so explicitly.\n"
        "4. Keep output as clean Markdown with concise sections.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{evidence_text}"
    )
    answer = llm_complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        settings,
    )
    return {
        "answer": answer.strip(),
        "answer_markdown": answer.strip(),
        "evidence": evidence,
        "evidence_references": evidence_references,
        "artifact_references": artifact_references,
    }


def graph_chat_answer(
    question: str,
    settings: dict[str, Any],
    graph_store_path: Path,
) -> dict[str, Any]:
    from scientific_graph_rag import run_query

    backend = "ollama" if settings.get("backend") == "Local Ollama" else "cloud"
    cloud_provider = "deepseek" if settings.get("cloud_provider") == "DeepSeek" else "openai"
    response = run_query(
        question=question,
        graph_store=graph_store_path,
        backend=backend,
        model=str(settings.get("model", "")).strip(),
        ollama_base_url=settings.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
        cloud_provider=cloud_provider,
        api_key=settings.get("api_key", "").strip(),
        base_url=settings.get("openai_base_url", "").strip(),
        max_queries=4,
        top_k_per_query=max(6, int(settings.get("multimodal_retrieval_limit", 10))),
    )
    artifacts = normalize_graph_artifacts(response.get("evidence_items", []))
    return {
        "answer": response.get("answer_markdown", "").strip(),
        "answer_markdown": response.get("answer_markdown", "").strip(),
        "evidence": response.get("evidence_items", []),
        "evidence_references": response.get("evidence_items", []),
        "artifact_references": artifacts,
        "query_plan": response.get("query_plan", []),
        "synthesis_error": response.get("synthesis_error", ""),
    }


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
    rag_skill = load_skill_file(AGENT_SKILLS_DIR, "multimodal_synthesizer_chat.md")
    if not rag_skill:
        return (
            "You are a multimodal scientific research copilot. Reason over retrieved text, tables, figures, and artifact references. "
            "Cite all document-grounded claims with evidence IDs and artifact paths when available."
        )
    return f"You are a multimodal scientific research copilot.\n\n{rag_skill}"


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


def clean_latex_response(text: str) -> str:
    clean = text.strip()
    clean = re.sub(r"^```(?:latex|tex)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    match = re.search(r"\\documentclass[\s\S]*?\\end\{document\}", clean)
    if match:
        return match.group(0).strip() + "\n"
    return clean.strip() + "\n"


def structurally_validate_latex(source: str) -> str | None:
    required_tokens = [
        "\\documentclass",
        "\\begin{document}",
        "\\end{document}",
    ]
    for token in required_tokens:
        if token not in source:
            return f"Missing required LaTeX token: {token}"
    if source.find("\\begin{document}") > source.find("\\end{document}"):
        return "The document body markers are out of order."
    return None


def validate_latex_candidate(candidate_source: str, paths: RuntimePaths) -> tuple[bool, str]:
    structural_issue = structurally_validate_latex(candidate_source)
    if structural_issue:
        return False, structural_issue

    if shutil.which("pdflatex") is None:
        return True, "pdflatex not available; structural validation only."

    temp_name = f".{paths.paper_tex.stem}.ai_validation.tex"
    temp_source_path = paths.paper_tex.parent / temp_name
    output_dir_path: Path | None = None
    try:
        atomic_write_text(temp_source_path, candidate_source)
        with tempfile.TemporaryDirectory(prefix="latex-validate-") as tmpdir:
            output_dir_path = Path(tmpdir)
            command = [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={tmpdir}",
                temp_name,
            ]
            result = subprocess.run(
                command,
                cwd=str(paths.paper_tex.parent),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if result.returncode != 0:
                log = (result.stdout or "") + "\n" + (result.stderr or "")
                return False, log.strip()[-4000:]
            return True, "Validation compile succeeded."
    except subprocess.TimeoutExpired as exc:
        return False, f"LaTeX validation timed out: {exc}"
    finally:
        try:
            temp_source_path.unlink(missing_ok=True)
        except Exception:
            pass
        if output_dir_path is not None:
            for suffix in (".aux", ".log", ".out", ".pdf", ".toc"):
                try:
                    (output_dir_path / f"{Path(temp_name).stem}{suffix}").unlink(missing_ok=True)
                except Exception:
                    pass


def modify_latex_source(current_source: str, instruction: str, settings: dict[str, Any], paths: RuntimePaths) -> str:
    system_prompt = (
        "You are a careful LaTeX co-author working on a scientific paper. "
        "You must read the entire source before editing it. "
        "Apply the user request directly to the provided source and return only one complete compilable LaTeX document, "
        "from \\documentclass through \\end{document}. "
        "Do not omit sections, packages, macros, bibliography commands, labels, or environments unless the user explicitly asks for removal. "
        "Keep indentation tidy, preserve existing structure where possible, and never add markdown fences or explanations."
    )
    validation_feedback = ""
    for attempt in range(1, 3):
        user_prompt = (
            f"User modification request:\n{instruction}\n\n"
            "Current full LaTeX source to edit:\n"
            f"{current_source}"
        )
        if validation_feedback:
            user_prompt += (
                "\n\nThe previous LaTeX candidate failed validation. "
                "Repair it and return a corrected full document only.\n"
                f"Validation feedback:\n{validation_feedback}"
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        candidate = clean_latex_response(llm_complete(messages, settings))
        is_valid, feedback = validate_latex_candidate(candidate, paths)
        if is_valid:
            return candidate
        validation_feedback = feedback

    raise RuntimeError(
        "The model returned LaTeX that failed validation twice.\n\n"
        f"Validation feedback:\n{validation_feedback}"
    )


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


def read_json_file(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def sync_chat_session_from_db(database_path: Path, case_id: int) -> None:
    st.session_state["chat_messages"] = load_case_chat_messages(database_path, case_id)
    st.session_state["chat_pending_question"] = ""
    st.session_state["chat_pending_mode"] = "standard"
    st.session_state["active_case_id"] = case_id


def maybe_auto_sync_case(database_path: Path, case_record: Any) -> tuple[bool, str]:
    if not minio_is_configured():
        return False, ""
    try:
        sync_case_to_minio(database_path, case_record)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def case_report_rows(case_record: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in list_saved_reports(case_paths(case_record)["bib_pdf"]):
        rows.append(
            {
                "report": path.name,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "chars": path.stat().st_size,
            }
        )
    return rows


def case_graph_rows(case_record: Any) -> list[dict[str, Any]]:
    graph_root = case_paths(case_record)["graph_store"] / "manifests"
    rows: list[dict[str, Any]] = []
    for path in sorted(graph_root.glob("*.json")) if graph_root.exists() else []:
        payload = read_json_file(path, {})
        rows.append(
            {
                "paper": payload.get("source_pdf_name") or payload.get("doc_id") or path.stem,
                "doc_id": payload.get("doc_id", path.stem),
                "status": payload.get("status", ""),
                "indexed_at": payload.get("indexed_at", ""),
            }
        )
    return rows


def case_latex_rows(case_record: Any) -> list[dict[str, Any]]:
    paths = case_paths(case_record)
    rows: list[dict[str, Any]] = []
    paper_path = paths["paper_tex"]
    if paper_path.exists():
        rows.append(
            {
                "artifact": paper_path.name,
                "type": "source",
                "modified": datetime.fromtimestamp(paper_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    compiled_root = paths["compiled_output"]
    if compiled_root.exists():
        for pdf_path in sorted(compiled_root.glob("*.pdf")):
            rows.append(
                {
                    "artifact": pdf_path.name,
                    "type": "compiled_pdf",
                    "modified": datetime.fromtimestamp(pdf_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                }
            )
    return rows


def render_login_screen() -> None:
    _, center, _ = st.columns([0.22, 0.56, 0.22])
    with center:
        st.title("Research Copilot")
        st.subheader("Sign in")
        st.caption("Use the demo credentials to enter the research workspace.")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", value="")
            password = st.text_input("Password", value="", type="password")
            submitted = st.form_submit_button("Login", type="primary", width="stretch")
        if submitted:
            if username == "admin" and password == "admin":
                st.session_state["is_authenticated"] = True
                st.session_state["auth_user"] = "admin"
                st.session_state.setdefault("app_view", "home")
                st.rerun()
            st.error("Invalid credentials. Use `admin` / `admin`.")
    st.stop()


def render_home_page(database_path: Path, selected_case: Any, all_cases: list[Any]) -> None:
    selected_paths = case_paths(selected_case)
    staged_pdfs = discover_pdfs(selected_paths["bib_pdf"])
    report_rows = case_report_rows(selected_case)
    graph_rows = case_graph_rows(selected_case)
    latex_rows = case_latex_rows(selected_case)
    sync_event = latest_sync_event(database_path, selected_case.id)

    st.title("Welcome")
    st.write(
        "This home space gives you a stable entry point into the research workspace, the legacy artifacts already in the repo, "
        "and the new case-management layer for fresh studies."
    )

    action_cols = st.columns([0.34, 0.33, 0.33])
    if action_cols[0].button("Open Deep Search Workspace", type="primary", width="stretch"):
        st.session_state["app_view"] = "workspace"
        st.rerun()
    if action_cols[1].button("Create New Case", width="stretch"):
        st.session_state["app_view"] = "new_case"
        st.rerun()
    if action_cols[2].button("Sync Current Case To MinIO", width="stretch"):
        with st.spinner("Syncing case artifacts to MinIO..."):
            try:
                result = sync_case_to_minio(database_path, selected_case)
                safe_toast(f"Synced {result['uploaded']} objects to MinIO")
                st.rerun()
            except Exception as exc:
                st.error(f"MinIO sync failed: {exc}")

    metrics = st.columns(5)
    metrics[0].metric("Cases", len(all_cases))
    metrics[1].metric("Staged PDFs", len(staged_pdfs))
    metrics[2].metric("Reports", len(report_rows))
    metrics[3].metric("Graph docs", len(graph_rows))
    metrics[4].metric("LaTeX artifacts", len(latex_rows))

    with st.container(border=True):
        st.markdown(f"### Active Case: {selected_case.name}")
        if selected_case.description:
            st.write(selected_case.description)
        if selected_case.research_goal:
            st.caption(f"Research goal: {selected_case.research_goal}")
        if sync_event:
            st.caption(
                f"Last MinIO sync: {sync_event.get('created_at')} | {sync_event.get('status')} | "
                f"{sync_event.get('artifact_count')} objects"
            )

    with st.expander("Edit case details", expanded=False):
        with st.form("edit_case_details"):
            edited_name = st.text_input("Case name", value=selected_case.name)
            edited_description = st.text_area("Description", value=selected_case.description, height=100)
            edited_goal = st.text_area("Research goal", value=selected_case.research_goal, height=90)
            if st.form_submit_button("Save case details", type="primary", width="stretch"):
                update_case_metadata(
                    database_path,
                    selected_case.id,
                    name=edited_name,
                    description=edited_description,
                    research_goal=edited_goal,
                )
                safe_toast("Case details updated")
                st.rerun()

    preview_left, preview_right = st.columns([1, 1], gap="large")
    with preview_left:
        st.markdown("### Legacy Reports")
        if report_rows:
            st.dataframe(report_rows, width="stretch", hide_index=True)
        else:
            st.info("No deep-search reports found for this case.")

        st.markdown("### LaTeX Representations")
        if latex_rows:
            st.dataframe(latex_rows, width="stretch", hide_index=True)
        else:
            st.info("No LaTeX source or compiled PDF was found for this case.")

    with preview_right:
        st.markdown("### Graph Artifacts")
        if graph_rows:
            st.dataframe(graph_rows, width="stretch", hide_index=True)
        else:
            st.info("No graph knowledge-store manifests were found for this case.")

        st.markdown("### Case Paths")
        st.code(
            "\n".join(
                [
                    f"root: {selected_paths['root']}",
                    f"bib_pdf: {selected_paths['bib_pdf']}",
                    f"multimodal_store: {selected_paths['multimodal_store']}",
                    f"graph_store: {selected_paths['graph_store']}",
                    f"compiled_output: {selected_paths['compiled_output']}",
                    f"paper_tex: {selected_paths['paper_tex']}",
                ]
            ),
            language="text",
        )


def render_new_case_page(database_path: Path, root: Path) -> None:
    st.title("Create A New Case")
    st.write("Each case gets its own PDF staging area, multimodal index store, graph store, compile output, and LaTeX source.")

    with st.form("new_case_form"):
        case_name = st.text_input("Case name", value="")
        case_description = st.text_area("Description", height=110)
        case_goal = st.text_area("Research goal", height=110)
        submitted = st.form_submit_button("Create case", type="primary", width="stretch")

    if submitted:
        if not case_name.strip():
            st.warning("Add a case name first.")
        else:
            new_case = create_managed_case(
                database_path,
                root=root,
                name=case_name.strip(),
                description=case_description.strip(),
                research_goal=case_goal.strip(),
            )
            st.session_state["selected_case_slug"] = new_case.slug
            st.session_state["app_view"] = "workspace"
            sync_chat_session_from_db(database_path, new_case.id)
            safe_toast(f"Created case: {new_case.name}")
            st.rerun()


def missing_runtime_modules(settings: dict[str, Any]) -> list[str]:
    required_modules = [
        "gpt_researcher",
        "litellm",
        "openai",
        "dotenv",
        "fitz",
        "PyPDF2",
        "requests",
        "qdrant_client",
        "mineru",
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
    div[data-testid="stChatMessage"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.65rem;
        background: var(--secondary-background-color);
    }
    div[data-testid="stChatMessage"] table {
        width: 100%;
        font-size: 0.92rem;
    }
    div[data-testid="stChatMessage"] th, div[data-testid="stChatMessage"] td {
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding: 0.42rem 0.5rem;
        vertical-align: top;
    }
    div[data-testid="stChatMessage"] pre {
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.22);
    }
    .st-key-multimodal_chat_input {
        position: sticky;
        bottom: 0;
        z-index: 30;
        background: var(--background-color);
        padding-top: 0.45rem;
        padding-bottom: 0.2rem;
        border-top: 1px solid rgba(128, 128, 128, 0.18);
    }
    .st-key-multimodal_chat_input textarea {
        padding-right: 14.5rem !important;
    }
    .st-key-chat_mode_selector_inline {
        position: relative;
        z-index: 45;
        margin-top: -4.0rem;
        margin-bottom: 0.2rem;
        height: 0;
        pointer-events: none;
        display: flex !important;
        justify-content: flex-end;
    }
    .st-key-chat_mode_selector_inline > div {
        pointer-events: auto;
        display: block !important;
        background: rgba(20, 22, 30, 0.92);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 999px;
        padding: 0.18rem 0.25rem;
        margin-right: 4.25rem !important;
        width: fit-content !important;
        max-width: fit-content;
        transform: translateY(0.12rem);
    }
    .st-key-chat_mode_selector_inline [role="radiogroup"] {
        gap: 0.2rem;
        flex-wrap: nowrap !important;
        justify-content: flex-end !important;
    }
    .st-key-chat_mode_selector_inline label {
        margin-bottom: 0 !important;
    }
    .st-key-chat_mode_selector_inline p {
        font-size: 0.78rem !important;
        margin: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DB_PATH = resolve_workspace_path(os.getenv("APP_DB_PATH", str(case_db_path(ROOT))))
init_case_db(APP_DB_PATH)
legacy_case = ensure_legacy_case(APP_DB_PATH, ROOT, DEFAULT_TITLE)

if not st.session_state.get("is_authenticated"):
    render_login_screen()

all_cases = list_managed_cases(APP_DB_PATH)
case_lookup = {case.slug: case for case in all_cases}
if not all_cases:
    fallback_case = create_managed_case(
        APP_DB_PATH,
        root=ROOT,
        name="Default Study",
        description="Auto-created workspace because no study existed yet.",
        research_goal="Start a new case-managed study.",
    )
    all_cases = list_managed_cases(APP_DB_PATH)
    case_lookup = {case.slug: case for case in all_cases}
else:
    fallback_case = all_cases[0]
if "selected_case_slug" not in st.session_state or st.session_state.get("selected_case_slug") not in case_lookup:
    st.session_state["selected_case_slug"] = (legacy_case.slug if legacy_case else fallback_case.slug)

selected_case = case_lookup.get(st.session_state["selected_case_slug"]) or fallback_case
if st.session_state.get("active_case_id") != selected_case.id:
    sync_chat_session_from_db(APP_DB_PATH, selected_case.id)

if "app_view" not in st.session_state:
    st.session_state["app_view"] = "home"


with st.sidebar:
    st.title("Research Copilot")
    st.caption(f"Signed in as `{st.session_state.get('auth_user', 'admin')}`")
    top_sidebar_cols = st.columns(2)
    if top_sidebar_cols[0].button("Home", width="stretch"):
        st.session_state["app_view"] = "home"
        st.rerun()
    if top_sidebar_cols[1].button("Logout", width="stretch"):
        st.session_state.clear()
        st.rerun()

    case_options = [(case.slug, f"{case.name}{' (legacy)' if case.is_legacy else ''}") for case in all_cases]
    selected_case_choice = st.selectbox(
        "Active case",
        case_options,
        index=next((index for index, item in enumerate(case_options) if item[0] == selected_case.slug), 0),
        format_func=lambda item: item[1],
    )
    if selected_case_choice[0] != selected_case.slug:
        st.session_state["selected_case_slug"] = selected_case_choice[0]
        st.rerun()
    selected_case = get_case_by_slug(APP_DB_PATH, selected_case_choice[0]) or selected_case

    nav_choice = st.radio(
        "Navigation",
        [("home", "Home"), ("workspace", "Research Workspace"), ("new_case", "New Case")],
        index=0 if st.session_state.get("app_view", "home") == "home" else (1 if st.session_state.get("app_view") == "workspace" else 2),
        format_func=lambda item: item[1],
    )
    st.session_state["app_view"] = nav_choice[0]

    workspace_title = st.text_input("Workspace title", value=selected_case.name)
    if workspace_title.strip() and workspace_title.strip() != selected_case.name:
        selected_case = update_case_metadata(
            APP_DB_PATH,
            selected_case.id,
            name=workspace_title.strip(),
            description=selected_case.description,
            research_goal=selected_case.research_goal,
        ) or selected_case
        st.session_state["selected_case_slug"] = selected_case.slug
    st.session_state["workspace_title"] = workspace_title.strip() or selected_case.name

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
        st.markdown("**Multimodal RAG defaults**")
        multimodal_text_model = st.text_input("RAG text model", value=os.getenv("MULTIMODAL_TEXT_MODEL", "gpt-oss:20b"))
        multimodal_image_model = st.text_input("Vision caption model", value=os.getenv("MULTIMODAL_IMAGE_MODEL", "llama3.2-vision:11b"))
        multimodal_embed_model = FIXED_MULTIMODAL_EMBED_MODEL
        st.caption(f"Multimodal embeddings: `{multimodal_embed_model}` (fixed)")
        mineru_backend = st.selectbox(
            "MinerU backend",
            ["pipeline", "hybrid-auto-engine", "vlm-auto-engine"],
            index=["pipeline", "hybrid-auto-engine", "vlm-auto-engine"].index(os.getenv("MINERU_BACKEND", "pipeline"))
            if os.getenv("MINERU_BACKEND", "pipeline") in ["pipeline", "hybrid-auto-engine", "vlm-auto-engine"]
            else 0,
        )
        mineru_method = st.selectbox(
            "MinerU method",
            ["auto", "txt", "ocr"],
            index=["auto", "txt", "ocr"].index(os.getenv("MINERU_METHOD", "auto")) if os.getenv("MINERU_METHOD", "auto") in ["auto", "txt", "ocr"] else 0,
        )
        mineru_lang = st.text_input("MinerU language", value=os.getenv("MINERU_LANG", "en"))
        mineru_timeout = st.number_input("MinerU timeout seconds", min_value=300, max_value=7200, step=300, value=env_int("MINERU_TIMEOUT", 1800))
        multimodal_retrieval_limit = st.number_input("RAG retrieved evidence items", min_value=3, max_value=40, value=env_int("MULTIMODAL_RETRIEVAL_LIMIT", 10))
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
        managed_paths = case_paths(selected_case)
        bib_pdf_raw = st.text_input("PDF staging directory", value=str(managed_paths["bib_pdf"]), disabled=True)
        multimodal_store_raw = st.text_input("Multimodal store directory", value=str(managed_paths["multimodal_store"]), disabled=True)
        graph_store_raw = st.text_input("Graph store directory", value=str(managed_paths["graph_store"]), disabled=True)
        compiled_output_raw = st.text_input("Compile output directory", value=str(managed_paths["compiled_output"]), disabled=True)
        paper_tex_raw = st.text_input("LaTeX source path", value=str(managed_paths["paper_tex"]), disabled=True)
        if st.button("Sync Current Case To MinIO", width="stretch"):
            with st.spinner("Uploading this case to MinIO..."):
                try:
                    sync_result = sync_case_to_minio(APP_DB_PATH, selected_case)
                    safe_toast(f"Synced {sync_result['uploaded']} objects to MinIO")
                except Exception as exc:
                    st.error(f"MinIO sync failed: {exc}")

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
    "multimodal_text_model": multimodal_text_model,
    "multimodal_image_model": multimodal_image_model,
    "multimodal_embed_model": multimodal_embed_model,
    "mineru_backend": mineru_backend,
    "mineru_method": mineru_method,
    "mineru_lang": mineru_lang,
    "mineru_timeout": int(mineru_timeout),
    "multimodal_retrieval_limit": int(multimodal_retrieval_limit),
}

paths = RuntimePaths(
    bib_pdf=resolve_workspace_path(bib_pdf_raw),
    compiled_output=resolve_workspace_path(compiled_output_raw),
    paper_tex=resolve_workspace_path(paper_tex_raw),
)
multimodal_store_path = resolve_workspace_path(multimodal_store_raw)
graph_store_path = resolve_workspace_path(graph_store_raw)
created_paper = ensure_bootstrap_files(paths, workspace_title)
sync_latex_session(paths.paper_tex)
missing_modules = missing_runtime_modules(settings)

if st.session_state.get("app_view") == "home":
    render_home_page(APP_DB_PATH, selected_case, all_cases)
    st.stop()
if st.session_state.get("app_view") == "new_case":
    render_new_case_page(APP_DB_PATH, ROOT)
    st.stop()

st.title(workspace_title)
st.caption(f"Active model: {settings['backend']} / {settings['model']} | Paper: {paths.paper_tex.relative_to(ROOT) if paths.paper_tex.is_relative_to(ROOT) else paths.paper_tex}")
st.caption(
    f"Case `{selected_case.slug}` | GPT Researcher `{GPT_RESEARCHER_ENGINE_PATH}` | Multimodal store `{multimodal_store_path}` | Retriever `{effective_retriever(settings)}`"
)
if created_paper:
    safe_toast(f"Initialized {paths.paper_tex.name}")
if missing_modules:
    st.error(
        "Missing runtime modules: "
        + ", ".join(missing_modules)
        + ". Install the updated requirements and restart Streamlit."
    )

tab_search, tab_chat, tab_latex = st.tabs(
    [
        "Deep Search & Multimodal Indexing",
        "Agentic Multimodal RAG Chat",
        "Live LaTeX Studio & Local Compiling",
    ]
)
maybe_switch_workspace_tab()


with tab_search:
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
        preflight_issues = runtime_preflight(settings, paths)
        if preflight_issues:
            st.error("\n".join(f"- {issue}" for issue in preflight_issues))
        elif not query.strip():
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
                preflight_issues = runtime_preflight(settings, paths)
                if preflight_issues:
                    st.error("\n".join(f"- {issue}" for issue in preflight_issues))
                elif not planned_queries:
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
                        synced, sync_error = maybe_auto_sync_case(APP_DB_PATH, selected_case)
                        if sync_error:
                            st.warning(f"MinIO sync skipped after deep search: {sync_error}")
                        elif synced:
                            st.caption("MinIO sync completed for the updated case artifacts.")
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
        render_markdown_block(report_text)
        render_report_resource_explorer(report_text, paths.bib_pdf, "report_resources")
        action_cols = st.columns([0.32, 0.68])
        if action_cols[0].button("Move To Index Page", type="primary", width="stretch"):
            request_workspace_tab_switch("Agentic Multimodal RAG Chat")
            st.rerun()
    else:
        st.info("Run Deep Search to generate a report, or select an existing markdown report from the list.")


with tab_chat:
    st.subheader("Agentic Multimodal RAG Chat")
    try:
        from multimodal_pipeline import clean_store, ingest_pdf, store_summary
        from scientific_graph_rag import build_all_graphs, clean_graph_store as clean_graph_store_runtime

        chat_store_summary = store_summary(multimodal_store_path)
    except Exception as exc:
        clean_store = None
        ingest_pdf = None
        build_all_graphs = None
        clean_graph_store_runtime = None
        chat_store_summary = {"documents": 0, "vector_records": 0, "images": 0, "tables": 0}
        st.error("Multimodal retrieval backend is not available.")
        st.code(str(exc), language="text")

    graph_summary: dict[str, Any] = {"documents": 0}
    try:
        from scientific_graph_rag import load_graphs

        graph_summary["documents"] = len(load_graphs(graph_store_path))
    except Exception:
        graph_summary["documents"] = 0

    st.markdown("### Staging Room")
    pdfs = discover_pdfs(paths.bib_pdf)
    staging_metrics = st.columns(6)
    staging_metrics[0].metric("Staged PDFs", len(pdfs))
    staging_metrics[1].metric("Indexed docs", chat_store_summary.get("documents", 0))
    staging_metrics[2].metric("Vector records", chat_store_summary.get("vector_records", 0))
    staging_metrics[3].metric("Tables", chat_store_summary.get("tables", 0))
    staging_metrics[4].metric("Images", chat_store_summary.get("images", 0))
    staging_metrics[5].metric("Graph docs", graph_summary.get("documents", 0))
    st.caption(f"Store: `{multimodal_store_path}` | Graph store: `{graph_store_path}`")

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

    index_col, reset_col = st.columns([0.72, 0.28])
    if index_col.button("Index", type="primary", width="stretch"):
        preflight_issues = runtime_preflight(settings, paths, require_pdfs=True)
        if preflight_issues:
            st.error("\n".join(f"- {issue}" for issue in preflight_issues))
        elif not pdfs:
            st.warning("Add PDFs to the staging directory before indexing.")
        elif ingest_pdf is None:
            st.error("Multimodal pipeline is not available.")
        else:
            results: list[dict[str, Any]] = []
            progress = st.progress(0)
            with st.status("Preparing multimodal indexing...", expanded=True) as status:
                st.write(f"Found {len(pdfs)} staged PDF(s).")
                st.write("Using MinerU parsing, multimodal enrichment, fixed mpnet embeddings, and local Qdrant indexing.")
                for index, pdf_path in enumerate(pdfs, start=1):
                    status.update(label=f"Indexing {index}/{len(pdfs)}: {pdf_path.name}", state="running")
                    st.write(f"Parsing and enriching `{pdf_path.name}`...")
                    try:
                        result = ingest_pdf(
                            pdf_path,
                            store_root=multimodal_store_path,
                            text_model=settings["multimodal_text_model"],
                            image_model=settings["multimodal_image_model"],
                            embed_model=settings["multimodal_embed_model"],
                            ollama_base_url=settings["ollama_base_url"],
                            use_mineru=True,
                            mineru_backend=settings["mineru_backend"],
                            mineru_method=settings["mineru_method"],
                            mineru_lang=settings["mineru_lang"],
                            mineru_timeout=int(settings["mineru_timeout"]),
                            enrich_images=True,
                            enrich_tables=True,
                            index=True,
                            force=True,
                        )
                        results.append(result)
                        st.write(
                            f"Indexed `{pdf_path.name}` with `{result.get('parser')}`: "
                            f"{result.get('vector_records', 0)} vectors, {result.get('images', 0)} images."
                        )
                    except Exception as exc:
                        results.append({"status": "error", "source_pdf": str(pdf_path), "error": str(exc)})
                        st.write(f"Error indexing `{pdf_path.name}`: {exc}")
                    progress.progress(index / len(pdfs))
                errors = [item for item in results if item.get("status") == "error"]
                if errors:
                    status.update(label=f"Indexing finished with {len(errors)} error(s)", state="error")
                    st.error(f"Indexing finished with {len(errors)} error(s). Check terminal logs for detailed traces.")
                else:
                    if build_all_graphs is not None and clean_graph_store_runtime is not None:
                        status.update(label="Building graph RAG store...", state="running")
                        st.write("Creating graph nodes, edges, manifests, and query-ready retrieval artifacts.")
                        try:
                            clean_graph_store_runtime(graph_store_path)
                            graph_backend = "ollama"
                            graph_model = str(settings["multimodal_text_model"]).strip() or str(settings["model"]).strip()
                            graph_cloud_provider = "openai"
                            graph_api_key = ""
                            graph_base_url = ""
                            if settings["backend"] == "Cloud API" and settings.get("api_key", "").strip():
                                graph_backend = "cloud"
                                graph_model = str(settings["model"]).strip()
                                graph_cloud_provider = "deepseek" if settings.get("cloud_provider") == "DeepSeek" else "openai"
                                graph_api_key = settings.get("api_key", "").strip()
                                graph_base_url = settings.get("openai_base_url", "").strip()
                            graph_results = build_all_graphs(
                                multimodal_store_path,
                                graph_store_path,
                                backend=graph_backend,
                                model=graph_model,
                                ollama_base_url=settings["ollama_base_url"],
                                cloud_provider=graph_cloud_provider,
                                api_key=graph_api_key,
                                base_url=graph_base_url,
                                enrich_root=True,
                            )
                            st.write(f"Built {len(graph_results)} graph document package(s).")
                        except Exception as exc:
                            errors.append({"status": "error", "source_pdf": "graph_build", "error": str(exc)})
                            st.write(f"Graph build error: {exc}")
                    if errors:
                        status.update(label=f"Indexing finished with {len(errors)} error(s)", state="error")
                        st.error(f"Indexing finished with {len(errors)} error(s). Check terminal logs for detailed traces.")
                    else:
                        status.update(label="Multimodal index is ready", state="complete")
                        safe_toast("Multimodal RAG index ready")
                        st.success(f"Indexed {len(results)} PDF(s) into `{multimodal_store_path}`.")
                        synced, sync_error = maybe_auto_sync_case(APP_DB_PATH, selected_case)
                        if sync_error:
                            st.warning(f"MinIO sync skipped after indexing: {sync_error}")
                        elif synced:
                            st.caption("MinIO sync completed for the indexed study artifacts.")
            st.dataframe(summarize_multimodal_results(results), width="stretch", hide_index=True)

    if reset_col.button("Reset Index", width="stretch"):
        if clean_store is None:
            st.error("Multimodal pipeline is not available.")
        else:
            clean_store(multimodal_store_path)
            if clean_graph_store_runtime is not None:
                clean_graph_store_runtime(graph_store_path)
            safe_toast("Multimodal index reset")
            st.rerun()

    st.divider()
    st.markdown("### Workspace Cross-Examination")
    st.caption(
        f"Query backend: Qdrant `{multimodal_store_path}` | generation: "
        f"{settings['backend']} / `{settings['model']}` | embeddings `{settings['multimodal_embed_model']}`"
    )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "chat_pending_question" not in st.session_state:
        st.session_state["chat_pending_question"] = ""
    if "chat_pending_mode" not in st.session_state:
        st.session_state["chat_pending_mode"] = "standard"
    if "chat_mode" not in st.session_state:
        st.session_state["chat_mode"] = "standard"

    toolbar_cols = st.columns([0.2, 0.8])
    if toolbar_cols[0].button("Clear conversation", width="stretch"):
        st.session_state["chat_messages"] = []
        st.session_state["chat_pending_question"] = ""
        st.session_state["chat_pending_mode"] = st.session_state.get("chat_mode", "standard")
        clear_case_chat_history(APP_DB_PATH, selected_case.id)
        st.rerun()

    st.markdown("**Ask the indexed workspace**")

    queued_question = st.session_state.get("chat_pending_question", "").strip()
    if queued_question:
        # Pop first to avoid infinite retry loops on generation errors.
        st.session_state["chat_pending_question"] = ""
        queued_mode = st.session_state.get("chat_pending_mode", "standard")
        preflight_issues = runtime_preflight(
            settings,
            paths,
            multimodal_store_path=multimodal_store_path,
            graph_store_path=graph_store_path,
            require_multimodal_index=(queued_mode == "standard"),
            require_graph_index=(queued_mode == "research"),
        )
        if preflight_issues:
            answer = "RAG chat cannot start yet:\n\n" + "\n".join(f"- {issue}" for issue in preflight_issues)
            artifacts = []
            evidence_refs = []
        else:
            with st.spinner(
                (
                    f"Retrieving Qdrant evidence and synthesizing with {settings['backend']} / {settings['model']}..."
                    if queued_mode == "standard"
                    else f"Running graph-backed research mode with {settings['backend']} / {settings['model']}..."
                )
            ):
                try:
                    if queued_mode == "research":
                        response = graph_chat_answer(
                            queued_question,
                            settings=settings,
                            graph_store_path=graph_store_path,
                        )
                    else:
                        response = multimodal_chat_answer(
                            queued_question,
                            settings=settings,
                            multimodal_store_path=multimodal_store_path,
                        )
                    answer = response["answer_markdown"]
                    artifacts = response.get("artifact_references", [])
                    evidence_refs = response.get("evidence_references", [])
                except Exception as exc:
                    answer = f"RAG chat failed:\n\n```text\n{exc}\n```"
                    artifacts = []
                    evidence_refs = []
        st.session_state["chat_messages"].append(
            {
                "role": "assistant",
                "content": answer,
                "artifacts": artifacts,
                "evidence": evidence_refs,
                "mode": queued_mode,
            }
        )
        persist_chat_message(
            APP_DB_PATH,
            selected_case.id,
            role="assistant",
            content=answer,
            mode=queued_mode,
            artifacts=artifacts,
            evidence=evidence_refs,
        )
        maybe_auto_sync_case(APP_DB_PATH, selected_case)
        st.rerun()

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("mode") == "research":
                st.caption("Research Mode")
            render_markdown_block(message["content"])
            if message.get("evidence"):
                render_evidence_resources(
                    message["evidence"],
                    bib_pdf_dir=paths.bib_pdf,
                    key_prefix=f"chat_evidence_{abs(hash(message['content']))}",
                )
            if message.get("artifacts"):
                render_artifacts(message["artifacts"])

    pending_question = st.chat_input(
        "Ask a high-precision question about papers, datasets, methods, tables, and figures...",
        key="multimodal_chat_input",
        width="stretch",
    )
    chat_mode = st.radio(
        "Reasoning mode",
        [("standard", "🧠 Standard RAG"), ("research", "🕸 Research Mode")],
        index=0 if st.session_state.get("chat_mode", "standard") == "standard" else 1,
        format_func=lambda item: item[1],
        key="chat_mode_selector_inline",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["chat_mode"] = chat_mode[0]
    if pending_question:
        st.session_state["chat_messages"].append(
            {
                "role": "user",
                "content": pending_question,
                "mode": st.session_state.get("chat_mode", "standard"),
            }
        )
        persist_chat_message(
            APP_DB_PATH,
            selected_case.id,
            role="user",
            content=pending_question,
            mode=st.session_state.get("chat_mode", "standard"),
            artifacts=[],
        )
        st.session_state["chat_pending_question"] = pending_question
        st.session_state["chat_pending_mode"] = st.session_state.get("chat_mode", "standard")
        st.rerun()


with tab_latex:
    st.subheader("AI Co-Author")
    modify_instruction = st.text_area(
        "Ask AI to modify your active LaTeX source directly",
        height=115,
    )

    if st.button("Apply AI Edit To Source", type="primary", width="stretch"):
        save_editor_if_changed(paths)
        preflight_issues = runtime_preflight(settings, paths, require_paper=True)
        if preflight_issues:
            st.error("\n".join(f"- {issue}" for issue in preflight_issues))
        elif not modify_instruction.strip():
            st.warning("Enter an editing instruction first.")
        else:
            with st.spinner("The co-author is rewriting the LaTeX source..."):
                try:
                    updated = modify_latex_source(st.session_state["latex_editor"], modify_instruction.strip(), settings, paths)
                    atomic_write_text(paths.paper_tex, updated)
                    st.session_state["latex_editor"] = updated
                    st.session_state["latex_saved_sha"] = sha256_text(updated)
                    safe_toast("LaTeX source updated")
                    st.rerun()
                except Exception as exc:
                    st.error("AI edit failed.")
                    st.code(str(exc), language="text")

    compile_result = st.session_state.get("latex_last_compile_result")
    if st.button("🚀 Compile Document", type="primary", width="stretch"):
        save_editor_if_changed(paths)
        with st.spinner("Running pdflatex twice..."):
            compile_result = compile_latex(paths)
        st.session_state["latex_last_compile_result"] = compile_result
        if compile_result["ok"]:
            st.success(f"Compilation succeeded: {compile_result['pdf_path']}")
            synced, sync_error = maybe_auto_sync_case(APP_DB_PATH, selected_case)
            if sync_error:
                st.warning(f"MinIO sync skipped after compilation: {sync_error}")
            elif synced:
                st.caption("MinIO sync completed for the compiled LaTeX artifacts.")
        else:
            st.error("Compilation failed.")

    source_col, preview_col = st.columns([1.08, 0.92], gap="large")
    with source_col:
        st.text_area("Active LaTeX source", key="latex_editor", height=760)
        save_editor_if_changed(paths)

    with preview_col:
        st.subheader("PDF Preview")
        st.write(f"Source: `{paths.paper_tex}`")
        st.write(f"Output: `{paths.compiled_output}`")
        pdf_path = paths.compiled_output / f"{paths.paper_tex.stem}.pdf"
        if pdf_path.exists():
            try:
                components.html(pdf_embed_html(pdf_path, height=760), height=780, scrolling=True)
                st.caption("Scrollable PDF viewer")
            except Exception:
                previews, page_total = render_pdf_preview(pdf_path, max_pages=3)
                st.caption(f"Compiled PDF · {page_total} page(s)")
                for page_number, preview in enumerate(previews, start=1):
                    st.image(preview, caption=f"Page {page_number}", width="stretch")
            st.download_button(
                "Download compiled PDF",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                width="stretch",
            )
        else:
            st.info("Compile the document to see the PDF preview here.")

        if compile_result and not compile_result.get("ok"):
            st.code(compile_result.get("log", ""), language="text")
        elif compile_result and compile_result.get("log"):
            with st.expander("Compilation log", expanded=False):
                st.code(compile_result.get("log", ""), language="text")
