from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

LOGGER = logging.getLogger("deepscipaper.graph_rag")

DEFAULT_SOURCE_STORE = Path(os.getenv("MULTIMODAL_STORE_DIR", "./multimodal_store"))
DEFAULT_GRAPH_STORE = Path(os.getenv("SCIENTIFIC_GRAPH_RAG_STORE_DIR", "./scientific_graph_rag_store"))
FIXED_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
FALLBACK_EMBED_DIM = 512
DEFAULT_TEXT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-oss:20b")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_CLOUD_MODEL = os.getenv("DEFAULT_CLOUD_MODEL", "gpt-4o")
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEFAULT_DEEPSEEK_MODEL", "deepseek-chat")
EMBEDDING_BACKEND_STATE = {"name": FIXED_EMBED_MODEL}


@dataclass(frozen=True)
class GraphStorePaths:
    root: Path
    graphs: Path
    queries: Path
    cache: Path
    manifests: Path


def setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    LOGGER.setLevel(level)


def graph_store_paths(root: Path | str = DEFAULT_GRAPH_STORE) -> GraphStorePaths:
    root_path = Path(root).expanduser().resolve()
    return GraphStorePaths(
        root=root_path,
        graphs=root_path / "graphs",
        queries=root_path / "queries",
        cache=root_path / "cache",
        manifests=root_path / "manifests",
    )


def ensure_graph_store(paths: GraphStorePaths) -> None:
    for path in (paths.root, paths.graphs, paths.queries, paths.cache, paths.manifests):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s<>\]\)\"']+", text or "")


def extract_citation_markers(text: str) -> list[str]:
    markers = set(re.findall(r"\[[0-9,\-\s]+\]", text or ""))
    markers.update(re.findall(r"\([A-Z][A-Za-z\-]+(?: et al\.)?,\s*\d{4}[a-z]?\)", text or ""))
    return sorted(markers)


@lru_cache(maxsize=1)
def load_sentence_embedding_model():
    from sentence_transformers import SentenceTransformer

    LOGGER.info("Loading embedding model from local cache | %s", FIXED_EMBED_MODEL)
    return SentenceTransformer(FIXED_EMBED_MODEL, local_files_only=True)


def hashed_embedding(text: str, dim: int = FALLBACK_EMBED_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = re.findall(r"[A-Za-z0-9_]{2,}", (text or "").lower())
    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % dim
        sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
        weight = 1.0 + min(len(token), 12) / 12.0
        vector[bucket] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def sentence_embed(texts: list[str]) -> list[list[float]]:
    clean = [normalize_space(text)[:12000] or "empty" for text in texts]
    try:
        model = load_sentence_embedding_model()
        vectors = model.encode(clean, normalize_embeddings=True, show_progress_bar=False)
        EMBEDDING_BACKEND_STATE["name"] = FIXED_EMBED_MODEL
        return vectors.tolist()
    except Exception as exc:
        LOGGER.warning(
            "Falling back to deterministic lexical embeddings because the local mpnet cache is unavailable | %s",
            exc,
        )
        EMBEDDING_BACKEND_STATE["name"] = f"hashed_lexical_{FALLBACK_EMBED_DIM}d"
        return [hashed_embedding(text) for text in clean]


def ollama_chat(messages: list[dict[str, str]], model: str, base_url: str, timeout_seconds: int = 300) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2200},
    }
    response = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    content = ((response.json().get("message") or {}).get("content") or "").strip()
    content = re.sub(r"<think>[\s\S]*?</think>\s*", "", content, flags=re.IGNORECASE).strip()
    return content


def cloud_chat(
    messages: list[dict[str, str]],
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    timeout_seconds: int = 300,
) -> str:
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout_seconds}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
    content = (response.choices[0].message.content or "").strip()
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", content, flags=re.IGNORECASE).strip()


def chat_complete(
    messages: list[dict[str, str]],
    backend: str,
    model: str,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    cloud_provider: str = "openai",
    api_key: str = "",
    base_url: str = "",
) -> str:
    if backend == "ollama":
        return ollama_chat(messages, model=model, base_url=ollama_base_url)
    if not api_key.strip():
        raise RuntimeError("Cloud backend selected but no API key was provided.")
    provider = cloud_provider.lower()
    if provider == "deepseek":
        return cloud_chat(
            messages,
            provider="deepseek",
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com",
        )
    return cloud_chat(
        messages,
        provider="openai",
        model=model,
        api_key=api_key,
        base_url=base_url or os.getenv("OPENAI_BASE_URL", ""),
    )


def list_record_files(source_store: Path) -> list[Path]:
    return sorted((source_store / "indexes").glob("*_records.json"))


def infer_root_metadata(records: list[dict[str, Any]], manifest: dict[str, Any], markdown_text: str) -> dict[str, Any]:
    title = manifest.get("source_pdf_name", "")
    abstract = ""
    keywords = []
    high_level_summary = ""
    topic = ""
    objectives = ""
    scope = ""

    for record in records:
        record_title = str(record.get("title", "")).strip()
        display = str(record.get("display_text", "")).strip()
        if not title and record_title:
            title = record_title
        if not abstract and "abstract" in record_title.lower():
            abstract = display[:3000]
        if not high_level_summary and record_title.lower() in {"summary", "a b s t r a c t", "abstract"}:
            high_level_summary = display[:3000]
        if not topic and record_title.lower() in {"keywords", "articleinfo"}:
            topic = display[:1000]

    if not abstract:
        match = re.search(r"##\s+(?:ABSTRACT|Abstract|A B S T R A C T)\s+(.*?)(?:\n##\s+|\Z)", markdown_text, flags=re.S)
        if match:
            abstract = normalize_space(match.group(1))[:3000]
    if not topic:
        match = re.search(r"Keywords:\s*(.+?)(?:\n\n|\n##|\Z)", markdown_text, flags=re.S)
        if match:
            topic = normalize_space(match.group(1))[:600]
            keywords = [item.strip(" ,;") for item in re.split(r"[,;\n]+", match.group(1)) if item.strip()]
    if not high_level_summary:
        high_level_summary = abstract or normalize_space(markdown_text[:2500])
    objectives = abstract or high_level_summary
    scope = normalize_space(" ".join(filter(None, [title, topic, objectives])))[:1500]

    return {
        "title": title or manifest.get("doc_id", ""),
        "abstract": abstract,
        "keywords": keywords[:20],
        "high_level_summary": high_level_summary,
        "global_topic": topic or title,
        "global_objectives": objectives,
        "global_scope": scope,
    }


def enrich_root_metadata_with_llm(
    root_meta: dict[str, Any],
    markdown_text: str,
    backend: str,
    model: str,
    ollama_base_url: str,
    cloud_provider: str,
    api_key: str,
    base_url: str,
) -> tuple[dict[str, Any], str]:
    excerpt = normalize_space(markdown_text)[:7000]
    prompt = (
        "You are enriching metadata for a scientific paper knowledge graph.\n"
        "Return JSON only with keys: title, abstract, keywords, high_level_summary, global_topic, global_objectives, global_scope.\n"
        "Keep keywords as a JSON array of short strings. Preserve scientific fidelity and do not invent claims.\n\n"
        f"Existing metadata:\n{json.dumps(root_meta, ensure_ascii=False)}\n\n"
        f"Paper excerpt:\n{excerpt}"
    )
    try:
        response = chat_complete(
            [{"role": "user", "content": prompt}],
            backend=backend,
            model=model,
            ollama_base_url=ollama_base_url,
            cloud_provider=cloud_provider,
            api_key=api_key,
            base_url=base_url,
        )
        match = re.search(r"\{[\s\S]*\}", response)
        if not match:
            return root_meta, ""
        payload = json.loads(match.group(0))
        merged = dict(root_meta)
        for key in ("title", "abstract", "high_level_summary", "global_topic", "global_objectives", "global_scope"):
            value = normalize_space(payload.get(key, ""))
            if value:
                merged[key] = value
        keywords = payload.get("keywords", [])
        if isinstance(keywords, list):
            merged["keywords"] = [normalize_space(item) for item in keywords if normalize_space(item)][:20]
        return merged, ""
    except Exception as exc:
        LOGGER.warning("Root metadata enrichment unavailable, keeping heuristic metadata | %s", exc)
        return root_meta, str(exc)


def node_id_from_record(record: dict[str, Any]) -> str:
    return str(record.get("record_id", "")).replace(":", "__")


def infer_node_type(record: dict[str, Any]) -> str:
    modality = str(record.get("modality", "")).lower()
    title = str(record.get("title", "")).lower()
    if modality == "image":
        return "figure"
    if modality == "table" or title.startswith("table"):
        return "table"
    if modality == "text" and title:
        return "section"
    return modality or "text"


def referenced_artifact_numbers(text: str, prefix: str) -> list[str]:
    pattern = re.compile(rf"\b{re.escape(prefix)}\s+([0-9]+[A-Za-z]?)", flags=re.I)
    return sorted(set(match.group(1) for match in pattern.finditer(text or "")))


def build_document_graph(
    record_path: Path,
    source_store: Path,
    graph_store: GraphStorePaths,
    *,
    backend: str,
    model: str,
    ollama_base_url: str,
    cloud_provider: str,
    api_key: str,
    base_url: str,
    enrich_root: bool,
) -> dict[str, Any]:
    records = read_json(record_path, [])
    if not records:
        raise RuntimeError(f"No records found in {record_path}")

    doc_id = str(records[0].get("doc_id", record_path.stem.replace("_records", "")))
    manifest_path = source_store / "manifests" / f"{doc_id}.json"
    manifest = read_json(manifest_path, {})
    markdown_path = Path(str(manifest.get("markdown_path", "")))
    markdown_text = markdown_path.read_text(encoding="utf-8", errors="replace") if markdown_path.exists() else ""
    root_meta = infer_root_metadata(records, manifest, markdown_text)
    enrichment_error = ""
    if enrich_root:
        root_meta, enrichment_error = enrich_root_metadata_with_llm(
            root_meta,
            markdown_text,
            backend=backend,
            model=model,
            ollama_base_url=ollama_base_url,
            cloud_provider=cloud_provider,
            api_key=api_key,
            base_url=base_url,
        )

    graph_dir = graph_store.graphs / doc_id
    graph_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building document graph | doc_id=%s | records=%s", doc_id, len(records))

    node_inputs = []
    ordered_nodes: list[dict[str, Any]] = []
    section_lookup: dict[str, list[str]] = {}
    title_lookup: dict[str, str] = {}

    root_node_id = f"{doc_id}__root"
    root_node = {
        "node_id": root_node_id,
        "doc_id": doc_id,
        "node_type": "paper_root",
        "modality": "root",
        "title": root_meta["title"],
        "content": root_meta["high_level_summary"],
        "asset_path": manifest.get("markdown_path", ""),
        "global_context": {
            "topic": root_meta["global_topic"],
            "objectives": root_meta["global_objectives"],
            "scope": root_meta["global_scope"],
        },
        "local_context": {
            "parent_section": None,
            "previous_node": None,
            "next_node": None,
            "related_nodes": [],
        },
        "paper_metadata": {
            "abstract": root_meta["abstract"],
            "keywords": root_meta["keywords"],
            "summary": root_meta["high_level_summary"],
            "source_pdf_name": manifest.get("source_pdf_name", ""),
            "source_pdf": manifest.get("source_pdf", ""),
        },
        "citations": [],
    }
    ordered_nodes.append(root_node)
    node_inputs.append(
        "\n".join(
            [
                root_meta["title"],
                root_meta["abstract"],
                " ".join(root_meta["keywords"]),
                root_meta["high_level_summary"],
            ]
        )
    )

    for index, record in enumerate(records):
        node_id = node_id_from_record(record)
        title = str(record.get("title", "")).strip()
        display_text = str(record.get("display_text", "")).strip() or str(record.get("text_for_embedding", "")).strip()
        section_title = str(record.get("section_title") or title or "Document").strip()
        title_lookup[title.lower()] = node_id
        section_lookup.setdefault(section_title.lower(), []).append(node_id)

        node = {
            "node_id": node_id,
            "doc_id": doc_id,
            "source_record_id": record.get("record_id"),
            "node_type": infer_node_type(record),
            "modality": record.get("modality"),
            "title": title,
            "content": display_text,
            "asset_path": record.get("asset_path", ""),
            "global_context": {
                "topic": root_meta["global_topic"],
                "objectives": root_meta["global_objectives"],
                "scope": root_meta["global_scope"],
            },
            "local_context": {
                "parent_section": section_title,
                "previous_node": None,
                "next_node": None,
                "related_nodes": [],
            },
            "paper_metadata": {
                "source_pdf_name": record.get("source_pdf_name", ""),
                "source_pdf": record.get("source_pdf", ""),
            },
            "citations": extract_citation_markers(display_text),
            "urls": extract_urls(display_text),
        }
        ordered_nodes.append(node)
        node_inputs.append("\n".join(filter(None, [title, section_title, display_text])))

    embeddings = sentence_embed(node_inputs)
    for node, vector in zip(ordered_nodes, embeddings):
        node["embedding"] = vector

    edges: list[dict[str, Any]] = []
    nodes_by_id = {node["node_id"]: node for node in ordered_nodes}

    text_nodes = [node for node in ordered_nodes if node["node_type"] != "paper_root"]
    for position, node in enumerate(text_nodes):
        prev_node = text_nodes[position - 1]["node_id"] if position > 0 else None
        next_node = text_nodes[position + 1]["node_id"] if position + 1 < len(text_nodes) else None
        node["local_context"]["previous_node"] = prev_node
        node["local_context"]["next_node"] = next_node
        edges.append({"source": root_node_id, "target": node["node_id"], "type": "contains", "score": 1.0})
        if prev_node:
            edges.append({"source": prev_node, "target": node["node_id"], "type": "next", "score": 1.0})
            edges.append({"source": node["node_id"], "target": prev_node, "type": "previous", "score": 1.0})

    section_representatives: dict[str, str] = {}
    for node in text_nodes:
        section_name = str(node["local_context"]["parent_section"] or "").lower()
        if section_name and section_name not in section_representatives:
            section_representatives[section_name] = node["node_id"]
        peers = [node_id for node_id in section_lookup.get(section_name, []) if node_id != node["node_id"]][:6]
        node["local_context"]["related_nodes"].extend(peers)
        for peer_id in peers:
            edges.append({"source": node["node_id"], "target": peer_id, "type": "same_section", "score": 0.8})
        if section_name in section_representatives:
            edges.append(
                {
                    "source": root_node_id,
                    "target": section_representatives[section_name],
                    "type": "section_entry",
                    "label": node["local_context"]["parent_section"],
                    "score": 1.0,
                }
            )

    table_nodes = [node for node in text_nodes if node["node_type"] == "table"]
    figure_nodes = [node for node in text_nodes if node["node_type"] == "figure"]
    table_by_num = {re.sub(r"[^0-9A-Za-z]", "", node["title"].split()[-1]): node["node_id"] for node in table_nodes if node["title"]}
    figure_by_num = {re.sub(r"[^0-9A-Za-z]", "", node["title"].split()[-1]): node["node_id"] for node in figure_nodes if node["title"]}

    for node in text_nodes:
        content = node["content"]
        for number in referenced_artifact_numbers(content, "Table"):
            target = table_by_num.get(re.sub(r"[^0-9A-Za-z]", "", number))
            if target:
                edges.append({"source": node["node_id"], "target": target, "type": "references_table", "score": 0.95})
                node["local_context"]["related_nodes"].append(target)
        for number in referenced_artifact_numbers(content, "Figure"):
            target = figure_by_num.get(re.sub(r"[^0-9A-Za-z]", "", number))
            if target:
                edges.append({"source": node["node_id"], "target": target, "type": "references_figure", "score": 0.95})
                node["local_context"]["related_nodes"].append(target)

    for node in text_nodes:
        if node["node_type"] in {"table", "figure"}:
            section_name = str(node["local_context"]["parent_section"] or "").lower()
            section_peers = [node_id for node_id in section_lookup.get(section_name, []) if node_id != node["node_id"]][:5]
            for peer_id in section_peers:
                edges.append({"source": node["node_id"], "target": peer_id, "type": "contextual_neighbor", "score": 0.7})

    for i, left in enumerate(text_nodes):
        similarities: list[tuple[float, str]] = []
        for j, right in enumerate(text_nodes):
            if i == j:
                continue
            score = cosine_similarity(left["embedding"], right["embedding"])
            if score >= 0.72:
                similarities.append((score, right["node_id"]))
        similarities.sort(reverse=True)
        for score, target in similarities[:4]:
            edges.append({"source": left["node_id"], "target": target, "type": "semantic_similarity", "score": round(score, 4)})

    for node in ordered_nodes:
        dedup_related = []
        seen = set()
        for related in node["local_context"]["related_nodes"]:
            if related not in seen:
                dedup_related.append(related)
                seen.add(related)
        node["local_context"]["related_nodes"] = dedup_related

    graph_payload = {
        "doc_id": doc_id,
        "root_node_id": root_node_id,
        "paper_metadata": root_meta,
        "manifest": manifest,
        "nodes": ordered_nodes,
        "edges": edges,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embedding_model": EMBEDDING_BACKEND_STATE["name"],
        "graph_enrichment_backend": {
            "backend": backend,
            "model": model,
            "cloud_provider": cloud_provider if backend == "cloud" else "",
            "enrich_root": enrich_root,
            "enrichment_error": enrichment_error,
        },
    }
    write_json(graph_dir / "graph.json", graph_payload)

    light_nodes = []
    for node in ordered_nodes:
        light = dict(node)
        light.pop("embedding", None)
        light_nodes.append(light)
    write_json(graph_dir / "nodes.json", light_nodes)
    write_json(graph_dir / "edges.json", edges)

    manifest_payload = {
        "doc_id": doc_id,
        "graph_path": str(graph_dir / "graph.json"),
        "node_count": len(ordered_nodes),
        "edge_count": len(edges),
        "source_record_path": str(record_path),
        "source_pdf_name": manifest.get("source_pdf_name", ""),
        "embedding_model": EMBEDDING_BACKEND_STATE["name"],
        "graph_enrichment_backend": {
            "backend": backend,
            "model": model,
            "cloud_provider": cloud_provider if backend == "cloud" else "",
            "enrich_root": enrich_root,
            "enrichment_error": enrichment_error,
        },
    }
    write_json(graph_store.manifests / f"{doc_id}.json", manifest_payload)
    LOGGER.info("Graph complete | doc_id=%s | nodes=%s | edges=%s", doc_id, len(ordered_nodes), len(edges))
    return manifest_payload


def build_all_graphs(
    source_store: Path,
    graph_store: Path,
    *,
    backend: str,
    model: str,
    ollama_base_url: str,
    cloud_provider: str,
    api_key: str,
    base_url: str,
    enrich_root: bool,
) -> list[dict[str, Any]]:
    source_store = source_store.expanduser().resolve()
    store_paths = graph_store_paths(graph_store)
    ensure_graph_store(store_paths)
    results = []
    for record_file in list_record_files(source_store):
        results.append(
            build_document_graph(
                record_file,
                source_store,
                store_paths,
                backend=backend,
                model=model,
                ollama_base_url=ollama_base_url,
                cloud_provider=cloud_provider,
                api_key=api_key,
                base_url=base_url,
                enrich_root=enrich_root,
            )
        )
    write_json(store_paths.root / "catalog.json", results)
    return results


def load_graphs(graph_store: Path) -> list[dict[str, Any]]:
    store_paths = graph_store_paths(graph_store)
    graphs = []
    for graph_file in sorted(store_paths.graphs.glob("*/graph.json")):
        graph = read_json(graph_file, {})
        if graph:
            graphs.append(graph)
    return graphs


def heuristic_query_expansions(question: str) -> list[str]:
    base = normalize_space(question)
    expansions = [base]
    if "benchmark" in base.lower() or "dataset" in base.lower():
        expansions.append(f"{base} benchmark dataset evaluation table figure")
    if "method" in base.lower() or "model" in base.lower():
        expansions.append(f"{base} architecture training experiment results")
    expansions.append(f"{base} evidence section table figure citation")
    deduped = []
    seen = set()
    for item in expansions:
        key = item.lower()
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped[:4]


def generate_search_queries(
    question: str,
    backend: str,
    model: str,
    ollama_base_url: str,
    cloud_provider: str,
    api_key: str,
    base_url: str,
    max_queries: int = 4,
) -> list[str]:
    prompt = (
        "Generate up to 4 semantically distinct scientific retrieval queries for the user question. "
        "Each query should target complementary evidence such as methods, datasets, figures, tables, evaluations, or claims. "
        "Return JSON only in the form {\"queries\": [\"...\", \"...\"]}.\n\n"
        f"Question: {question}"
    )
    try:
        response = chat_complete(
            [{"role": "user", "content": prompt}],
            backend=backend,
            model=model,
            ollama_base_url=ollama_base_url,
            cloud_provider=cloud_provider,
            api_key=api_key,
            base_url=base_url,
        )
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            payload = json.loads(match.group(0))
            queries = [normalize_space(item) for item in payload.get("queries", []) if normalize_space(item)]
            if queries:
                return queries[:max_queries]
    except Exception as exc:
        LOGGER.warning("LLM query expansion failed, using heuristic fallback | %s", exc)
    return heuristic_query_expansions(question)[:max_queries]


def rank_graph_nodes(graphs: list[dict[str, Any]], queries: list[str], top_k_per_query: int = 10) -> list[dict[str, Any]]:
    query_vectors = sentence_embed(queries)
    aggregated: dict[str, dict[str, Any]] = {}

    for query_text, query_vec in zip(queries, query_vectors):
        LOGGER.info("Executing graph query | %s", query_text)
        candidates: list[dict[str, Any]] = []
        for graph in graphs:
            for node in graph.get("nodes", []):
                if node.get("node_type") == "paper_root":
                    continue
                score = cosine_similarity(query_vec, node.get("embedding", []))
                if score <= 0:
                    continue
                candidates.append(
                    {
                        "doc_id": graph.get("doc_id"),
                        "node_id": node.get("node_id"),
                        "score": score,
                        "query": query_text,
                        "node": node,
                    }
                )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        for candidate in candidates[:top_k_per_query]:
            key = candidate["node_id"]
            existing = aggregated.get(key)
            if existing is None:
                aggregated[key] = {
                    "doc_id": candidate["doc_id"],
                    "node_id": key,
                    "best_score": candidate["score"],
                    "queries": [candidate["query"]],
                    "node": candidate["node"],
                }
            else:
                existing["best_score"] = max(existing["best_score"], candidate["score"])
                if candidate["query"] not in existing["queries"]:
                    existing["queries"].append(candidate["query"])

    ranked = sorted(aggregated.values(), key=lambda item: item["best_score"], reverse=True)
    return ranked


def expand_with_relationships(graphs: list[dict[str, Any]], ranked_nodes: list[dict[str, Any]], limit: int = 14) -> list[dict[str, Any]]:
    graph_map = {graph["doc_id"]: graph for graph in graphs}
    selected: dict[str, dict[str, Any]] = {}
    for item in ranked_nodes[:limit]:
        selected[item["node_id"]] = item
        graph = graph_map.get(item["doc_id"], {})
        node_id = item["node_id"]
        neighbors = []
        for edge in graph.get("edges", []):
            if edge.get("source") == node_id and edge.get("type") in {"references_table", "references_figure", "contextual_neighbor", "same_section"}:
                neighbors.append(edge.get("target"))
            if edge.get("target") == node_id and edge.get("type") in {"references_table", "references_figure"}:
                neighbors.append(edge.get("source"))
        nodes_by_id = {node["node_id"]: node for node in graph.get("nodes", [])}
        for neighbor_id in neighbors[:5]:
            if neighbor_id in selected or neighbor_id not in nodes_by_id:
                continue
            selected[neighbor_id] = {
                "doc_id": item["doc_id"],
                "node_id": neighbor_id,
                "best_score": max(0.45, float(item["best_score"]) - 0.08),
                "queries": item["queries"],
                "node": nodes_by_id[neighbor_id],
            }
    return sorted(selected.values(), key=lambda item: item["best_score"], reverse=True)


def build_evidence_markdown(graphs: list[dict[str, Any]], selected_nodes: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    graph_map = {graph["doc_id"]: graph for graph in graphs}
    evidence_items = []
    lines = []
    for index, item in enumerate(selected_nodes, start=1):
        node = item["node"]
        graph = graph_map[item["doc_id"]]
        paper = graph.get("paper_metadata", {})
        evidence = {
            "evidence_id": f"E{index}",
            "doc_id": item["doc_id"],
            "paper_title": paper.get("title", ""),
            "source_pdf_name": node.get("paper_metadata", {}).get("source_pdf_name", ""),
            "node_id": node.get("node_id"),
            "node_type": node.get("node_type"),
            "section": node.get("local_context", {}).get("parent_section"),
            "asset_path": node.get("asset_path", ""),
            "score": round(float(item["best_score"]), 4),
            "queries": item["queries"],
            "content": node.get("content", ""),
        }
        evidence_items.append(evidence)
        lines.append(
            "\n".join(
                [
                    f"[{evidence['evidence_id']}] paper={evidence['paper_title']}",
                    f"pdf={evidence['source_pdf_name']}",
                    f"node={evidence['node_id']}",
                    f"type={evidence['node_type']}",
                    f"section={evidence['section']}",
                    f"score={evidence['score']}",
                    f"asset={evidence['asset_path']}",
                    f"queries={', '.join(evidence['queries'])}",
                    evidence["content"][:4000],
                ]
            )
        )
    return "\n\n".join(lines), evidence_items


def synthesize_answer(
    question: str,
    evidence_markdown: str,
    backend: str,
    model: str,
    ollama_base_url: str,
    cloud_provider: str,
    api_key: str,
    base_url: str,
) -> str:
    system = (
        "You are an expert scientific research synthesizer operating over document knowledge graphs. "
        "Use only the provided evidence. Answer with scientific rigor, cite evidence IDs like [E1], "
        "reference figures, tables, and sections when relevant, preserve uncertainty, and maintain traceability "
        "to original documents and artifact paths."
    )
    user = (
        f"Research question:\n{question}\n\n"
        "Retrieved graph evidence:\n"
        f"{evidence_markdown}\n\n"
        "Write a research-oriented response in Markdown. Include:\n"
        "- direct answer\n"
        "- evidence-backed synthesis\n"
        "- figures/tables/sections when relevant\n"
        "- limitations or uncertainty\n"
        "- short source traceability summary"
    )
    return chat_complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        backend=backend,
        model=model,
        ollama_base_url=ollama_base_url,
        cloud_provider=cloud_provider,
        api_key=api_key,
        base_url=base_url,
    )


def deterministic_fallback_answer(question: str, evidence_items: list[dict[str, Any]]) -> str:
    lines = [
        f"## Research Question\n\n{question}",
        "## Retrieval Summary",
        "",
        "LLM synthesis was unavailable, so this response is a deterministic evidence digest built from the knowledge-graph retrieval layer.",
        "",
    ]
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for item in evidence_items:
        by_paper.setdefault(item["paper_title"] or item["source_pdf_name"], []).append(item)

    for paper_title, items in list(by_paper.items())[:8]:
        items = sorted(items, key=lambda row: row["score"], reverse=True)
        top = items[0]
        lines.append(f"### {paper_title}")
        lines.append("")
        lines.append(f"- Top evidence: [{top['evidence_id']}] section `{top.get('section') or 'n/a'}` score `{top['score']}`")
        lines.append(f"- Source PDF: `{top.get('source_pdf_name', '')}`")
        if top.get("asset_path"):
            lines.append(f"- Artifact path: `{top['asset_path']}`")
        lines.append(f"- Evidence focus: `{top.get('node_type', '')}`")
        lines.append("")
        for evidence in items[:3]:
            snippet = normalize_space(evidence.get("content", ""))[:500]
            lines.append(f"- [{evidence['evidence_id']}] {snippet}")
        lines.append("")

    lines.append("## Traceability")
    lines.append("")
    for evidence in evidence_items[:12]:
        lines.append(
            f"- [{evidence['evidence_id']}] `{evidence['source_pdf_name']}` | section `{evidence.get('section') or 'n/a'}` | `{evidence.get('asset_path') or ''}`"
        )
    return "\n".join(lines).strip()


def run_query(
    question: str,
    graph_store: Path,
    backend: str,
    model: str,
    ollama_base_url: str,
    cloud_provider: str,
    api_key: str,
    base_url: str,
    max_queries: int,
    top_k_per_query: int,
) -> dict[str, Any]:
    store_paths = graph_store_paths(graph_store)
    graphs = load_graphs(graph_store)
    if not graphs:
        raise RuntimeError("No document graphs found. Run the `build` command first.")
    LOGGER.info("Loaded graphs | documents=%s", len(graphs))

    query_plan = generate_search_queries(
        question,
        backend=backend,
        model=model,
        ollama_base_url=ollama_base_url,
        cloud_provider=cloud_provider,
        api_key=api_key,
        base_url=base_url,
        max_queries=max_queries,
    )
    LOGGER.info("Query expansion complete | queries=%s", len(query_plan))
    ranked = rank_graph_nodes(graphs, query_plan, top_k_per_query=top_k_per_query)
    expanded = expand_with_relationships(graphs, ranked, limit=max(10, top_k_per_query))
    evidence_markdown, evidence_items = build_evidence_markdown(graphs, expanded[:18])
    synthesis_error = ""
    try:
        answer = synthesize_answer(
            question,
            evidence_markdown,
            backend=backend,
            model=model,
            ollama_base_url=ollama_base_url,
            cloud_provider=cloud_provider,
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as exc:
        synthesis_error = str(exc)
        LOGGER.warning("Synthesis backend unavailable, returning deterministic evidence digest | %s", exc)
        answer = deterministic_fallback_answer(question, evidence_items)
    payload = {
        "question": question,
        "query_plan": query_plan,
        "evidence_count": len(evidence_items),
        "evidence_items": evidence_items,
        "answer_markdown": answer,
        "synthesis_error": synthesis_error,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    write_json(store_paths.queries / f"query_{stamp}.json", payload)
    write_text(store_paths.queries / f"query_{stamp}.md", answer + "\n")
    return payload


def clean_graph_store(graph_store: Path) -> None:
    root = graph_store.expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
        LOGGER.info("Removed graph store | %s", root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scientific knowledge-graph multimodal RAG over multimodal_store artifacts.")
    parser.add_argument("--quiet", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build per-document knowledge graphs from multimodal_store indexes.")
    build.add_argument("--source-store", default=str(DEFAULT_SOURCE_STORE))
    build.add_argument("--graph-store", default=str(DEFAULT_GRAPH_STORE))
    build.add_argument("--backend", choices=["ollama", "cloud"], default="ollama")
    build.add_argument("--model", default=DEFAULT_TEXT_MODEL)
    build.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    build.add_argument("--cloud-provider", choices=["openai", "deepseek"], default="openai")
    build.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    build.add_argument("--base-url", default="")
    build.add_argument("--no-enrich-root", action="store_true", help="Skip LLM enrichment for paper root metadata.")

    query = subparsers.add_parser("query", help="Run graph-based scientific retrieval and synthesis.")
    query.add_argument("question")
    query.add_argument("--graph-store", default=str(DEFAULT_GRAPH_STORE))
    query.add_argument("--backend", choices=["ollama", "cloud"], default="ollama")
    query.add_argument("--model", default=DEFAULT_TEXT_MODEL)
    query.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    query.add_argument("--cloud-provider", choices=["openai", "deepseek"], default="openai")
    query.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    query.add_argument("--base-url", default="")
    query.add_argument("--max-queries", type=int, default=4)
    query.add_argument("--top-k-per-query", type=int, default=10)
    query.add_argument("--format", choices=["markdown", "json"], default="markdown")

    clean = subparsers.add_parser("clean", help="Remove the isolated graph RAG store.")
    clean.add_argument("--graph-store", default=str(DEFAULT_GRAPH_STORE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=not args.quiet)

    if args.command == "build":
        results = build_all_graphs(
            Path(args.source_store),
            Path(args.graph_store),
            backend=args.backend,
            model=args.model,
            ollama_base_url=args.ollama_base_url,
            cloud_provider=args.cloud_provider,
            api_key=args.api_key,
            base_url=args.base_url,
            enrich_root=not args.no_enrich_root,
        )
        print(
            json.dumps(
                {
                    "built_documents": len(results),
                    "graph_store": str(Path(args.graph_store).resolve()),
                    "backend": args.backend,
                    "model": args.model,
                    "cloud_provider": args.cloud_provider if args.backend == "cloud" else "",
                    "root_enrichment": not args.no_enrich_root,
                },
                indent=2,
            )
        )
        return

    if args.command == "query":
        payload = run_query(
            question=args.question,
            graph_store=Path(args.graph_store),
            backend=args.backend,
            model=args.model,
            ollama_base_url=args.ollama_base_url,
            cloud_provider=args.cloud_provider,
            api_key=args.api_key,
            base_url=args.base_url,
            max_queries=args.max_queries,
            top_k_per_query=args.top_k_per_query,
        )
        if args.format == "json":
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(payload["answer_markdown"])
        return

    if args.command == "clean":
        clean_graph_store(Path(args.graph_store))
        print(json.dumps({"removed": True, "graph_store": str(Path(args.graph_store).resolve())}, indent=2))
        return


if __name__ == "__main__":
    main()
