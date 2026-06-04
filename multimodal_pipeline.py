from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


DEFAULT_STORE_DIR = Path(os.getenv("MULTIMODAL_STORE_DIR", "./multimodal_store"))
DEFAULT_COLLECTION = os.getenv("MULTIMODAL_QDRANT_COLLECTION", "deepscipaper_multimodal")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEXT_MODEL = os.getenv("MULTIMODAL_TEXT_MODEL", os.getenv("DEFAULT_MODEL", "gpt-oss:20b"))
DEFAULT_IMAGE_MODEL = os.getenv("MULTIMODAL_IMAGE_MODEL", os.getenv("DEFAULT_MODEL", "gpt-oss:20b"))
FIXED_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_EMBED_MODEL = FIXED_EMBED_MODEL
DEFAULT_MINERU_BACKEND = os.getenv("MINERU_BACKEND", "pipeline")
DEFAULT_MINERU_METHOD = os.getenv("MINERU_METHOD", "auto")
DEFAULT_MINERU_LANG = os.getenv("MINERU_LANG", "en")


LOGGER = logging.getLogger("deepscipaper.multimodal")


def setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    LOGGER.setLevel(level)


@dataclass(frozen=True)
class StorePaths:
    root: Path
    documents: Path
    assets: Path
    indexes: Path
    manifests: Path
    qdrant: Path


def store_paths(root: Path | str = DEFAULT_STORE_DIR) -> StorePaths:
    root_path = Path(root).expanduser().resolve()
    return StorePaths(
        root=root_path,
        documents=root_path / "documents",
        assets=root_path / "assets",
        indexes=root_path / "indexes",
        manifests=root_path / "manifests",
        qdrant=root_path / "qdrant",
    )


def ensure_store(paths: StorePaths) -> None:
    for path in (paths.root, paths.documents, paths.assets, paths.indexes, paths.manifests, paths.qdrant):
        path.mkdir(parents=True, exist_ok=True)


def safe_slug(value: str, fallback: str = "document") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug[:120] or fallback


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{int(time.time() * 1000)}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp_path.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def list_pdfs(pdf_dir: Path | str) -> list[Path]:
    root = Path(pdf_dir)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*.pdf")
        if path.is_file() and ".ipynb_checkpoints" not in path.parts and not any(part.startswith(".") for part in path.parts)
    )


def doc_id_for_pdf(pdf_path: Path) -> str:
    return f"{safe_slug(pdf_path.stem)}__{sha256_file(pdf_path)[:12]}"


def find_mineru_command() -> str | None:
    return shutil.which("mineru") or shutil.which("magic-pdf")


def run_mineru(
    pdf_path: Path,
    doc_dir: Path,
    timeout_seconds: int = 1800,
    backend: str = DEFAULT_MINERU_BACKEND,
    method: str = DEFAULT_MINERU_METHOD,
    lang: str = DEFAULT_MINERU_LANG,
) -> dict[str, Any]:
    command_name = find_mineru_command()
    if not command_name:
        return {"ok": False, "reason": "MinerU CLI not found on PATH."}

    output_dir = doc_dir / "mineru"
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(command_name).name == "mineru":
        command = [
            command_name,
            "-p",
            str(pdf_path),
            "-o",
            str(output_dir),
            "-b",
            backend,
            "-m",
            method,
            "-l",
            lang,
        ]
    else:
        command = [command_name, "-p", str(pdf_path), "-o", str(output_dir)]

    LOGGER.info("MinerU start | pdf=%s | backend=%s | method=%s | lang=%s", pdf_path.name, backend, method, lang)
    LOGGER.info("MinerU command | %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    LOGGER.info("MinerU finished | pdf=%s | returncode=%s", pdf_path.name, result.returncode)
    if result.stdout.strip():
        LOGGER.info("MinerU stdout tail | %s", result.stdout[-1200:].strip())
    if result.stderr.strip():
        LOGGER.warning("MinerU stderr tail | %s", result.stderr[-1200:].strip())
    return {
        "ok": result.returncode == 0,
        "command": command,
        "backend": backend,
        "method": method,
        "lang": lang,
        "stdout": result.stdout[-8000:],
        "stderr": result.stderr[-8000:],
        "output_dir": str(output_dir),
    }


def find_mineru_outputs(doc_dir: Path) -> tuple[str, dict[str, Any], list[Path]]:
    mineru_dir = doc_dir / "mineru"
    markdown_files = sorted(mineru_dir.rglob("*.md"), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    json_files = sorted(mineru_dir.rglob("*.json"), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    image_files = sorted(
        path
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp")
        for path in mineru_dir.rglob(ext)
        if path.is_file()
    )
    markdown = markdown_files[0].read_text(encoding="utf-8", errors="replace") if markdown_files else ""
    raw_json = read_json(json_files[0], {}) if json_files else {}
    return markdown, raw_json, image_files


def fallback_parse_pdf(pdf_path: Path, doc_dir: Path, asset_dir: Path) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    import fitz

    LOGGER.warning("PyMuPDF fallback parser start | pdf=%s", pdf_path.name)
    document = fitz.open(pdf_path)
    pages: list[dict[str, Any]] = []
    markdown_parts: list[str] = [f"# {pdf_path.stem}"]
    extracted_images: list[dict[str, Any]] = []
    image_dir = asset_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        page_number = page_index + 1
        text = page.get_text("text").strip()
        pages.append({"page": page_number, "text": text})
        markdown_parts.append(f"\n\n## Page {page_number}\n\n{text}")

        for image_index, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]
            try:
                pixmap = fitz.Pixmap(document, xref)
                if pixmap.n >= 5:
                    pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                image_path = image_dir / f"page_{page_number:04d}_image_{image_index:02d}.png"
                pixmap.save(image_path)
                extracted_images.append({"page": page_number, "path": str(image_path), "xref": xref})
            except Exception:
                continue

    document.close()
    LOGGER.info("PyMuPDF fallback parser complete | pdf=%s | pages=%s | images=%s", pdf_path.name, len(pages), len(extracted_images))
    raw_json = {"parser": "pymupdf_fallback", "pages": pages, "images": extracted_images}
    return "\n".join(markdown_parts), raw_json, extracted_images


def extract_markdown_tables(markdown: str) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    current: list[str] = []
    current_section = "Document"
    for line in markdown.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if heading:
            current_section = heading.group(2).strip()
        if line.strip().startswith("|") and line.strip().endswith("|"):
            current.append(line)
        else:
            if len(current) >= 2:
                tables.append(
                    {
                        "table_id": f"table_{len(tables) + 1:04d}",
                        "markdown": "\n".join(current),
                        "section_title": current_section,
                    }
                )
            current = []
    if len(current) >= 2:
        tables.append(
            {
                "table_id": f"table_{len(tables) + 1:04d}",
                "markdown": "\n".join(current),
                "section_title": current_section,
            }
        )
    return tables


def infer_section_for_asset(markdown: str, asset_name: str) -> str:
    if not markdown.strip() or not asset_name.strip():
        return "Unknown section"
    lines = markdown.splitlines()
    current_section = "Document"
    target = asset_name.lower()
    for line in lines:
        heading = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if heading:
            current_section = heading.group(2).strip()
        if target in line.lower():
            return current_section
    return "Unknown section"


def split_markdown_sections(markdown: str, max_chars: int = 5000) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_title = "Document"
    current_lines: list[str] = []
    for line in markdown.splitlines():
        heading = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
        if heading and current_lines:
            sections.extend(chunk_section(current_title, "\n".join(current_lines), max_chars))
            current_title = heading.group(2).strip()
            current_lines = [line]
        else:
            if heading:
                current_title = heading.group(2).strip()
            current_lines.append(line)
    if current_lines:
        sections.extend(chunk_section(current_title, "\n".join(current_lines), max_chars))
    return sections


def chunk_section(title: str, text: str, max_chars: int) -> list[dict[str, Any]]:
    clean = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not clean:
        return []
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        if end < len(clean):
            boundary = clean.rfind("\n\n", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        chunks.append({"title": title, "text": clean[start:end].strip(), "chunk_index": len(chunks)})
        start = end
    return chunks


def ollama_chat(
    messages: list[dict[str, Any]],
    model: str,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_seconds: int = 180,
    num_predict: int = 900,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": num_predict},
    }
    response = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    content = ((response.json().get("message") or {}).get("content") or "").strip()
    content = re.sub(r"<think>[\s\S]*?</think>\s*", "", content, flags=re.IGNORECASE).strip()
    return content


def caption_image(
    image_path: Path,
    source_pdf_name: str,
    section_title: str,
    model: str = DEFAULT_IMAGE_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> dict[str, Any]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = (
        "You are creating retrieval-ready scientific figure notes.\n"
        f"Source paper: {source_pdf_name}\n"
        f"Source section: {section_title}\n"
        f"Figure file: {image_path.name}\n\n"
        "Describe this figure or graph in detail. Include visible axes, labels, trends, variables, "
        "units, key findings, uncertainty/limitations, and why this figure is relevant to model training, "
        "benchmarking, datasets, or evaluation in this source section. If not a chart, provide a precise technical description."
    )
    try:
        text = ollama_chat(
            [{"role": "user", "content": prompt, "images": [image_b64]}],
            model=model,
            base_url=base_url,
            timeout_seconds=240,
            num_predict=1000,
        )
        return {"ok": bool(text), "caption": text, "model": model}
    except Exception as exc:
        return {"ok": False, "caption": "", "model": model, "error": str(exc)}


def summarize_table(
    table_markdown: str,
    source_pdf_name: str,
    section_title: str,
    table_id: str,
    model: str = DEFAULT_TEXT_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> dict[str, Any]:
    prompt = (
        "You are creating retrieval-ready scientific table notes.\n"
        f"Source paper: {source_pdf_name}\n"
        f"Source section: {section_title}\n"
        f"Table id: {table_id}\n\n"
        "Summarize the table for high-quality retrieval. Name variables, population/samples, model names, "
        "metrics, units, strongest comparisons, statistical caveats, and direct implications for dataset design, "
        "training strategy, or benchmark evaluation.\n\n"
        f"{table_markdown[:7000]}"
    )
    try:
        text = ollama_chat(
            [{"role": "user", "content": prompt}],
            model=model,
            base_url=base_url,
            timeout_seconds=180,
            num_predict=800,
        )
        return {"ok": bool(text), "summary": text, "model": model}
    except Exception as exc:
        return {"ok": False, "summary": "", "model": model, "error": str(exc)}


def clean_embedding_text(text: str, max_chars: int = 12000) -> str:
    clean = re.sub(r"\s+", " ", (text or "").replace("\x00", " ")).strip()
    return (clean or "empty retrieval text")[:max_chars]


@lru_cache(maxsize=1)
def load_sentence_embedding_model(model_name: str = FIXED_EMBED_MODEL):
    from sentence_transformers import SentenceTransformer

    LOGGER.info("Loading sentence embedding model | model=%s", model_name)
    return SentenceTransformer(model_name)


def sentence_embed(texts: list[str], requested_model: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    if requested_model != FIXED_EMBED_MODEL:
        LOGGER.warning(
            "Embedding model override requested (%s) but ignored. Using fixed model %s.",
            requested_model,
            FIXED_EMBED_MODEL,
        )
    cleaned_texts = [clean_embedding_text(text) for text in texts]
    model = load_sentence_embedding_model(FIXED_EMBED_MODEL)
    vectors = model.encode(cleaned_texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def qdrant_client(paths: StorePaths):
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:
        raise RuntimeError("Install qdrant-client with `python -m pip install -r requirements.txt`.") from exc
    return QdrantClient(path=str(paths.qdrant))


def ensure_qdrant_collection(client: Any, collection: str, vector_size: int) -> None:
    from qdrant_client import models

    existing = {item.name for item in client.get_collections().collections}
    if collection in existing:
        try:
            collection_info = client.get_collection(collection_name=collection)
            vectors_cfg = collection_info.config.params.vectors
            existing_size = None
            if hasattr(vectors_cfg, "size"):
                existing_size = int(vectors_cfg.size)
            elif isinstance(vectors_cfg, dict) and vectors_cfg:
                first_cfg = next(iter(vectors_cfg.values()))
                if hasattr(first_cfg, "size"):
                    existing_size = int(first_cfg.size)
                elif isinstance(first_cfg, dict) and "size" in first_cfg:
                    existing_size = int(first_cfg["size"])
            if existing_size and existing_size != vector_size:
                LOGGER.warning(
                    "Qdrant collection vector size mismatch | collection=%s | existing=%s | required=%s | recreating",
                    collection,
                    existing_size,
                    vector_size,
                )
                client.delete_collection(collection_name=collection)
            else:
                return
        except Exception as exc:
            LOGGER.warning("Could not inspect existing Qdrant collection `%s`: %s", collection, exc)
            return
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def index_records(
    records: list[dict[str, Any]],
    paths: StorePaths,
    collection: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> dict[str, Any]:
    if not records:
        LOGGER.warning("Index skipped | no records")
        return {"indexed": 0, "collection": collection}
    from qdrant_client import models

    texts = [record["text_for_embedding"][:8000] for record in records]
    LOGGER.info("Embedding records | count=%s | model=%s", len(texts), FIXED_EMBED_MODEL)
    embeddings = sentence_embed(texts, requested_model=embed_model)
    if not embeddings:
        LOGGER.warning("Index skipped | embedding backend returned no vectors")
        return {"indexed": 0, "collection": collection}
    client = qdrant_client(paths)
    ensure_qdrant_collection(client, collection, len(embeddings[0]))
    LOGGER.info("Qdrant collection ready | collection=%s | vector_size=%s", collection, len(embeddings[0]))
    points = []
    for record, vector in zip(records, embeddings):
        point_id = int(hashlib.sha1(record["record_id"].encode("utf-8")).hexdigest()[:15], 16)
        points.append(models.PointStruct(id=point_id, vector=vector, payload=record))
    client.upsert(collection_name=collection, points=points)
    LOGGER.info("Qdrant upsert complete | collection=%s | points=%s", collection, len(points))
    return {"indexed": len(points), "collection": collection}


def search_records(
    query: str,
    paths: StorePaths,
    collection: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    limit: int = 8,
) -> list[dict[str, Any]]:
    client = qdrant_client(paths)
    LOGGER.info("Query embedding start | model=%s | limit=%s", FIXED_EMBED_MODEL, limit)
    query_vector = sentence_embed([query], requested_model=embed_model)[0]
    try:
        if hasattr(client, "query_points"):
            query_response = client.query_points(collection_name=collection, query=query_vector, limit=limit, with_payload=True)
            results = getattr(query_response, "points", query_response)
        else:
            results = client.search(collection_name=collection, query_vector=query_vector, limit=limit, with_payload=True)
    except Exception as exc:
        message = str(exc)
        if "not aligned" in message or "Vector dimension error" in message:
            raise RuntimeError(
                "Qdrant index dimension mismatch detected. Reset and rebuild the multimodal index with the fixed "
                f"embedding model `{FIXED_EMBED_MODEL}` using:\n"
                "`python multimodal_pipeline.py clean --store ./multimodal_store`\n"
                "`python multimodal_pipeline.py ingest --pdf-dir ./bib_pdf --store ./multimodal_store --force`"
            ) from exc
        raise
    LOGGER.info("Qdrant retrieval complete | collection=%s | hits=%s", collection, len(results))
    return [{"score": item.score, **(item.payload or {})} for item in results]


def ingest_pdf(
    pdf_path: Path,
    store_root: Path | str = DEFAULT_STORE_DIR,
    text_model: str = DEFAULT_TEXT_MODEL,
    image_model: str = DEFAULT_IMAGE_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    use_mineru: bool = True,
    mineru_backend: str = DEFAULT_MINERU_BACKEND,
    mineru_method: str = DEFAULT_MINERU_METHOD,
    mineru_lang: str = DEFAULT_MINERU_LANG,
    mineru_timeout: int = 1800,
    enrich_images: bool = False,
    enrich_tables: bool = True,
    index: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    paths = store_paths(store_root)
    ensure_store(paths)
    pdf_path = pdf_path.resolve()
    digest = sha256_file(pdf_path)
    doc_id = doc_id_for_pdf(pdf_path)
    doc_dir = paths.documents / doc_id
    asset_dir = paths.assets / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = paths.manifests / f"{doc_id}.json"
    existing = read_json(manifest_path, {})
    if not force and existing.get("source_sha256") == digest and existing.get("status") == "indexed":
        LOGGER.info("Skipping already indexed PDF | pdf=%s | doc_id=%s", pdf_path.name, doc_id)
        return existing | {"skipped": True}

    LOGGER.info("Ingest start | pdf=%s | doc_id=%s | mineru=%s | enrich_tables=%s | enrich_images=%s | index=%s", pdf_path.name, doc_id, use_mineru, enrich_tables, enrich_images, index)
    mineru_result = {"ok": False, "reason": "disabled"}
    markdown = ""
    raw_json: dict[str, Any] = {}
    images: list[dict[str, Any]] = []

    if use_mineru:
        try:
            mineru_result = run_mineru(
                pdf_path,
                doc_dir,
                timeout_seconds=mineru_timeout,
                backend=mineru_backend,
                method=mineru_method,
                lang=mineru_lang,
            )
            if mineru_result.get("ok"):
                markdown, raw_json, mineru_images = find_mineru_outputs(doc_dir)
                images = [{"path": str(path), "page": None, "source": "mineru"} for path in mineru_images]
                LOGGER.info("MinerU artifacts detected | pdf=%s | markdown_chars=%s | raw_json=%s | images=%s", pdf_path.name, len(markdown), bool(raw_json), len(images))
            else:
                LOGGER.warning("MinerU failed | pdf=%s | reason=%s", pdf_path.name, mineru_result.get("reason") or "see stderr")
        except Exception as exc:
            mineru_result = {"ok": False, "reason": str(exc)}
            LOGGER.exception("MinerU exception | pdf=%s", pdf_path.name)

    if not markdown:
        markdown, raw_json, images = fallback_parse_pdf(pdf_path, doc_dir, asset_dir)

    markdown_path = doc_dir / "document.md"
    raw_json_path = doc_dir / "raw.json"
    write_text(markdown_path, markdown)
    atomic_write_json(raw_json_path, raw_json)

    tables = extract_markdown_tables(markdown)
    LOGGER.info("Artifact split | pdf=%s | markdown_chars=%s | tables=%s | images=%s", pdf_path.name, len(markdown), len(tables), len(images))
    table_records = []
    for table_index, table in enumerate(tables, start=1):
        LOGGER.info("Table enrichment | pdf=%s | table=%s/%s | enabled=%s", pdf_path.name, table_index, len(tables), enrich_tables)
        summary = (
            summarize_table(
                table["markdown"],
                source_pdf_name=pdf_path.name,
                section_title=str(table.get("section_title") or "Unknown section"),
                table_id=table["table_id"],
                model=text_model,
                base_url=ollama_base_url,
            )
            if enrich_tables
            else {"ok": False, "summary": ""}
        )
        table_path = doc_dir / "tables" / f"{table['table_id']}.md"
        write_text(table_path, table["markdown"])
        table_records.append({**table, "path": str(table_path), "summary": summary})

    image_records = []
    for image_index, image in enumerate(images, start=1):
        image_path = Path(image["path"])
        inferred_section = infer_section_for_asset(markdown, image_path.name)
        LOGGER.info("Image enrichment | pdf=%s | image=%s/%s | enabled=%s | path=%s", pdf_path.name, image_index, len(images), enrich_images, image_path.name)
        caption = (
            caption_image(
                image_path,
                source_pdf_name=pdf_path.name,
                section_title=inferred_section,
                model=image_model,
                base_url=ollama_base_url,
            )
            if enrich_images
            else {"ok": False, "caption": ""}
        )
        image_records.append({**image, "image_id": f"image_{image_index:04d}", "section_title": inferred_section, "caption": caption})

    section_chunks = split_markdown_sections(markdown)
    LOGGER.info("Text chunking complete | pdf=%s | chunks=%s", pdf_path.name, len(section_chunks))
    records: list[dict[str, Any]] = []
    for section in section_chunks:
        record_id = f"{doc_id}:text:{section['chunk_index']:04d}:{hashlib.sha1(section['text'].encode()).hexdigest()[:8]}"
        records.append(
            {
                "record_id": record_id,
                "modality": "text",
                "doc_id": doc_id,
                "source_pdf": str(pdf_path),
                "source_pdf_name": pdf_path.name,
                "title": section["title"],
                "asset_path": str(markdown_path),
                "text_for_embedding": f"{section['title']}\n{section['text']}",
                "display_text": section["text"][:3500],
            }
        )

    for table in table_records:
        summary_text = table.get("summary", {}).get("summary") or table["markdown"]
        records.append(
            {
                "record_id": f"{doc_id}:table:{table['table_id']}",
                "modality": "table",
                "doc_id": doc_id,
                "source_pdf": str(pdf_path),
                "source_pdf_name": pdf_path.name,
                "title": table["table_id"],
                "section_title": table.get("section_title", "Unknown section"),
                "asset_path": table["path"],
                "text_for_embedding": summary_text,
                "display_text": table["markdown"][:3500],
            }
        )

    for image in image_records:
        caption_text = image.get("caption", {}).get("caption") or f"Image extracted from {pdf_path.name}"
        records.append(
            {
                "record_id": f"{doc_id}:image:{image['image_id']}",
                "modality": "image",
                "doc_id": doc_id,
                "source_pdf": str(pdf_path),
                "source_pdf_name": pdf_path.name,
                "title": image["image_id"],
                "section_title": image.get("section_title", "Unknown section"),
                "asset_path": image["path"],
                "text_for_embedding": caption_text,
                "display_text": caption_text[:3500],
            }
        )

    index_result = index_records(records, paths, embed_model=embed_model, ollama_base_url=ollama_base_url) if index else {"indexed": 0}
    manifest = {
        "doc_id": doc_id,
        "status": "indexed" if index else "parsed",
        "source_pdf": str(pdf_path),
        "source_pdf_name": pdf_path.name,
        "source_sha256": digest,
        "indexed_at": datetime.now().isoformat(timespec="seconds"),
        "parser": "mineru" if mineru_result.get("ok") else "pymupdf_fallback",
        "mineru": mineru_result,
        "markdown_path": str(markdown_path),
        "raw_json_path": str(raw_json_path),
        "text_chunks": len(section_chunks),
        "tables": len(table_records),
        "images": len(image_records),
        "vector_records": len(records),
        "embedding_model": FIXED_EMBED_MODEL,
        "index": index_result,
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(paths.indexes / f"{doc_id}_records.json", records)
    LOGGER.info("Ingest complete | pdf=%s | parser=%s | records=%s | status=%s", pdf_path.name, manifest["parser"], len(records), manifest["status"])
    return manifest


def ingest_pdf_dir(
    pdf_dir: Path | str,
    store_root: Path | str = DEFAULT_STORE_DIR,
    limit: int | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    pdfs = list_pdfs(pdf_dir)
    if limit:
        pdfs = pdfs[:limit]
    LOGGER.info("Batch ingest start | pdf_dir=%s | pdf_count=%s | limit=%s", Path(pdf_dir).resolve(), len(pdfs), limit or "all")
    results = []
    for pdf_index, pdf_path in enumerate(pdfs, start=1):
        LOGGER.info("Batch progress | %s/%s | %s", pdf_index, len(pdfs), pdf_path.name)
        try:
            results.append(ingest_pdf(pdf_path, store_root=store_root, **kwargs))
        except Exception as exc:
            LOGGER.exception("Batch ingest error | pdf=%s", pdf_path.name)
            results.append({"status": "error", "source_pdf": str(pdf_path), "error": str(exc)})
    LOGGER.info("Batch ingest complete | processed=%s | errors=%s", len(results), sum(1 for item in results if item.get("status") == "error"))
    return results


def clean_store(store_root: Path | str = DEFAULT_STORE_DIR) -> dict[str, Any]:
    paths = store_paths(store_root)
    if not paths.root.exists():
        LOGGER.info("Clean skipped | store does not exist | %s", paths.root)
        return {"removed": False, "store": str(paths.root), "reason": "missing"}
    LOGGER.warning("Removing multimodal store | %s", paths.root)
    shutil.rmtree(paths.root)
    return {"removed": True, "store": str(paths.root)}


def multimodal_answer(
    question: str,
    store_root: Path | str = DEFAULT_STORE_DIR,
    text_model: str = DEFAULT_TEXT_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    limit: int = 10,
) -> dict[str, Any]:
    paths = store_paths(store_root)
    LOGGER.info("Agentic query start | question=%s", question[:180])
    evidence = search_records(question, paths, embed_model=embed_model, ollama_base_url=ollama_base_url, limit=limit)
    LOGGER.info("Agentic planner | routing over modalities=text/table/image | evidence_items=%s", len(evidence))
    evidence_text = "\n\n".join(
        (
            f"[E{index}] modality={item.get('modality')} score={item.get('score'):.3f} "
            f"source={item.get('source_pdf_name')} title={item.get('title')} asset={item.get('asset_path')}\n"
            f"{item.get('display_text', '')}"
        )
        for index, item in enumerate(evidence, start=1)
    )
    prompt = (
        "You are a multimodal scientific research synthesizer. Answer using only the retrieved evidence. "
        "Return concise Markdown. Use short sections and bullets instead of a large table. "
        "Cite every factual claim with evidence IDs like [E1], [E2]. "
        "Do not invent paper titles, metrics, datasets, figures, tables, or artifact paths. "
        "Only mention an image/table artifact if the matching evidence item has modality=image or modality=table. "
        "If evidence is insufficient, say so clearly.\n\n"
        f"Question:\n{question}\n\nEvidence:\n{evidence_text}"
    )
    LOGGER.info("Synthesis start | model=%s", text_model)
    answer = ollama_chat([{"role": "user", "content": prompt}], model=text_model, base_url=ollama_base_url, timeout_seconds=240, num_predict=2400)
    evidence_references = [
        {
            "index": index,
            "modality": item.get("modality"),
            "source_pdf_name": item.get("source_pdf_name"),
            "asset_path": item.get("asset_path"),
            "score": item.get("score"),
            "title": item.get("title"),
        }
        for index, item in enumerate(evidence, start=1)
    ]
    artifact_references = [item for item in evidence_references if item.get("modality") in {"table", "image"}]
    artifact_markdown = "\n".join(
        (
            f"- [{item['index']}] `{item.get('modality')}` | `{item.get('source_pdf_name')}` | "
            f"`{item.get('title') or ''}` | `{item.get('asset_path')}`"
        )
        for item in artifact_references
    )
    answer_markdown = answer.strip()
    if artifact_markdown:
        answer_markdown = f"{answer_markdown}\n\n## Image And Table Artifact References\n{artifact_markdown}"
    LOGGER.info("Synthesis complete | answer_chars=%s | artifact_refs=%s", len(answer_markdown), len(artifact_references))
    return {
        "answer": answer_markdown,
        "answer_markdown": answer_markdown,
        "artifact_references": artifact_references,
        "evidence_references": evidence_references,
        "evidence": evidence,
    }


def store_summary(store_root: Path | str = DEFAULT_STORE_DIR) -> dict[str, Any]:
    paths = store_paths(store_root)
    manifests = sorted(paths.manifests.glob("*.json")) if paths.manifests.exists() else []
    payloads = [read_json(path, {}) for path in manifests]
    return {
        "documents": len(payloads),
        "text_chunks": sum(int(item.get("text_chunks", 0)) for item in payloads),
        "tables": sum(int(item.get("tables", 0)) for item in payloads),
        "images": sum(int(item.get("images", 0)) for item in payloads),
        "vector_records": sum(int(item.get("vector_records", 0)) for item in payloads),
        "last_indexed": max((item.get("indexed_at", "") for item in payloads), default=""),
    }


def watch_drop_zone(pdf_dir: Path, poll_seconds: int = 5, **kwargs: Any) -> None:
    known = {path: sha256_file(path) for path in list_pdfs(pdf_dir)}
    print(f"Watching {pdf_dir} for new PDFs.")
    while True:
        for pdf_path in list_pdfs(pdf_dir):
            digest = sha256_file(pdf_path)
            if known.get(pdf_path) != digest:
                print(f"Ingesting {pdf_path}")
                ingest_pdf(pdf_path, **kwargs)
                known[pdf_path] = digest
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSciPaper multimodal ingestion and RAG pipeline")
    parser.add_argument("--quiet", action="store_true", help="Only print warnings/errors.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--pdf-dir", default="./bib_pdf")
    ingest_parser.add_argument("--store", default=str(DEFAULT_STORE_DIR))
    ingest_parser.add_argument("--limit", type=int, default=None)
    ingest_parser.add_argument("--skip-mineru", action="store_true")
    ingest_parser.add_argument("--mineru-backend", default=DEFAULT_MINERU_BACKEND)
    ingest_parser.add_argument("--mineru-method", default=DEFAULT_MINERU_METHOD)
    ingest_parser.add_argument("--mineru-lang", default=DEFAULT_MINERU_LANG)
    ingest_parser.add_argument("--mineru-timeout", type=int, default=1800)
    ingest_parser.add_argument("--enrich-images", action="store_true")
    ingest_parser.add_argument("--skip-tables", action="store_true")
    ingest_parser.add_argument("--skip-index", action="store_true")
    ingest_parser.add_argument("--force", action="store_true", help="Rebuild artifacts even when the PDF hash is already indexed.")
    ingest_parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    ingest_parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    ingest_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ingest_parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("question")
    query_parser.add_argument("--store", default=str(DEFAULT_STORE_DIR))
    query_parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    query_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    query_parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    query_parser.add_argument("--limit", type=int, default=10)
    query_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")

    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("--store", default=str(DEFAULT_STORE_DIR))

    watch_parser = subparsers.add_parser("watch")
    watch_parser.add_argument("--pdf-dir", default="./bib_pdf")
    watch_parser.add_argument("--store", default=str(DEFAULT_STORE_DIR))
    watch_parser.add_argument("--poll-seconds", type=int, default=5)

    args = parser.parse_args()
    setup_logging(verbose=not args.quiet)
    requested_embed_model = getattr(args, "embed_model", FIXED_EMBED_MODEL)
    if requested_embed_model != FIXED_EMBED_MODEL:
        LOGGER.warning("Ignoring requested embed model `%s`; fixed model `%s` is always used.", requested_embed_model, FIXED_EMBED_MODEL)
    if args.command == "ingest":
        results = ingest_pdf_dir(
            args.pdf_dir,
            store_root=args.store,
            limit=args.limit,
            text_model=args.text_model,
            image_model=args.image_model,
            embed_model=args.embed_model,
            ollama_base_url=args.ollama_base_url,
            use_mineru=not args.skip_mineru,
            mineru_backend=args.mineru_backend,
            mineru_method=args.mineru_method,
            mineru_lang=args.mineru_lang,
            mineru_timeout=args.mineru_timeout,
            enrich_images=args.enrich_images,
            enrich_tables=not args.skip_tables,
            index=not args.skip_index,
            force=args.force,
        )
        print(json.dumps({"count": len(results), "summary": store_summary(args.store), "results": results}, indent=2, ensure_ascii=False))
    elif args.command == "query":
        response = multimodal_answer(args.question, args.store, args.text_model, args.embed_model, args.ollama_base_url, args.limit)
        if args.format == "json":
            print(json.dumps(response, indent=2, ensure_ascii=False))
        else:
            print(response["answer_markdown"])
    elif args.command == "clean":
        print(json.dumps(clean_store(args.store), indent=2, ensure_ascii=False))
    elif args.command == "watch":
        watch_drop_zone(Path(args.pdf_dir), store_root=args.store)


if __name__ == "__main__":
    main()
