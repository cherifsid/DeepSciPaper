from __future__ import annotations

import io
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_slug(value: str, fallback: str = "case") -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-").lower()
    return slug[:80] or fallback


@dataclass(frozen=True)
class CaseRecord:
    id: int
    slug: str
    name: str
    description: str
    research_goal: str
    root_dir: str
    bib_pdf_dir: str
    multimodal_store_dir: str
    graph_store_dir: str
    compiled_output_dir: str
    paper_tex_path: str
    minio_prefix: str
    is_legacy: bool
    created_at: str
    updated_at: str


def repo_state_dir(root: Path) -> Path:
    return root / "app_state"


def db_path(root: Path) -> Path:
    return repo_state_dir(root) / "research_copilot.db"


def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_case(row: sqlite3.Row | None) -> CaseRecord | None:
    if row is None:
        return None
    return CaseRecord(
        id=int(row["id"]),
        slug=str(row["slug"]),
        name=str(row["name"]),
        description=str(row["description"] or ""),
        research_goal=str(row["research_goal"] or ""),
        root_dir=str(row["root_dir"]),
        bib_pdf_dir=str(row["bib_pdf_dir"]),
        multimodal_store_dir=str(row["multimodal_store_dir"]),
        graph_store_dir=str(row["graph_store_dir"]),
        compiled_output_dir=str(row["compiled_output_dir"]),
        paper_tex_path=str(row["paper_tex_path"]),
        minio_prefix=str(row["minio_prefix"]),
        is_legacy=bool(int(row["is_legacy"])),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def init_db(database_path: Path) -> None:
    with connect(database_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                research_goal TEXT DEFAULT '',
                root_dir TEXT NOT NULL,
                bib_pdf_dir TEXT NOT NULL,
                multimodal_store_dir TEXT NOT NULL,
                graph_store_dir TEXT NOT NULL,
                compiled_output_dir TEXT NOT NULL,
                paper_tex_path TEXT NOT NULL,
                minio_prefix TEXT NOT NULL,
                is_legacy INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                mode TEXT DEFAULT 'standard',
                content TEXT NOT NULL,
                artifacts_json TEXT DEFAULT '[]',
                evidence_json TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(id)
            );

            CREATE TABLE IF NOT EXISTS sync_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_count INTEGER NOT NULL DEFAULT 0,
                message TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(id)
            );
            """
        )
        chat_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()}
        if "evidence_json" not in chat_columns:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN evidence_json TEXT DEFAULT '[]'")


def list_cases(database_path: Path) -> list[CaseRecord]:
    with connect(database_path) as conn:
        rows = conn.execute("SELECT * FROM cases ORDER BY is_legacy DESC, updated_at DESC, name ASC").fetchall()
    return [row_to_case(row) for row in rows if row_to_case(row) is not None]


def get_case_by_slug(database_path: Path, slug: str) -> CaseRecord | None:
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM cases WHERE slug = ?", (slug,)).fetchone()
    return row_to_case(row)


def get_case_by_id(database_path: Path, case_id: int) -> CaseRecord | None:
    with connect(database_path) as conn:
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    return row_to_case(row)


def _next_available_slug(database_path: Path, requested_slug: str) -> str:
    slug = safe_slug(requested_slug)
    if get_case_by_slug(database_path, slug) is None:
        return slug
    index = 2
    while get_case_by_slug(database_path, f"{slug}-{index}") is not None:
        index += 1
    return f"{slug}-{index}"


def case_paths(case: CaseRecord) -> dict[str, Path]:
    return {
        "root": Path(case.root_dir),
        "bib_pdf": Path(case.bib_pdf_dir),
        "multimodal_store": Path(case.multimodal_store_dir),
        "graph_store": Path(case.graph_store_dir),
        "compiled_output": Path(case.compiled_output_dir),
        "paper_tex": Path(case.paper_tex_path),
    }


def ensure_legacy_case(database_path: Path, root: Path, workspace_title: str) -> CaseRecord | None:
    def root_has_legacy_content(repo_root: Path) -> bool:
        bib_pdf = repo_root / "bib_pdf"
        multimodal_manifests = repo_root / "multimodal_store" / "manifests"
        graph_manifests = repo_root / "scientific_graph_rag_store" / "manifests"
        compiled_output = repo_root / "compiled_output"
        if bib_pdf.exists() and any(path.is_file() for path in bib_pdf.iterdir()):
            return True
        if multimodal_manifests.exists() and any(path.is_file() for path in multimodal_manifests.iterdir()):
            return True
        if graph_manifests.exists() and any(path.is_file() for path in graph_manifests.iterdir()):
            return True
        if compiled_output.exists() and any(path.is_file() and path.suffix.lower() == ".pdf" for path in compiled_output.iterdir()):
            return True
        return False

    legacy = get_case_by_slug(database_path, "legacy-workspace")
    if not root_has_legacy_content(root):
        if legacy is not None:
            with connect(database_path) as conn:
                conn.execute("DELETE FROM chat_messages WHERE case_id = ?", (legacy.id,))
                conn.execute("DELETE FROM sync_events WHERE case_id = ?", (legacy.id,))
                conn.execute("DELETE FROM cases WHERE id = ?", (legacy.id,))
        return None
    now = utc_now_iso()
    payload = {
        "slug": "legacy-workspace",
        "name": workspace_title,
        "description": "Existing workspace data discovered at the repository root.",
        "research_goal": "Preserve and browse the original deep-search workspace.",
        "root_dir": str(root.resolve()),
        "bib_pdf_dir": str((root / "bib_pdf").resolve()),
        "multimodal_store_dir": str((root / "multimodal_store").resolve()),
        "graph_store_dir": str((root / "scientific_graph_rag_store").resolve()),
        "compiled_output_dir": str((root / "compiled_output").resolve()),
        "paper_tex_path": str((root / "paper.tex").resolve()),
        "minio_prefix": "cases/legacy-workspace",
        "is_legacy": 1,
    }
    with connect(database_path) as conn:
        if legacy is None:
            conn.execute(
                """
                INSERT INTO cases (
                    slug, name, description, research_goal, root_dir, bib_pdf_dir,
                    multimodal_store_dir, graph_store_dir, compiled_output_dir,
                    paper_tex_path, minio_prefix, is_legacy, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["slug"],
                    payload["name"],
                    payload["description"],
                    payload["research_goal"],
                    payload["root_dir"],
                    payload["bib_pdf_dir"],
                    payload["multimodal_store_dir"],
                    payload["graph_store_dir"],
                    payload["compiled_output_dir"],
                    payload["paper_tex_path"],
                    payload["minio_prefix"],
                    payload["is_legacy"],
                    now,
                    now,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE cases
                SET name = ?, description = ?, research_goal = ?, root_dir = ?, bib_pdf_dir = ?,
                    multimodal_store_dir = ?, graph_store_dir = ?, compiled_output_dir = ?,
                    paper_tex_path = ?, minio_prefix = ?, updated_at = ?
                WHERE slug = ?
                """,
                (
                    payload["name"],
                    payload["description"],
                    payload["research_goal"],
                    payload["root_dir"],
                    payload["bib_pdf_dir"],
                    payload["multimodal_store_dir"],
                    payload["graph_store_dir"],
                    payload["compiled_output_dir"],
                    payload["paper_tex_path"],
                    payload["minio_prefix"],
                    now,
                    payload["slug"],
                ),
            )
    return get_case_by_slug(database_path, "legacy-workspace")  # type: ignore[return-value]


def create_case(database_path: Path, root: Path, name: str, description: str = "", research_goal: str = "") -> CaseRecord:
    slug = _next_available_slug(database_path, name)
    case_root = (root / "cases" / slug).resolve()
    bib_pdf = case_root / "bib_pdf"
    multimodal_store = case_root / "multimodal_store"
    graph_store = case_root / "scientific_graph_rag_store"
    compiled_output = case_root / "compiled_output"
    paper_tex = case_root / "paper.tex"
    for path in (case_root, bib_pdf, multimodal_store, graph_store, compiled_output):
        path.mkdir(parents=True, exist_ok=True)

    now = utc_now_iso()
    with connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO cases (
                slug, name, description, research_goal, root_dir, bib_pdf_dir,
                multimodal_store_dir, graph_store_dir, compiled_output_dir,
                paper_tex_path, minio_prefix, is_legacy, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                slug,
                name.strip() or slug,
                description.strip(),
                research_goal.strip(),
                str(case_root),
                str(bib_pdf),
                str(multimodal_store),
                str(graph_store),
                str(compiled_output),
                str(paper_tex),
                f"cases/{slug}",
                now,
                now,
            ),
        )
    return get_case_by_slug(database_path, slug)  # type: ignore[return-value]


def update_case_metadata(database_path: Path, case_id: int, *, name: str, description: str, research_goal: str) -> CaseRecord | None:
    now = utc_now_iso()
    with connect(database_path) as conn:
        conn.execute(
            """
            UPDATE cases
            SET name = ?, description = ?, research_goal = ?, updated_at = ?
            WHERE id = ?
            """,
            (name.strip(), description.strip(), research_goal.strip(), now, case_id),
        )
    return get_case_by_id(database_path, case_id)


def list_chat_messages(database_path: Path, case_id: int) -> list[dict[str, Any]]:
    with connect(database_path) as conn:
        rows = conn.execute(
            "SELECT role, mode, content, artifacts_json, evidence_json, created_at FROM chat_messages WHERE case_id = ? ORDER BY id ASC",
            (case_id,),
        ).fetchall()
    messages: list[dict[str, Any]] = []
    for row in rows:
        try:
            artifacts = json.loads(row["artifacts_json"] or "[]")
        except Exception:
            artifacts = []
        try:
            evidence = json.loads(row["evidence_json"] or "[]")
        except Exception:
            evidence = []
        messages.append(
            {
                "role": str(row["role"]),
                "mode": str(row["mode"] or "standard"),
                "content": str(row["content"]),
                "artifacts": artifacts,
                "evidence": evidence,
                "created_at": str(row["created_at"]),
            }
        )
    return messages


def append_chat_message(
    database_path: Path,
    case_id: int,
    *,
    role: str,
    content: str,
    mode: str = "standard",
    artifacts: list[dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
) -> None:
    with connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (case_id, role, mode, content, artifacts_json, evidence_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                role,
                mode,
                content,
                json.dumps(artifacts or [], ensure_ascii=False),
                json.dumps(evidence or [], ensure_ascii=False),
                utc_now_iso(),
            ),
        )


def clear_chat_messages(database_path: Path, case_id: int) -> None:
    with connect(database_path) as conn:
        conn.execute("DELETE FROM chat_messages WHERE case_id = ?", (case_id,))


def record_sync_event(
    database_path: Path,
    case_id: int,
    *,
    provider: str,
    status: str,
    artifact_count: int,
    message: str,
) -> None:
    with connect(database_path) as conn:
        conn.execute(
            """
            INSERT INTO sync_events (case_id, provider, status, artifact_count, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (case_id, provider, status, artifact_count, message, utc_now_iso()),
        )


def latest_sync_event(database_path: Path, case_id: int) -> dict[str, Any] | None:
    with connect(database_path) as conn:
        row = conn.execute(
            """
            SELECT provider, status, artifact_count, message, created_at
            FROM sync_events
            WHERE case_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (case_id,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def export_case_payload(database_path: Path, case: CaseRecord) -> dict[str, Any]:
    return {
        "case": {
            "id": case.id,
            "slug": case.slug,
            "name": case.name,
            "description": case.description,
            "research_goal": case.research_goal,
            "is_legacy": case.is_legacy,
            "paths": {key: str(value) for key, value in case_paths(case).items()},
            "minio_prefix": case.minio_prefix,
            "created_at": case.created_at,
            "updated_at": case.updated_at,
        },
        "chat_messages": list_chat_messages(database_path, case.id),
    }


def minio_is_configured() -> bool:
    import os

    return bool(
        os.getenv("MINIO_ENDPOINT", "").strip()
        and os.getenv("MINIO_ACCESS_KEY", "").strip()
        and os.getenv("MINIO_SECRET_KEY", "").strip()
        and os.getenv("MINIO_BUCKET", "").strip()
    )


def _require_minio_client():
    try:
        from minio import Minio
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Install the `minio` package to enable artifact sync.") from exc
    return Minio


def sync_case_to_minio(database_path: Path, case: CaseRecord) -> dict[str, Any]:
    import os

    if not minio_is_configured():
        raise RuntimeError("MinIO is not configured. Set MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, and MINIO_BUCKET.")

    Minio = _require_minio_client()
    endpoint = os.getenv("MINIO_ENDPOINT", "").strip()
    bucket = os.getenv("MINIO_BUCKET", "").strip()
    secure = os.getenv("MINIO_SECURE", "false").strip().lower() in {"1", "true", "yes", "on"}
    region = os.getenv("MINIO_REGION", "").strip() or None
    client = Minio(
        endpoint,
        access_key=os.getenv("MINIO_ACCESS_KEY", "").strip(),
        secret_key=os.getenv("MINIO_SECRET_KEY", "").strip(),
        secure=secure,
        region=region,
    )
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    upload_count = 0
    uploaded_objects: list[str] = []
    paths = case_paths(case)

    def upload_bytes(object_name: str, payload: bytes, content_type: str = "application/json") -> None:
        nonlocal upload_count
        client.put_object(bucket, object_name, io.BytesIO(payload), len(payload), content_type=content_type)
        upload_count += 1
        uploaded_objects.append(object_name)

    def upload_file_tree(label: str, root_path: Path) -> None:
        nonlocal upload_count
        if not root_path.exists():
            return
        if root_path.is_file():
            object_name = f"{case.minio_prefix}/{label}/{root_path.name}"
            client.fput_object(bucket, object_name, str(root_path))
            upload_count += 1
            uploaded_objects.append(object_name)
            return
        for file_path in sorted(path for path in root_path.rglob("*") if path.is_file()):
            rel = file_path.relative_to(root_path).as_posix()
            object_name = f"{case.minio_prefix}/{label}/{rel}"
            client.fput_object(bucket, object_name, str(file_path))
            upload_count += 1
            uploaded_objects.append(object_name)

    upload_file_tree("bib_pdf", paths["bib_pdf"])
    upload_file_tree("multimodal_store", paths["multimodal_store"])
    upload_file_tree("scientific_graph_rag_store", paths["graph_store"])
    upload_file_tree("compiled_output", paths["compiled_output"])
    upload_file_tree("paper", paths["paper_tex"])

    payload = export_case_payload(database_path, case)
    upload_bytes(
        f"{case.minio_prefix}/metadata/case_export.json",
        json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
    )

    record_sync_event(
        database_path,
        case.id,
        provider="minio",
        status="success",
        artifact_count=upload_count,
        message=f"Synced {upload_count} objects to bucket {bucket}.",
    )
    return {"bucket": bucket, "prefix": case.minio_prefix, "uploaded": upload_count, "objects": uploaded_objects}
