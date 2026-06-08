from __future__ import annotations

import json
import mimetypes
import os
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from scientific_graph_rag import DEFAULT_GRAPH_STORE, graph_store_paths, load_graphs, read_json, run_query


load_dotenv()

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "graph_rag_web_ui"
DEFAULT_HOST = os.getenv("GRAPH_RAG_WEB_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("GRAPH_RAG_WEB_PORT", "5000"))


def json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def load_summary(graph_store: Path) -> dict:
    manifests_dir = graph_store_paths(graph_store).manifests
    manifests = [read_json(path, {}) for path in sorted(manifests_dir.glob("*.json"))] if manifests_dir.exists() else []
    graphs = load_graphs(graph_store)
    return {
        "graph_store": str(graph_store.resolve()),
        "documents": len(manifests),
        "graphs_loaded": len(graphs),
        "items": manifests,
    }


class GraphRagRequestHandler(SimpleHTTPRequestHandler):
    server_version = "GraphRagWeb/0.1"

    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/summary":
            summary = load_summary(Path(os.getenv("SCIENTIFIC_GRAPH_RAG_STORE_DIR", str(DEFAULT_GRAPH_STORE))))
            json_response(self, summary)
            return

        if parsed.path == "/api/file":
            self.serve_workspace_file(parsed)
            return

        if parsed.path in {"/", "/index.html"}:
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/query":
            json_response(self, {"error": "Not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
        except Exception as exc:
            json_response(self, {"error": f"Invalid JSON payload: {exc}"}, status=400)
            return

        question = str(payload.get("question", "")).strip()
        if not question:
            json_response(self, {"error": "Question is required."}, status=400)
            return

        try:
            result = run_query(
                question=question,
                graph_store=Path(payload.get("graph_store") or os.getenv("SCIENTIFIC_GRAPH_RAG_STORE_DIR", str(DEFAULT_GRAPH_STORE))),
                backend=str(payload.get("backend") or "ollama"),
                model=str(payload.get("model") or os.getenv("DEFAULT_MODEL", "gpt-oss:20b")),
                ollama_base_url=str(payload.get("ollama_base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
                cloud_provider=str(payload.get("cloud_provider") or "openai"),
                api_key=str(payload.get("api_key") or ""),
                base_url=str(payload.get("base_url") or ""),
                max_queries=int(payload.get("max_queries") or 4),
                top_k_per_query=int(payload.get("top_k_per_query") or 10),
            )
            json_response(self, result)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=500)

    def serve_workspace_file(self, parsed) -> None:
        params = parse_qs(parsed.query)
        raw_path = str((params.get("path") or [""])[0]).strip()
        if not raw_path:
            json_response(self, {"error": "Missing file path."}, status=400)
            return

        file_path = Path(raw_path).expanduser().resolve()
        try:
            file_path.relative_to(ROOT)
        except ValueError:
            json_response(self, {"error": "Requested file is outside the project root."}, status=403)
            return

        if not file_path.exists() or not file_path.is_file():
            json_response(self, {"error": "File not found."}, status=404)
            return

        mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    handler = partial(GraphRagRequestHandler, directory=str(STATIC_DIR))
    with ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), handler) as server:
        print(f"Graph RAG web UI running at http://{DEFAULT_HOST}:{DEFAULT_PORT}")
        server.serve_forever()


if __name__ == "__main__":
    main()
