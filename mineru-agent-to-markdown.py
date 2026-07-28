#!/usr/bin/env python3
"""Convert a local document to Markdown with MinerU's HTTP API.

The default mode uses MinerU's token-free Agent signed upload flow:
1. POST /api/v1/agent/parse/file to get a task_id and upload URL.
2. PUT the local file to the signed upload URL.
3. Poll /api/v1/agent/parse/{task_id}.
4. Download markdown_url and save it locally.

Use --mode precise for MinerU's token-based v4 API, which supports larger
documents and returns a zip containing full.md.
"""

from __future__ import annotations

import argparse
import io
import os
import posixpath
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

import requests


AGENT_BASE_URL = "https://mineru.net/api/v1/agent"
PRECISE_BASE_URL = "https://mineru.net/api/v4"
DEFAULT_OUTPUT_DIR = Path("derived/pdf-markdown")
DONE = "done"
FAILED = "failed"


class MineruError(RuntimeError):
    """Raised when MinerU returns an unsuccessful response."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a local file to Markdown through MinerU's Agent API."
    )
    parser.add_argument("file", type=Path, help="Local PDF/document/image file to parse.")
    parser.add_argument(
        "--mode",
        choices=("agent", "precise"),
        default="agent",
        help="API mode. agent is token-free but limited; precise requires MINERU_API_TOKEN.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output Markdown path. Defaults to derived/pdf-markdown/<input-stem>.md.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory when --output is omitted. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        help=(
            "Directory for files extracted from precise-mode result zip. "
            "Defaults to <output-stem> next to the Markdown file."
        ),
    )
    parser.add_argument(
        "--language",
        default="ch",
        help="OCR language code. MinerU default is ch; use en for English papers.",
    )
    parser.add_argument(
        "--model-version",
        default="vlm",
        choices=("pipeline", "vlm", "MinerU-HTML"),
        help="Model for --mode precise. Default: vlm.",
    )
    parser.add_argument(
        "--token",
        help="MinerU API token for --mode precise. Defaults to MINERU_API_TOKEN.",
    )
    parser.add_argument(
        "--page-range",
        help="PDF page range accepted by Agent API, e.g. 1-10 or 5.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR. MinerU defaults to false for the Agent API.",
    )
    parser.add_argument(
        "--disable-table",
        action="store_true",
        help="Disable table recognition.",
    )
    parser.add_argument(
        "--disable-formula",
        action="store_true",
        help="Disable formula recognition.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Polling timeout in seconds. Default: 600.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Polling interval in seconds. Default: 3.",
    )
    parser.add_argument(
        "--use-proxy",
        action="store_true",
        help="Honor HTTP(S)_PROXY/ALL_PROXY environment variables (disabled by default).",
    )
    return parser.parse_args()


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        response = session.request(method, url, timeout=60, **kwargs)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise MineruError(f"{method} {url} failed: {exc}") from exc
    except ValueError as exc:
        raise MineruError(f"{method} {url} returned non-JSON response") from exc

    if payload.get("code") != 0:
        msg = payload.get("msg") or "unknown MinerU API error"
        trace_id = payload.get("trace_id")
        suffix = f" trace_id={trace_id}" if trace_id else ""
        raise MineruError(f"MinerU API error: {msg}{suffix}")

    return payload


def auth_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def create_agent_task(
    session: requests.Session,
    file_path: Path,
    language: str,
    page_range: str | None,
    is_ocr: bool,
    enable_table: bool,
    enable_formula: bool,
) -> tuple[str, str]:
    payload: dict[str, Any] = {
        "file_name": file_path.name,
        "language": language,
        "enable_table": enable_table,
        "is_ocr": is_ocr,
        "enable_formula": enable_formula,
    }
    if page_range:
        payload["page_range"] = page_range

    result = request_json(session, "POST", f"{AGENT_BASE_URL}/parse/file", json=payload)
    data = result["data"]
    return data["task_id"], data["file_url"]


def create_precise_batch(
    session: requests.Session,
    file_path: Path,
    token: str,
    language: str,
    page_range: str | None,
    is_ocr: bool,
    enable_table: bool,
    enable_formula: bool,
    model_version: str,
) -> tuple[str, str]:
    file_entry = {
        "name": file_path.name,
        "data_id": file_path.stem,
    }
    if page_range:
        file_entry["page_ranges"] = page_range

    payload = {
        "files": [file_entry],
        "model_version": model_version,
        "language": language,
        "is_ocr": is_ocr,
        "enable_table": enable_table,
        "enable_formula": enable_formula,
    }

    result = request_json(
        session,
        "POST",
        f"{PRECISE_BASE_URL}/file-urls/batch",
        headers=auth_headers(token),
        json=payload,
    )
    data = result["data"]
    file_urls = data.get("file_urls") or []
    if not file_urls:
        raise MineruError("v4 upload URL response did not include file_urls")
    return data["batch_id"], file_urls[0]


def upload_file(session: requests.Session, file_path: Path, file_url: str) -> None:
    try:
        with file_path.open("rb") as handle:
            response = session.put(file_url, data=handle, timeout=300)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MineruError(f"upload failed: {exc}") from exc


def poll_agent_result(
    session: requests.Session,
    task_id: str,
    timeout: int,
    interval: float,
) -> str:
    deadline = time.monotonic() + timeout
    last_state = ""

    while time.monotonic() < deadline:
        result = request_json(session, "GET", f"{AGENT_BASE_URL}/parse/{task_id}")
        data = result["data"]
        state = data.get("state", "")

        if state != last_state:
            print(f"state={state}", file=sys.stderr)
            last_state = state

        if state == DONE:
            markdown_url = data.get("markdown_url")
            if not markdown_url:
                raise MineruError("task finished but response did not include markdown_url")
            return markdown_url

        if state == FAILED:
            raise MineruError(data.get("err_msg") or "MinerU parse task failed")

        time.sleep(interval)

    raise MineruError(f"timed out after {timeout}s; task_id={task_id}")


def poll_precise_result(
    session: requests.Session,
    batch_id: str,
    token: str,
    timeout: int,
    interval: float,
) -> str:
    deadline = time.monotonic() + timeout
    last_state = ""

    while time.monotonic() < deadline:
        result = request_json(
            session,
            "GET",
            f"{PRECISE_BASE_URL}/extract-results/batch/{batch_id}",
            headers=auth_headers(token),
        )
        extract_results = result["data"].get("extract_result") or []
        if not extract_results:
            raise MineruError("v4 batch result did not include extract_result")
        data = extract_results[0]
        state = data.get("state", "")

        if state != last_state:
            progress = data.get("extract_progress") or {}
            pages = ""
            if progress.get("total_pages"):
                pages = f" pages={progress.get('extracted_pages', 0)}/{progress['total_pages']}"
            print(f"state={state}{pages}", file=sys.stderr)
            last_state = state

        if state == DONE:
            full_zip_url = data.get("full_zip_url")
            if not full_zip_url:
                raise MineruError("task finished but response did not include full_zip_url")
            return full_zip_url

        if state == FAILED:
            raise MineruError(data.get("err_msg") or "MinerU parse task failed")

        time.sleep(interval)

    raise MineruError(f"timed out after {timeout}s; batch_id={batch_id}")


def download_markdown(session: requests.Session, markdown_url: str) -> str:
    try:
        response = session.get(markdown_url, timeout=120)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        raise MineruError(f"Markdown download failed: {exc}") from exc


def is_remote_or_special_link(link: str) -> bool:
    parsed = urlparse(link)
    return bool(parsed.scheme or parsed.netloc or link.startswith("#"))


def safe_member_path(member_name: str) -> Path:
    path = Path(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise MineruError(f"unsafe path in result zip: {member_name}")
    return path


def rewrite_markdown_asset_links(
    markdown: str,
    link_map: dict[str, str],
) -> str:
    def rewrite_link(link: str) -> str:
        if is_remote_or_special_link(link):
            return link
        normalized = posixpath.normpath(link).lstrip("/")
        return link_map.get(normalized, link)

    def rewrite_markdown_image(match: re.Match[str]) -> str:
        alt_text = match.group("alt")
        link = match.group("link")
        return f"![{alt_text}]({rewrite_link(link)})"

    def rewrite_html_src(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        quote = match.group("quote")
        link = match.group("link")
        return f"{prefix}{quote}{rewrite_link(link)}{quote}"

    markdown = re.sub(
        r"!\[(?P<alt>[^\]]*)\]\((?P<link>[^)\s]+)\)",
        rewrite_markdown_image,
        markdown,
    )
    return re.sub(
        r"(?P<prefix><img\b[^>]*\bsrc=)(?P<quote>[\"'])(?P<link>[^\"']+)(?P=quote)",
        rewrite_html_src,
        markdown,
        flags=re.IGNORECASE,
    )


def download_full_md_from_zip(
    session: requests.Session,
    full_zip_url: str,
    assets_dir: Path,
    markdown_parent: Path,
) -> tuple[str, int]:
    try:
        response = session.get(full_zip_url, timeout=300)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MineruError(f"result zip download failed: {exc}") from exc

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            full_md = next((name for name in names if name.endswith("/full.md")), None)
            if full_md is None and "full.md" in names:
                full_md = "full.md"
            if full_md is None:
                markdown_files = [name for name in names if name.endswith(".md")]
                if not markdown_files:
                    raise MineruError("result zip did not contain a Markdown file")
                full_md = markdown_files[0]
            markdown = archive.read(full_md).decode("utf-8")
            full_md_dir = posixpath.dirname(full_md)
            link_map: dict[str, str] = {}
            extracted_count = 0

            for name in names:
                if name == full_md:
                    continue

                dest_rel_name = name
                if full_md_dir and name.startswith(f"{full_md_dir}/"):
                    dest_rel_name = name[len(full_md_dir) + 1 :]

                dest_rel_path = safe_member_path(dest_rel_name)
                dest_path = assets_dir / dest_rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(archive.read(name))
                extracted_count += 1

                markdown_rel = os.path.relpath(dest_path, markdown_parent).replace(os.sep, "/")
                link_map[posixpath.normpath(dest_rel_name).lstrip("/")] = markdown_rel
                link_map[posixpath.normpath(name).lstrip("/")] = markdown_rel

            return rewrite_markdown_asset_links(markdown, link_map), extracted_count
    except zipfile.BadZipFile as exc:
        raise MineruError("result download was not a valid zip file") from exc


def default_output_path(file_path: Path, output_dir: Path) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", file_path.stem).strip("-")
    return output_dir / f"{safe_stem or 'mineru-output'}.md"


def default_assets_dir(output_path: Path) -> Path:
    return output_path.with_suffix("")


def main() -> int:
    args = parse_args()
    file_path = args.file.expanduser().resolve()
    if not file_path.is_file():
        print(f"error: file not found: {file_path}", file=sys.stderr)
        return 2

    output_path = (args.output or default_output_path(file_path, args.output_dir)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = (args.assets_dir or default_assets_dir(output_path)).resolve()

    session = requests.Session()
    if not args.use_proxy:
        session.trust_env = False

    try:
        if args.mode == "agent":
            task_id, file_url = create_agent_task(
                session=session,
                file_path=file_path,
                language=args.language,
                page_range=args.page_range,
                is_ocr=args.ocr,
                enable_table=not args.disable_table,
                enable_formula=not args.disable_formula,
            )
            print(f"task_id={task_id}", file=sys.stderr)
            upload_file(session, file_path, file_url)
            print("upload=done", file=sys.stderr)
            markdown_url = poll_agent_result(session, task_id, args.timeout, args.interval)
            markdown = download_markdown(session, markdown_url)
        else:
            token = args.token or os.environ.get("MINERU_API_TOKEN")
            if not token:
                raise MineruError("--mode precise requires --token or MINERU_API_TOKEN")
            batch_id, file_url = create_precise_batch(
                session=session,
                file_path=file_path,
                token=token,
                language=args.language,
                page_range=args.page_range,
                is_ocr=args.ocr,
                enable_table=not args.disable_table,
                enable_formula=not args.disable_formula,
                model_version=args.model_version,
            )
            print(f"batch_id={batch_id}", file=sys.stderr)
            upload_file(session, file_path, file_url)
            print("upload=done", file=sys.stderr)
            full_zip_url = poll_precise_result(session, batch_id, token, args.timeout, args.interval)
            markdown, extracted_count = download_full_md_from_zip(
                session,
                full_zip_url,
                assets_dir,
                output_path.parent,
            )
            print(f"assets={assets_dir} files={extracted_count}", file=sys.stderr)
    except MineruError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(markdown, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
