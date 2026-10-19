"""Uploaded file -> text: text-layer PDFs via pypdf, scanned PDFs and images via Gemini OCR, text-ish files decoded."""

from __future__ import annotations

import asyncio
import io
import mimetypes
from pathlib import PurePath
from typing import TYPE_CHECKING

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError

from app.ingest.web import parse_html
from app.schemas import SourceDoc, UploadedFile

if TYPE_CHECKING:
    from app.llm.gemini import GeminiClient

SCANNED_CHARS_PER_PAGE = 200  # below this a PDF is treated as scanned and sent to OCR
MAX_PDF_PAGES = 150
IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
TEXT_MIMES = {"application/json", "application/xml", "application/xhtml+xml", "application/csv"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".html", ".htm", ".xml", ".log"}
MAGIC = ((b"%PDF-", "application/pdf"), (b"\x89PNG\r\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF8", "image/gif"))


def _mime(f: UploadedFile) -> str:
    """Trust file contents over the declared type (browsers send octet-stream or wrong types for many uploads)."""
    for prefix, mime in MAGIC:
        if f.data.startswith(prefix):
            return mime
    if f.data[:4] == b"RIFF" and f.data[8:12] == b"WEBP":
        return "image/webp"
    declared = (f.content_type or "").split(";")[0].strip().lower()
    if declared and declared != "application/octet-stream":
        return declared
    return mimetypes.guess_type(f.filename)[0] or ""


def _pdf_text(data: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        raise PdfReadError("PDF is password-protected")
    pages = []
    for page in reader.pages[:MAX_PDF_PAGES]:
        try:
            pages.append((page.extract_text() or "").strip())
        except Exception:  # one broken page must not lose the rest of the document
            pages.append("")
    return "\n\n".join(p for p in pages if p), len(reader.pages)


async def _extract_pdf(f: UploadedFile, gemini: GeminiClient | None) -> SourceDoc:
    try:
        text, pages = await asyncio.to_thread(_pdf_text, f.data)
    except PyPdfError as e:
        return _fail(f, f"could not read PDF: {e}")
    scanned = len(text) < SCANNED_CHARS_PER_PAGE * max(1, min(pages, MAX_PDF_PAGES))
    if scanned and gemini is not None:
        try:
            ocr_text = await gemini.ocr(f.data, "application/pdf", f.filename)
        except Exception as e:  # fall back to whatever text layer exists
            if not text:
                return _fail(f, f"OCR failed: {type(e).__name__}")
        else:
            if ocr_text.strip():
                return SourceDoc(kind="file", label=f.filename, text=ocr_text, meta={"ocr": True, "pages": pages})
    if not text:
        reason = "scanned PDF with no text layer" + ("" if gemini else ": needs GEMINI_API_KEY for OCR")
        return _fail(f, reason)
    return SourceDoc(kind="file", label=f.filename, text=text, meta={"ocr": False, "pages": pages})


async def _extract_image(f: UploadedFile, mime: str, gemini: GeminiClient | None) -> SourceDoc:
    if gemini is None:
        return _fail(f, "needs GEMINI_API_KEY for OCR")
    text = (await gemini.ocr(f.data, mime, f.filename)).strip()
    if not text:
        return _fail(f, "OCR found no text")
    return SourceDoc(kind="file", label=f.filename, text=text, meta={"ocr": True, "pages": 1})


def _extract_text(f: UploadedFile, ext: str) -> SourceDoc:
    text = f.data.decode("utf-8-sig", errors="replace").replace("\x00", "")
    if ext in {".html", ".htm"} or text.lstrip()[:15].lower().startswith(("<!doctype html", "<html")):
        text = parse_html(f.data)[0]
    text = text.strip()
    if not text:
        return _fail(f, "file is empty")
    return SourceDoc(kind="file", label=f.filename, text=text, meta={"ocr": False, "pages": 1})


def _fail(f: UploadedFile, error: str) -> SourceDoc:
    return SourceDoc(kind="file", label=f.filename, ok=False, error=error)


async def extract_file(f: UploadedFile, gemini: GeminiClient | None) -> SourceDoc:
    try:
        mime, ext = _mime(f), PurePath(f.filename).suffix.lower()
        if mime == "application/pdf":
            return await _extract_pdf(f, gemini)
        if mime in IMAGE_MIMES:
            return await _extract_image(f, mime, gemini)
        if mime.startswith("text/") or mime in TEXT_MIMES or ext in TEXT_EXTS:
            return _extract_text(f, ext)
        return _fail(f, f"unsupported file type: {mime or ext or 'unknown'}")
    except Exception as e:  # contract: never raise; one bad upload must not abort the whole request
        return _fail(f, f"{type(e).__name__}: {e}"[:200])
