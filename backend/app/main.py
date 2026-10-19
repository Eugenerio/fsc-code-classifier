"""FastAPI app: health, FSC code lookup, and the classify endpoints (plain JSON and NDJSON stream)."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, NamedTuple

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.config import get_settings
from app.decide import get_decider
from app.fsc import FscCatalog
from app.pipeline import PipelineError, run_pipeline, startup
from app.schemas import (
    ClassifyResponse,
    CompanyInput,
    ErrorEvent,
    FscCode,
    ResultEvent,
    StageEvent,
    UploadedFile,
)

log = logging.getLogger(__name__)

MAX_NAME, MAX_NOTES, MAX_URL = 200, 20_000, 2_048
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")  # keeps \t \n \r
_DOMAIN = re.compile(r"^(?=.{4,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        await startup()
    except Exception:
        log.exception("startup failed; the first request will retry")
    yield


app = FastAPI(title="FSC Code Classifier", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- input handling


class Submission(NamedTuple):
    company: CompanyInput
    files: list[UploadedFile]


def _clean(value: str | None, limit: int) -> str | None:
    cleaned = _CONTROL_CHARS.sub("", value or "").strip()[:limit].strip()
    return cleaned or None


def _email_domain(raw: str | None) -> str | None:
    """Accepts `b.com` or `a@b.com` and keeps the domain."""
    value = _clean(raw, MAX_URL)
    if value is None:
        return None
    domain = value.rpartition("@")[2].strip().lower().rstrip(".")
    if not _DOMAIN.match(domain):
        raise HTTPException(422, f"email_domain {value!r} is not a domain like example.com or name@example.com")
    return domain


async def read_submission(
    company_name: Annotated[str, Form()],
    website_url: Annotated[str | None, Form()] = None,
    email_domain: Annotated[str | None, Form()] = None,
    notes: Annotated[str | None, Form()] = None,
    files: Annotated[list[UploadFile | str] | None, File()] = None,  # browsers send "" when no file is chosen
) -> Submission:
    settings = get_settings()
    name = _clean(company_name, MAX_NAME)
    if name is None:
        raise HTTPException(422, "company_name must not be empty")
    company = CompanyInput(
        name=name,
        website_url=_clean(website_url, MAX_URL),
        email_domain=_email_domain(email_domain),
        notes=_clean(notes, MAX_NOTES),
    )

    uploads = [f for f in files or [] if isinstance(f, StarletteUploadFile) and f.filename]
    if len(uploads) > settings.max_files:
        raise HTTPException(422, f"At most {settings.max_files} files per request (got {len(uploads)})")
    limit = settings.max_upload_mb * 1024 * 1024
    loaded: list[UploadedFile] = []
    for upload in uploads:
        data = await upload.read(limit + 1)
        filename = Path(upload.filename or "upload").name[:255]
        if len(data) > limit:
            raise HTTPException(413, f"{filename} is larger than the {settings.max_upload_mb} MB limit")
        loaded.append(UploadedFile(filename=filename, content_type=upload.content_type, data=data))
    return Submission(company, loaded)


# --------------------------------------------------------------------------- endpoints


@app.get("/api/health")
async def health() -> dict[str, Any]:
    rt = await startup()
    return {
        "status": "ok",
        "gemini": rt.gemini is not None,
        "jev": bool(get_settings().typesafe_api_key),
        "decider": get_decider(rt.gemini).name,
        "fsc_count": len(rt.catalog.codes),
        "retrieval": [*rt.retriever.active_channels, *(["llm"] if rt.gemini else [])],
    }


def _term_score(term: str, c: FscCode) -> int:
    """0 = no match; code prefix > word start in the description > anywhere in description/group name."""
    if c.code.startswith(term):
        return 3
    description = c.description.lower()
    if re.search(rf"\b{re.escape(term)}", description):
        return 2
    return int(term in description or term in c.group_name.lower())


def _lookup(catalog: FscCatalog, q: str, limit: int) -> list[FscCode]:
    terms = q.lower().split()[:8]
    if not terms:
        return catalog.codes[:limit]
    scored: list[tuple[int, FscCode]] = []
    for c in catalog.codes:
        scores = [_term_score(t, c) for t in terms]
        if all(scores):
            scored.append((sum(scores), c))
    scored.sort(key=lambda s: (-s[0], s[1].code))
    return [c for _, c in scored[:limit]]


@app.get("/api/fsc")
async def fsc_lookup(q: str = "", limit: Annotated[int, Query(ge=1, le=200)] = 25) -> list[FscCode]:
    """Substring search over code, description and group name; powers the UI's code lookup."""
    return _lookup((await startup()).catalog, q, limit)


@app.post("/api/classify")
async def classify(sub: Annotated[Submission, Depends(read_submission)]) -> ClassifyResponse:
    try:
        return await run_pipeline(sub.company, sub.files)
    except PipelineError as exc:
        raise HTTPException(422, str(exc)) from exc


def _ndjson(event: BaseModel) -> str:
    return event.model_dump_json() + "\n"


async def _stream(sub: Submission) -> AsyncIterator[str]:
    """Runs the pipeline in a task and relays its events through a queue as they happen."""
    queue: asyncio.Queue[BaseModel | None] = asyncio.Queue()

    async def emit(event: StageEvent) -> None:
        await queue.put(event)

    async def work() -> None:
        try:
            await queue.put(ResultEvent(data=await run_pipeline(sub.company, sub.files, emit)))
        except PipelineError as exc:
            await queue.put(ErrorEvent(message=str(exc)))
        except Exception as exc:
            log.exception("pipeline crashed")
            await queue.put(ErrorEvent(message=f"Internal error: {type(exc).__name__}"))
        finally:
            await queue.put(None)

    task = asyncio.create_task(work())
    try:
        while (event := await queue.get()) is not None:
            yield _ndjson(event)
    finally:
        task.cancel()  # client went away mid-run; a finished task ignores this


@app.post("/api/classify/stream")
async def classify_stream(sub: Annotated[Submission, Depends(read_submission)]) -> StreamingResponse:
    return StreamingResponse(
        _stream(sub),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
