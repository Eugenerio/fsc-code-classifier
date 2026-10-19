"""Orchestration: ingest -> research -> profile -> retrieve -> decide -> rank, with per-stage events and warnings."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

from app.config import Settings, get_settings
from app.decide import Decider, DeciderError, RetrievalOnlyDecider, decider_chain
from app.fsc import load_catalog
from app.ingest.files import extract_file
from app.ingest.web import fetch_site, guess_site_from_email_domain, normalize_url
from app.llm.gemini import get_gemini
from app.profile import build_profile, source_infos, total_chars
from app.propose import propose_codes
from app.retrieval import Retriever, Routing, build_queries, merge_candidates
from app.schemas import (
    BackendInfo,
    Candidate,
    CandidateSource,
    ClassifyResponse,
    CompanyInput,
    CompanyProfile,
    Decision,
    FscMatch,
    SourceDoc,
    Stage,
    StageEvent,
    StageTiming,
    UploadedFile,
)
from app.scoring import rank_matches

if TYPE_CHECKING:
    from app.fsc import FscCatalog
    from app.llm.gemini import GeminiClient

log = logging.getLogger(__name__)

EmitFn = Callable[[StageEvent], Awaitable[None]]
StageStatus = Literal["done", "skipped", "error"]


class PipelineError(ValueError):
    """The request cannot be processed at all (nothing to classify)."""


# --------------------------------------------------------------------------- shared runtime


@dataclass(frozen=True)
class Runtime:
    """Process-wide and read-only after startup: the catalog, the built retrieval indexes and the Gemini client."""

    catalog: FscCatalog
    retriever: Retriever
    gemini: GeminiClient | None


_runtime: Runtime | None = None
_startup_lock = asyncio.Lock()


async def startup() -> Runtime:
    """Idempotent; called from the FastAPI lifespan, and lazily by run_pipeline for scripts and tests."""
    global _runtime
    if _runtime is None:
        async with _startup_lock:
            if _runtime is None:
                catalog = await asyncio.to_thread(load_catalog)
                gemini = get_gemini()
                retriever = Retriever(catalog, gemini)
                try:
                    await retriever.build()
                except Exception:
                    log.exception("retriever build failed; retrieval will be unavailable until restart")
                _runtime = Runtime(catalog, retriever, gemini)
    return _runtime


def _why(exc: BaseException) -> str:
    """One-line, length-capped reason for warnings (provider errors can be huge JSON blobs)."""
    text = " ".join(str(exc).split())[:200]
    return text if isinstance(exc, DeciderError) else f"{type(exc).__name__}: {text}"


# --------------------------------------------------------------------------- stage bookkeeping


@dataclass
class _Step:
    detail: str | None = None


class _Tracker:
    """Emits stage events, records timings, and turns a stage exception into an `error` stage plus a warning."""

    def __init__(self, emit: EmitFn | None) -> None:
        self._emit = emit
        self.timings: list[StageTiming] = []
        self.warnings: list[str] = []

    async def _finish(self, stage: Stage, status: StageStatus, elapsed_ms: int, detail: str | None) -> None:
        self.timings.append(StageTiming(stage=stage, status=status, elapsed_ms=elapsed_ms, detail=detail))
        if self._emit:
            await self._emit(StageEvent(stage=stage, status=status, detail=detail, elapsed_ms=elapsed_ms))

    @asynccontextmanager
    async def stage(self, stage: Stage) -> AsyncIterator[_Step]:
        """A failure degrades instead of aborting: the stage is marked `error`, a warning is added, the run goes on."""
        started = time.perf_counter()
        if self._emit:
            await self._emit(StageEvent(stage=stage, status="running"))
        step, status = _Step(), "done"
        try:
            yield step
        except Exception as exc:
            log.exception("stage %s failed", stage)
            status, step.detail = "error", _why(exc)
            self.warnings.append(f"The {stage} stage failed ({step.detail}); continuing with what is available.")
        await self._finish(stage, status, int((time.perf_counter() - started) * 1000), step.detail)

    async def skip(self, stage: Stage, reason: str) -> None:
        await self._finish(stage, "skipped", 0, reason)


# --------------------------------------------------------------------------- helpers


def _key_warnings(settings: Settings, gemini: GeminiClient | None) -> list[str]:
    """Say exactly what degraded and which env var fixes it."""
    out: list[str] = []
    if gemini is None:
        out.append(
            "GEMINI_API_KEY is not set: using the heuristic profile extractor and BM25-only recall; scanned PDFs and "
            "images cannot be OCR'd and web research is off. Set GEMINI_API_KEY in backend/.env for full quality."
        )
    if not settings.typesafe_api_key and settings.decider in ("auto", "jev"):
        instead = "Gemini judges the candidates" if gemini is not None else "scores are retrieval-only (unverified)"
        out.append(
            f"TYPESAFE_API_KEY is not set: the Jev decision layer is off and {instead}. "
            "Set TYPESAFE_API_KEY to enable Jev."
        )
    if settings.decider == "gemini" and gemini is None:
        out.append("DECIDER=gemini but GEMINI_API_KEY is not set: scores are retrieval-only.")
    return out


def _resolve_site(company: CompanyInput, warnings: list[str]) -> str | None:
    site = normalize_url(company.website_url)
    if company.website_url and company.website_url.strip() and site is None:
        warnings.append(f"Website URL {company.website_url!r} is not valid and was ignored.")
    return site or guess_site_from_email_domain(company.email_domain)


def _ingest_detail(docs: list[SourceDoc]) -> str:
    ok = [d for d in docs if d.ok]
    pages, uploads, notes = (sum(d.kind == kind for d in ok) for kind in ("website", "file", "notes"))
    parts = [
        text
        for n, text in (
            (pages, f"{pages} page{'s' * (pages != 1)}"),
            (uploads, f"{uploads} file{'s' * (uploads != 1)}"),
            (notes, "notes"),
        )
        if n
    ]
    return f"{', '.join(parts) or 'no sources'}, {total_chars(docs):,} chars"


async def _retrieve(
    profile: CompanyProfile, rt: Runtime, warnings: list[str]
) -> tuple[list[Candidate], list[CandidateSource], Routing]:
    # search_routed (not search): the shared retriever's `last_routing` would race between concurrent requests
    jobs = {"recall": rt.retriever.search_routed(build_queries(profile))}
    if rt.gemini is not None:
        jobs["llm"] = propose_codes(profile, rt.catalog, rt.gemini)
    results = await asyncio.gather(*jobs.values(), return_exceptions=True)

    lists: list[list[Candidate]] = []
    channels: list[CandidateSource] = []
    routing = Routing()
    for (name, _), result in zip(jobs.items(), results, strict=True):
        if isinstance(result, BaseException):
            log.warning("%s channel failed: %r", name, result)
            label = "BM25/embedding" if name == "recall" else "LLM"
            warnings.append(f"{label} recall failed ({_why(result)}); relying on the other channels.")
            continue
        if name == "recall":
            result, routing = result
        lists.append(result)
        channels += rt.retriever.active_channels if name == "recall" else ["llm"]
        if name == "recall" and rt.gemini is not None and "embedding" not in rt.retriever.active_channels:
            warnings.append("Embedding recall is unavailable (the embedding call failed): BM25 + LLM recall only.")
    return merge_candidates(*lists), channels, routing


async def _decide(
    profile: CompanyProfile, candidates: list[Candidate], rt: Runtime, warnings: list[str]
) -> tuple[Decider, list[Decision]]:
    chain = decider_chain(rt.gemini)  # always ends with retrieval-only, which cannot fail
    for decider, fallback in pairwise(chain):
        try:
            return decider, await decider.decide(profile, candidates, rt.catalog)
        except Exception as exc:
            if not isinstance(exc, DeciderError):
                log.exception("%s decider crashed", decider.name)
            warnings.append(f"The {decider.name} decider failed ({_why(exc)}); fell back to {fallback.name}.")
    return chain[-1], await chain[-1].decide(profile, candidates, rt.catalog)


# --------------------------------------------------------------------------- entrypoint


async def run_pipeline(
    company: CompanyInput, files: list[UploadedFile], emit: EmitFn | None = None
) -> ClassifyResponse:
    """Safe to call concurrently: all per-run state is local and the shared Runtime is read-only."""
    started = time.perf_counter()
    settings = get_settings()
    rt = await startup()
    tr = _Tracker(emit)
    tr.warnings += _key_warnings(settings, rt.gemini)
    site = _resolve_site(company, tr.warnings)

    docs: list[SourceDoc] = []
    if company.notes and company.notes.strip():
        docs.append(SourceDoc(kind="notes", label="notes", text=company.notes.strip()))
    async with tr.stage("ingest") as step:

        async def crawl() -> list[SourceDoc]:
            return await fetch_site(site, max_pages=settings.max_site_pages) if site else []

        site_docs, *file_docs = await asyncio.gather(crawl(), *(extract_file(f, rt.gemini) for f in files))
        docs += [*site_docs, *file_docs]
        step.detail = _ingest_detail(docs)
    if failed := [d for d in docs if not d.ok]:
        shown = "; ".join(f"{d.label} ({d.error})" for d in failed[:3])
        tr.warnings.append(f"{len(failed)} of {len(docs)} sources failed: {shown}")
    if not company.name.strip() and not total_chars(docs):
        raise PipelineError("Nothing to classify: no company name and no usable evidence.")

    if rt.gemini is None:
        await tr.skip("research", "needs GEMINI_API_KEY")
    elif not settings.enable_web_research:
        await tr.skip("research", "disabled (ENABLE_WEB_RESEARCH=false)")
    elif (chars := total_chars(docs)) >= settings.sparse_evidence_chars:
        await tr.skip("research", f"evidence is sufficient ({chars:,} chars)")
    else:
        async with tr.stage("research") as step:
            text = await rt.gemini.research_company(company.name, site, company.email_domain)
            if text.strip():
                docs.append(SourceDoc(kind="web_research", label="google_search", text=text))
            step.detail = f"google_search: {len(text):,} chars" if text.strip() else "no public information found"
    if (chars := total_chars(docs)) < settings.sparse_evidence_chars:
        tr.warnings.append(
            f"Sparse evidence ({chars:,} chars): results may be unreliable. Add a website, notes or a document."
        )

    profile = CompanyProfile(name=company.name)
    async with tr.stage("profile") as step:
        profile = await build_profile(company, docs, rt.gemini)
        step.detail = f"{len(profile.products)} product lines via {settings.gemini_model if rt.gemini else 'heuristic'}"

    candidates: list[Candidate] = []
    channels: list[CandidateSource] = []
    routing = Routing()
    async with tr.stage("retrieve") as step:
        candidates, channels, routing = await _retrieve(profile, rt, tr.warnings)
        step.detail = f"{len(candidates)} candidates ({'+'.join(channels) or 'no channel'})"
    if not candidates:
        tr.warnings.append("No FSC candidates were retrieved; add more detail about what the company sells.")

    decider: Decider = RetrievalOnlyDecider()
    decisions: list[Decision] = []
    async with tr.stage("decide") as step:
        decider, decisions = await _decide(profile, candidates, rt, tr.warnings)
        unit = "questions" if decider.name == "jev" else "candidates"
        step.detail = f"{decider.model or 'retrieval-only'}: {len(candidates)} {unit}"

    matches: list[FscMatch] = []
    async with tr.stage("rank") as step:
        matches = rank_matches(
            candidates,
            decisions,
            rt.catalog,
            group_prior=decider.group_prior,
            domain_prior=decider.domain_prior,
            routing=routing,
            profile=profile,
            decider_name=decider.name,
        )
        step.detail = f"{len(matches)} codes" + (f", top {matches[0].code} ({matches[0].score:.2f})" if matches else "")
    if matches and all(m.confidence == "low" for m in matches):
        tr.warnings.append("No code reached medium confidence; treat the list as leads to review, not answers.")

    return ClassifyResponse(
        company=company,
        profile=profile,
        fsc_codes=matches,
        sources=source_infos(docs),
        warnings=tr.warnings,
        stages=tr.timings,
        backend=BackendInfo(
            extractor=settings.gemini_model if rt.gemini else "heuristic",
            decider=decider.name,
            decider_model=decider.model,
            retrieval=channels,
        ),
        total_ms=int((time.perf_counter() - started) * 1000),
    )
