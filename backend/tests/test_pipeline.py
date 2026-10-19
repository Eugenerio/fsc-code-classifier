from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app import decide, pipeline, scoring
from app.config import Settings
from app.decide import DeciderError
from app.fsc import FscCatalog
from app.pipeline import PipelineError, Runtime, run_pipeline
from app.retrieval import Routing
from app.schemas import (
    Candidate,
    ClassifyResponse,
    CompanyInput,
    CompanyProfile,
    Decision,
    FscCode,
    ProductLine,
    SourceDoc,
    StageEvent,
    UploadedFile,
)

CATALOG = FscCatalog(
    [
        FscCode(code=c, description=d, sos_ric="S9C/AX", group=c[:2], group_name=n)
        for c, d, n in [
            ("5305", "Screws", "Hardware and Abrasives"),
            ("5306", "Bolts", "Hardware and Abrasives"),
            ("4730", "Hose, Pipe, Tube Fittings", "Pipe, Tubing, Hose, and Fittings"),
        ]
    ]
)
MODEL = Settings(_env_file=None).gemini_model  # the default is a deployment choice, not a contract
STAGES = ["ingest", "research", "profile", "retrieve", "decide", "rank"]
BIG_PAGE = "We manufacture stainless steel hex bolts and socket screws. " * 40  # > sparse_evidence_chars


class FakeRetriever:
    active_channels = ["bm25", "embedding"]

    def __init__(self, error: Exception | None = None) -> None:
        self.error, self.queries, self.routing = error, None, Routing()

    async def search_routed(self, queries, **_: Any) -> tuple[list[Candidate], Routing]:
        self.queries = queries
        if self.error:
            raise self.error
        return [
            Candidate(code="5306", retrieval_score=1.0, sources=["bm25", "embedding"], matched_products=["hex bolts"]),
            Candidate(code="5305", retrieval_score=0.8, sources=["bm25"], matched_products=["hex bolts"]),
            Candidate(code="4730", retrieval_score=0.3, sources=["embedding"]),
        ], self.routing


class FakeGemini:
    def __init__(self, research: str = "") -> None:
        self.research, self.research_calls = research, []

    async def research_company(self, name: str, website: str | None, domain: str | None) -> str:
        self.research_calls.append((name, website, domain))
        return self.research


class FakeDecider:
    def __init__(self, name="jev", model="jev-1.13.0", error: Exception | None = None, delay: bool = False) -> None:
        self.name, self.model, self.error, self.delay = name, model, error, delay
        self.group_prior: dict[str, float] = {}
        self.domain_prior: dict[str, float] = {}
        self.seen_profile: CompanyProfile | None = None

    async def decide(self, profile, candidates, catalog) -> list[Decision]:
        if self.delay:
            await asyncio.sleep(0)  # let concurrent runs interleave
        self.seen_profile = profile
        if self.error:
            raise self.error
        probs = {"5306": 0.95, "5305": 0.9, "4730": 0.05}
        return [Decision(code=c.code, probability=probs[c.code]) for c in candidates]


def profile_for(company: CompanyInput) -> CompanyProfile:
    return CompanyProfile(
        name=company.name,
        summary="Fastener maker.",
        products=[ProductLine(name="hex bolts", description="Stainless hex bolts", role="manufacturer")],
    )


@pytest.fixture
def world(monkeypatch):
    """Everything the pipeline touches, faked: no network, no keys; tweak attributes before calling run."""

    class World:
        gemini: Any = FakeGemini()
        retriever = FakeRetriever()
        chain_factory = staticmethod(lambda gemini: [FakeDecider(), decide.RetrievalOnlyDecider()])
        site_docs: list[SourceDoc] = [
            SourceDoc(kind="website", label="https://acme.test/", text=BIG_PAGE),
            SourceDoc(kind="website", label="https://acme.test/products", text="Products: hex bolts."),
        ]
        file_doc = SourceDoc(kind="file", label="capability.pdf", text="Capability statement.", meta={"pages": 2})
        profile_error: Exception | None = None
        profile_docs: list[SourceDoc] = []
        settings: dict[str, Any] = {"typesafe_api_key": "tk", "decider": "auto"}

        def install(self) -> World:
            monkeypatch.setattr(pipeline, "_runtime", Runtime(CATALOG, self.retriever, self.gemini))
            monkeypatch.setattr(pipeline, "decider_chain", lambda gemini: self.chain_factory(gemini))
            settings = Settings(_env_file=None, **{"gemini_api_key": None, "typesafe_api_key": None, **self.settings})
            for module in (pipeline, decide, scoring):
                monkeypatch.setattr(module, "get_settings", lambda: settings)
            return self

    w = World()

    async def fake_fetch_site(url: str, *, max_pages: int | None = None, **_: Any) -> list[SourceDoc]:
        return w.site_docs

    async def fake_extract_file(f: UploadedFile, gemini: Any) -> SourceDoc:
        return w.file_doc

    async def fake_build_profile(company: CompanyInput, docs: list[SourceDoc], gemini: Any) -> CompanyProfile:
        w.profile_docs = docs
        if w.profile_error:
            raise w.profile_error
        return profile_for(company)

    async def fake_propose(profile, catalog, gemini, max_codes: int = 25) -> list[Candidate]:
        return [Candidate(code="5305", retrieval_score=1.0, sources=["llm"], llm_reason="Sells screws")]

    monkeypatch.setattr(pipeline, "fetch_site", fake_fetch_site)
    monkeypatch.setattr(pipeline, "extract_file", fake_extract_file)
    monkeypatch.setattr(pipeline, "build_profile", fake_build_profile)
    monkeypatch.setattr(pipeline, "propose_codes", fake_propose)
    return w


COMPANY = CompanyInput(name="Acme Fasteners", website_url="https://acme.test", notes="Fasteners for oil & gas.")
PDF = UploadedFile(filename="capability.pdf", content_type="application/pdf", data=b"%PDF")


async def run(company: CompanyInput = COMPANY, files=(PDF,)) -> tuple[ClassifyResponse, list[StageEvent]]:
    events: list[StageEvent] = []

    async def emit(event: StageEvent) -> None:
        events.append(event)

    return await run_pipeline(company, list(files), emit), events


async def test_full_run_emits_ordered_events_and_builds_the_response(world):
    world.install()
    result, events = await run()

    assert [(e.stage, e.status) for e in events] == [
        ("ingest", "running"), ("ingest", "done"),
        ("research", "skipped"),
        ("profile", "running"), ("profile", "done"),
        ("retrieve", "running"), ("retrieve", "done"),
        ("decide", "running"), ("decide", "done"),
        ("rank", "running"), ("rank", "done"),
    ]  # fmt: skip
    assert all(e.elapsed_ms is None for e in events if e.status == "running")
    assert all(e.elapsed_ms is not None for e in events if e.status != "running")
    details = {e.stage: e.detail for e in events if e.status != "running"}
    chars = len(BIG_PAGE) + len("Products: hex bolts.") + len("Capability statement.") + len(COMPANY.notes or "")
    assert details["ingest"] == f"2 pages, 1 file, notes, {chars:,} chars"
    assert details["research"].startswith("evidence is sufficient")
    assert details["profile"] == f"1 product lines via {MODEL}"
    assert details["retrieve"] == "3 candidates (bm25+embedding+llm)"
    assert details["decide"] == "jev-1.13.0: 3 questions"
    assert details["rank"] == "3 codes, top 5306 (0.96)"

    assert [(s.stage, s.status) for s in result.stages] == [
        ("ingest", "done"), ("research", "skipped"), ("profile", "done"),
        ("retrieve", "done"), ("decide", "done"), ("rank", "done"),
    ]  # fmt: skip
    assert [m.code for m in result.fsc_codes] == ["5306", "5305", "4730"]
    top, second, padded = result.fsc_codes
    assert (top.score, top.confidence, top.decider_probability) == (0.9625, "high", 0.95)
    assert top.matched_products == ["hex bolts"] and top.rationale == "matched: hex bolts"
    assert second.rationale == "Sells screws"  # LLM-channel reason survives the merge
    assert padded.confidence == "low" and padded.score < 0.45  # padded in to reach min_results
    assert result.backend.model_dump() == {
        "extractor": MODEL,
        "decider": "jev",
        "decider_model": "jev-1.13.0",
        "retrieval": ["bm25", "embedding", "llm"],
    }
    assert [(s.kind, s.label, s.ok) for s in result.sources] == [
        ("notes", "notes", True),
        ("website", "https://acme.test/", True),
        ("website", "https://acme.test/products", True),
        ("file", "capability.pdf", True),
    ]
    assert result.warnings == []
    assert result.company == COMPANY and result.profile.name == "Acme Fasteners" and result.total_ms >= 0
    ClassifyResponse.model_validate(result.model_dump())  # serialisable round trip


async def test_sparse_evidence_triggers_web_research(world):
    paragraph = "Acme makes stainless fasteners for oilfield customers."
    world.gemini = FakeGemini(research=paragraph)
    world.site_docs = []
    world.install()
    result, events = await run(CompanyInput(name="Acme", email_domain="acme.test"), files=())

    assert world.gemini.research_calls == [("Acme", "https://acme.test", "acme.test")]  # site guessed from the domain
    research = next(e for e in events if e.stage == "research" and e.status == "done")
    assert research.detail == f"google_search: {len(paragraph)} chars"
    assert [d.kind for d in world.profile_docs] == ["web_research"]
    assert [s.kind for s in result.sources] == ["web_research"]
    assert any("Sparse evidence" in w for w in result.warnings)  # still thin: only the research paragraph


async def test_research_that_finds_nothing_is_reported_not_fatal(world):
    world.site_docs = []
    world.install()
    result, events = await run(CompanyInput(name="Acme"), files=())
    assert (
        next(e for e in events if e.stage == "research" and e.status == "done").detail == "no public information found"
    )
    assert result.fsc_codes


async def test_keyless_run_degrades_everywhere_and_says_how(world):
    world.gemini, world.settings, world.chain_factory = None, {}, decide.decider_chain
    world.retriever.active_channels = ["bm25"]
    world.install()
    result, events = await run()

    assert result.backend.model_dump() == {
        "extractor": "heuristic", "decider": "retrieval", "decider_model": None, "retrieval": ["bm25"],
    }  # fmt: skip
    assert any(w.startswith("GEMINI_API_KEY is not set") and "heuristic" in w for w in result.warnings)
    assert any(w.startswith("TYPESAFE_API_KEY is not set") and "retrieval-only" in w for w in result.warnings)
    research = next(e for e in events if e.stage == "research")
    assert (research.status, research.detail) == ("skipped", "needs GEMINI_API_KEY")
    assert (
        next(e for e in events if e.stage == "decide" and e.status == "done").detail == "retrieval-only: 3 candidates"
    )
    assert all(m.decider_probability is None and m.confidence != "high" for m in result.fsc_codes)


async def test_gemini_only_uses_gemini_judge_and_names_the_missing_jev_key(world):
    world.settings = {}
    world.chain_factory = lambda gemini: [
        FakeDecider(name="gemini", model="gemini-2.5-flash"),
        decide.RetrievalOnlyDecider(),
    ]
    world.install()
    result, _ = await run()
    assert result.backend.decider == "gemini"
    assert [w for w in result.warnings if "TYPESAFE_API_KEY" in w] == [
        "TYPESAFE_API_KEY is not set: the Jev decision layer is off and Gemini judges the candidates. "
        "Set TYPESAFE_API_KEY to enable Jev."
    ]


async def test_decider_fallback_chain_jev_to_gemini_to_retrieval(world):
    gemini_fails = FakeDecider(name="gemini", model="gemini-2.5-flash", error=DeciderError("quota"))
    world.chain_factory = lambda g: [
        FakeDecider(error=DeciderError("authentication failed (401) - check TYPESAFE_API_KEY")),
        gemini_fails,
        decide.RetrievalOnlyDecider(),
    ]
    world.install()
    result, events = await run()

    assert result.backend.decider == "retrieval" and result.backend.decider_model is None
    assert result.warnings == [
        "The jev decider failed (authentication failed (401) - check TYPESAFE_API_KEY); fell back to gemini.",
        "The gemini decider failed (quota); fell back to retrieval.",
    ]
    assert next(e for e in events if e.stage == "decide" and e.status == "done").status == "done"
    assert result.fsc_codes


async def test_unexpected_decider_crash_also_falls_back(world):
    world.chain_factory = lambda g: [FakeDecider(error=KeyError("boom")), decide.RetrievalOnlyDecider()]
    world.install()
    result, _ = await run()
    assert result.backend.decider == "retrieval"
    assert "jev decider failed" in result.warnings[0]


async def test_failed_recall_channel_and_failed_stage_degrade_with_warnings(world):
    world.retriever = FakeRetriever(error=RuntimeError("index missing"))
    world.profile_error = ValueError("bad json")
    world.install()
    result, events = await run()

    profile_event = next(e for e in events if e.stage == "profile" and e.status != "running")
    assert profile_event.status == "error" and profile_event.detail == "ValueError: bad json"
    assert result.profile.name == "Acme Fasteners" and result.profile.products == []  # minimal profile, run continues
    assert result.backend.retrieval == ["llm"]  # only the surviving channel is reported
    assert any("BM25/embedding recall failed (RuntimeError: index missing)" in w for w in result.warnings)
    assert not any("Embedding recall is unavailable" in w for w in result.warnings)
    assert any("profile stage failed" in w for w in result.warnings)
    assert [m.code for m in result.fsc_codes] == ["5305"]
    assert [s.status for s in result.stages if s.stage == "profile"] == ["error"]


async def test_missing_embedding_channel_is_called_out_when_gemini_is_configured(world):
    world.retriever.active_channels = ["bm25"]
    world.install()
    result, _ = await run()
    assert result.backend.retrieval == ["bm25", "llm"]
    assert any("Embedding recall is unavailable" in w for w in result.warnings)


async def test_failed_sources_are_reported_with_their_errors(world):
    world.site_docs = [SourceDoc(kind="website", label="https://acme.test/", ok=False, error="HTTP 403")]
    world.file_doc = SourceDoc(kind="file", label="scan.png", ok=False, error="needs GEMINI_API_KEY for OCR")
    world.install()
    result, _ = await run(CompanyInput(name="Acme", website_url="https://acme.test", notes="x" * 2000))

    failed = next(w for w in result.warnings if "sources failed" in w)
    assert (
        failed.startswith("2 of 3 sources failed") and "HTTP 403" in failed and "needs GEMINI_API_KEY for OCR" in failed
    )
    assert [s.ok for s in result.sources] == [True, False, False]


async def test_invalid_website_url_is_ignored_with_a_warning(world):
    world.install()
    result, _ = await run(CompanyInput(name="Acme", website_url="not a url", notes="x" * 2000), files=())
    assert "Website URL 'not a url' is not valid and was ignored." in result.warnings
    assert [s.kind for s in result.sources] == ["notes"]


async def test_nothing_to_classify_is_a_hard_error(world):
    world.site_docs, world.file_doc = [], SourceDoc(kind="file", label="x", ok=False, error="unreadable")
    world.install()
    with pytest.raises(PipelineError, match="no company name"):
        await run(CompanyInput(name="  "), files=(PDF,))


async def test_emit_is_optional(world):
    world.install()
    result = await run_pipeline(COMPANY, [], None)
    assert result.fsc_codes


async def test_concurrent_runs_do_not_share_state(world):
    deciders: list[FakeDecider] = []

    def chain(gemini: Any) -> list[Any]:
        deciders.append(FakeDecider(delay=True))
        return [deciders[-1], decide.RetrievalOnlyDecider()]

    world.chain_factory = chain
    world.install()
    names = [f"Company {i}" for i in range(8)]
    results = await asyncio.gather(*(run_pipeline(CompanyInput(name=n, notes="bolts"), []) for n in names))

    assert [r.company.name for r in results] == names
    assert [r.profile.name for r in results] == names
    assert sorted(d.seen_profile.name for d in deciders) == sorted(names)  # each decider saw exactly its own run
    assert all([m.code for m in r.fsc_codes] == ["5306", "5305", "4730"] for r in results)


async def test_startup_builds_the_runtime_once_even_when_called_concurrently(monkeypatch):
    built: list[str] = []

    class SlowRetriever:
        active_channels = ["bm25"]

        def __init__(self, catalog: Any, gemini: Any) -> None:
            built.append("init")

        async def build(self) -> None:
            await asyncio.sleep(0.01)
            built.append("build")

    monkeypatch.setattr(pipeline, "_runtime", None)
    monkeypatch.setattr(pipeline, "load_catalog", lambda: CATALOG)
    monkeypatch.setattr(pipeline, "get_gemini", lambda: None)
    monkeypatch.setattr(pipeline, "Retriever", SlowRetriever)

    runtimes = await asyncio.gather(*(pipeline.startup() for _ in range(5)))
    assert built == ["init", "build"] and all(r is runtimes[0] for r in runtimes)
    assert runtimes[0].catalog is CATALOG and runtimes[0].gemini is None


async def test_startup_survives_a_failing_retriever_build(monkeypatch):
    class BrokenRetriever:
        active_channels: list[str] = []

        def __init__(self, catalog: Any, gemini: Any) -> None: ...

        async def build(self) -> None:
            raise RuntimeError("no index")

    monkeypatch.setattr(pipeline, "_runtime", None)
    monkeypatch.setattr(pipeline, "load_catalog", lambda: CATALOG)
    monkeypatch.setattr(pipeline, "get_gemini", lambda: None)
    monkeypatch.setattr(pipeline, "Retriever", BrokenRetriever)
    assert (await pipeline.startup()).retriever.active_channels == []


async def test_retriever_routing_priors_reach_scoring_and_the_response(world):
    world.retriever.routing = Routing(group_prior={"47": 1.0}, domain_prior={"S9C/AX": 1.0})
    world.install()
    result, _ = await run()
    by_code = {m.code: m for m in result.fsc_codes}
    assert by_code["4730"].signals["group_prior"] == 1.0 and by_code["4730"].signals["domain_prior"] == 1.0
    assert by_code["4730"].signals["boost"] == 0.05  # a judge ran: k = 0.05
    assert by_code["5305"].signals["group_prior"] == 0.0 and by_code["5305"].signals["domain_prior"] == 1.0
    assert by_code["5305"].supply_domain == "Construction" and by_code["5305"].agency == "DLA"
