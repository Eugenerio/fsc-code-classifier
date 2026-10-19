"""Shared data contract. Every module and the frontend types are written against this file.

Do not change field names/types without updating docs/ARCHITECTURE.md and frontend/lib/types.ts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- input


class CompanyInput(BaseModel):
    name: str
    website_url: str | None = None
    email_domain: str | None = None
    notes: str | None = None


class UploadedFile(BaseModel):
    filename: str
    content_type: str | None = None
    data: bytes


# --------------------------------------------------------------------------- FSC catalog


class FscCode(BaseModel):
    code: str  # always 4 chars, zero padded, e.g. "5305"
    description: str
    sos_ric: str  # e.g. "S9C/AX"
    group: str  # first two digits, e.g. "53"
    group_name: str  # e.g. "Hardware and Abrasives"
    marker: str | None = None  # "**" = GSA + DLA (commercial goods), "*" = subsistence / live animals
    # Derived from the Legend sheet by the catalog loader (see docs/ARCHITECTURE.md, "FSC feature vector")
    agency: str = ""  # "DLA" | "GSA": the marketplace a solicitation for this class lives in
    supply_domain: str = ""  # short buying-centre label cutting across groups, e.g. "General & Industrial"
    marker_note: str | None = None

    @property
    def marker_kind(self) -> Literal["commercial", "subsistence"] | None:
        return {"**": "commercial", "*": "subsistence"}.get(self.marker or "")


# --------------------------------------------------------------------------- evidence

SourceKind = Literal["website", "file", "notes", "web_research"]


class SourceDoc(BaseModel):
    """Internal: one piece of evidence with its extracted text."""

    kind: SourceKind
    label: str  # page URL, filename, "notes", "google_search"
    text: str = ""
    ok: bool = True
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)  # e.g. {"ocr": True, "pages": 3}


class SourceInfo(BaseModel):
    """API response: SourceDoc without the text."""

    kind: SourceKind
    label: str
    chars: int
    ok: bool
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- profile

Role = Literal["manufacturer", "distributor", "reseller", "service", "unknown"]


class ProductLine(BaseModel):
    name: str  # short, catalog-style noun phrase: "stainless steel hex bolts"
    description: str = ""  # one sentence, what it is / used for
    keywords: list[str] = Field(default_factory=list)  # synonyms in government-catalog vocabulary
    role: Role = "unknown"


class CompanyProfile(BaseModel):
    name: str
    summary: str = ""
    business_types: list[str] = Field(default_factory=list)  # e.g. ["manufacturer", "machine shop"]
    products: list[ProductLine] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)  # oil & gas, aerospace, ...


# --------------------------------------------------------------------------- retrieval / decision

CandidateSource = Literal["bm25", "embedding", "llm"]


class Candidate(BaseModel):
    code: str
    retrieval_score: float  # 0..1, fused + normalised (1.0 == best candidate)
    sources: list[CandidateSource] = Field(default_factory=list)
    matched_products: list[str] = Field(default_factory=list)  # ProductLine.name that surfaced it
    llm_reason: str | None = None  # from the "LLM reads the whole FSC list" channel


class Decision(BaseModel):
    code: str
    probability: float  # 0..1 that the company supplies products in this FSC class
    confidence: float | None = None  # decider-reported certainty (Jev) if available
    rationale: str | None = None


# --------------------------------------------------------------------------- output

Confidence = Literal["high", "medium", "low"]


class FscMatch(BaseModel):
    code: str
    description: str
    sos_ric: str
    group: str
    group_name: str
    score: float  # final 0..1 ranking score
    confidence: Confidence
    decider_probability: float | None = None  # Jev / Gemini probability
    retrieval_score: float = 0.0
    matched_products: list[str] = Field(default_factory=list)
    rationale: str | None = None
    agency: str = ""
    supply_domain: str = ""
    marker_note: str | None = None
    # How the score was assembled: {"group_prior", "domain_prior", "boost", "marker_guard"}
    signals: dict[str, float] = Field(default_factory=dict)


Stage = Literal["ingest", "research", "profile", "retrieve", "decide", "rank"]


class StageTiming(BaseModel):
    stage: Stage
    status: Literal["done", "skipped", "error"]
    elapsed_ms: int = 0
    detail: str | None = None


class BackendInfo(BaseModel):
    extractor: str  # e.g. "gemini-2.5-flash" or "heuristic"
    decider: Literal["jev", "gemini", "retrieval"]
    decider_model: str | None = None  # e.g. "jev-1.13.0"
    retrieval: list[CandidateSource] = Field(default_factory=list)  # channels that were active


class ClassifyResponse(BaseModel):
    company: CompanyInput
    profile: CompanyProfile
    fsc_codes: list[FscMatch]  # sorted by score desc
    sources: list[SourceInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stages: list[StageTiming] = Field(default_factory=list)
    backend: BackendInfo
    total_ms: int = 0


# --------------------------------------------------------------------------- streaming (NDJSON)


class StageEvent(BaseModel):
    event: Literal["stage"] = "stage"
    stage: Stage
    status: Literal["running", "done", "skipped", "error"]
    detail: str | None = None
    elapsed_ms: int | None = None


class ResultEvent(BaseModel):
    event: Literal["result"] = "result"
    data: ClassifyResponse


class ErrorEvent(BaseModel):
    event: Literal["error"] = "error"
    message: str
