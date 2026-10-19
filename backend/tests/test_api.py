from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import decide, main, pipeline, scoring
from app.config import Settings
from app.fsc import FscCatalog
from app.pipeline import PipelineError, Runtime
from app.schemas import (
    BackendInfo,
    ClassifyResponse,
    CompanyInput,
    CompanyProfile,
    FscCode,
    StageEvent,
    UploadedFile,
)

CATALOG = FscCatalog(
    [
        FscCode(code=c, description=d, sos_ric="S9C/AX", group=c[:2], group_name=n)
        for c, d, n in [
            ("5305", "Screws", "Hardware and Abrasives"),
            ("5306", "Bolts", "Hardware and Abrasives"),
            ("5310", "Nuts and Washers", "Hardware and Abrasives"),
            ("4730", "Hose, Pipe, Tube Fittings", "Pipe, Tubing, Hose, and Fittings"),
            ("5120", "Hand Tools, Nonedged, Nonpowered", "Hand Tools"),
        ]
    ]
)


class FakeRetriever:
    active_channels = ["bm25", "embedding"]


def settings(**kw: Any) -> Settings:
    return Settings(_env_file=None, **{"gemini_api_key": None, "typesafe_api_key": None, **kw})


def response_for(company: CompanyInput) -> ClassifyResponse:
    return ClassifyResponse(
        company=company,
        profile=CompanyProfile(name=company.name),
        fsc_codes=[],
        backend=BackendInfo(extractor="heuristic", decider="retrieval"),
    )


@pytest.fixture
def seen(monkeypatch) -> dict[str, Any]:
    """Replaces the pipeline behind the endpoints and records what main hands to it."""
    record: dict[str, Any] = {"events": [StageEvent(stage="ingest", status="running")]}

    async def fake_run(company: CompanyInput, files: list[UploadedFile], emit=None) -> ClassifyResponse:
        record.update(company=company, files=files)
        for event in record["events"]:
            if emit:
                await emit(event)
        if error := record.get("error"):
            raise error
        return response_for(company)

    monkeypatch.setattr(main, "run_pipeline", fake_run)
    return record


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(pipeline, "_runtime", Runtime(CATALOG, FakeRetriever(), None))  # type: ignore[arg-type]
    monkeypatch.setattr(main, "get_settings", lambda: settings(max_upload_mb=1, max_files=2))
    monkeypatch.setattr(decide, "get_settings", lambda: settings())
    return TestClient(main.app)  # no `with`: skip the lifespan, the runtime is already installed


def form(**fields: str) -> dict[str, str]:
    return {"company_name": "Acme Fasteners", **fields}


def txt(name: str = "notes.txt", data: bytes = b"hello") -> tuple[str, tuple[str, bytes, str]]:
    return ("files", (name, data, "text/plain"))


# --------------------------------------------------------------------------- health + lookup


def test_health_reports_active_backends(client, monkeypatch):
    body = client.get("/api/health").json()
    assert body == {
        "status": "ok",
        "gemini": False,
        "jev": False,
        "decider": "retrieval",
        "fsc_count": 5,
        "retrieval": ["bm25", "embedding"],
    }
    monkeypatch.setattr(main, "get_settings", lambda: settings(typesafe_api_key="tk"))
    monkeypatch.setattr(decide, "get_settings", lambda: settings(typesafe_api_key="tk"))
    assert client.get("/api/health").json() | {"x": 0} == body | {"jev": True, "decider": "jev", "x": 0}


def test_fsc_lookup_ranks_code_prefix_then_words(client):
    assert [c["code"] for c in client.get("/api/fsc", params={"q": "bolt"}).json()] == ["5306"]
    assert [c["code"] for c in client.get("/api/fsc", params={"q": "53"}).json()] == ["5305", "5306", "5310"]
    assert [c["code"] for c in client.get("/api/fsc", params={"q": "hand tools"}).json()] == ["5120"]
    assert [c["code"] for c in client.get("/api/fsc", params={"q": "hardware nuts"}).json()] == [
        "5310"
    ]  # group name + word
    assert client.get("/api/fsc", params={"q": "zzz"}).json() == []
    assert len(client.get("/api/fsc", params={"limit": 2}).json()) == 2
    assert client.get("/api/fsc", params={"limit": 0}).status_code == 422
    first = client.get("/api/fsc", params={"q": "screws"}).json()[0]
    assert first == CATALOG.get("5305").model_dump()


# --------------------------------------------------------------------------- classify


def test_classify_parses_form_and_files(client, seen):
    r = client.post(
        "/api/classify",
        data=form(website_url="  https://acme.test  ", email_domain="Sales@ACME.test", notes="  hello \x00 world  "),
        files=[txt("a.txt", b"alpha"), txt("../../etc/b.md", b"beta")],
    )
    assert r.status_code == 200, r.text
    assert r.json()["company"] == {
        "name": "Acme Fasteners",
        "website_url": "https://acme.test",
        "email_domain": "acme.test",
        "notes": "hello  world",
    }
    assert [(f.filename, f.data, f.content_type) for f in seen["files"]] == [
        ("a.txt", b"alpha", "text/plain"),
        ("b.md", b"beta", "text/plain"),  # path components stripped
    ]
    ClassifyResponse.model_validate(r.json())


def test_classify_accepts_bare_domain_and_blank_optionals(client, seen):
    r = client.post("/api/classify", data=form(email_domain="acme.test", website_url="", notes="   "))
    assert r.status_code == 200
    assert seen["company"] == CompanyInput(name="Acme Fasteners", email_domain="acme.test")
    assert seen["files"] == []


def test_classify_limits_text_fields(client, seen):
    r = client.post("/api/classify", data=form(company_name="N" * 500, notes="x" * 30_000))
    assert r.status_code == 200
    assert len(seen["company"].name) == 200 and len(seen["company"].notes) == 20_000


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"company_name": "   "}, "company_name must not be empty"),
        (form(email_domain="not a domain"), "email_domain"),
        (form(email_domain="a@b"), "email_domain"),
    ],
)
def test_classify_validation_errors_are_422(client, seen, data, fragment):
    r = client.post("/api/classify", data=data)
    assert r.status_code == 422 and fragment in r.json()["detail"]
    assert "company" not in seen  # the pipeline never ran


def test_classify_missing_name_is_422(client, seen):
    assert client.post("/api/classify", data={"notes": "x"}).status_code == 422


def test_classify_too_many_files_is_422_and_oversize_file_is_413(client, seen):
    r = client.post("/api/classify", data=form(), files=[txt(f"{i}.txt") for i in range(3)])
    assert r.status_code == 422 and "At most 2 files" in r.json()["detail"]

    r = client.post("/api/classify", data=form(), files=[txt("ok.txt"), txt("big.pdf", b"x" * (1024 * 1024 + 1))])
    assert r.status_code == 413 and "big.pdf" in r.json()["detail"] and "1 MB" in r.json()["detail"]

    exactly = client.post("/api/classify", data=form(), files=[txt("edge.txt", b"x" * 1024 * 1024)])
    assert exactly.status_code == 200


def test_classify_ignores_empty_file_parts(client, seen):
    r = client.post("/api/classify", data=form(), files=[("files", ("", b"", "application/octet-stream"))])
    assert r.status_code == 200 and seen["files"] == []


def test_classify_pipeline_error_is_422(client, seen):
    seen["error"] = PipelineError("Nothing to classify")
    r = client.post("/api/classify", data=form())
    assert r.status_code == 422 and r.json() == {"detail": "Nothing to classify"}


# --------------------------------------------------------------------------- NDJSON stream


def test_stream_returns_ndjson_events_then_exactly_one_result(client, seen):
    seen["events"] = [
        StageEvent(stage="ingest", status="running"),
        StageEvent(stage="ingest", status="done", detail="1 file", elapsed_ms=3),
    ]
    r = client.post("/api/classify/stream", data=form(), files=[txt()])
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert r.headers["x-accel-buffering"] == "no"
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert [(e["event"], e.get("stage"), e.get("status")) for e in lines] == [
        ("stage", "ingest", "running"),
        ("stage", "ingest", "done"),
        ("result", None, None),
    ]
    assert lines[1]["detail"] == "1 file" and lines[1]["elapsed_ms"] == 3
    assert r.text.endswith("\n")
    ClassifyResponse.model_validate(lines[-1]["data"])


def test_stream_validation_errors_are_plain_http_errors(client, seen):
    r = client.post("/api/classify/stream", data={"company_name": " "})
    assert r.status_code == 422 and r.headers["content-type"].startswith("application/json")


def test_stream_pipeline_failures_become_a_single_error_event(client, seen):
    seen["error"] = PipelineError("Nothing to classify")
    lines = [json.loads(line) for line in client.post("/api/classify/stream", data=form()).text.splitlines()]
    assert lines[-1] == {"event": "error", "message": "Nothing to classify"}
    assert [e["event"] for e in lines].count("result") == 0

    seen["error"] = RuntimeError("secret internals /etc/passwd")
    last = json.loads(client.post("/api/classify/stream", data=form()).text.splitlines()[-1])
    assert last == {"event": "error", "message": "Internal error: RuntimeError"}  # no leaked details


async def test_stream_relays_events_while_the_pipeline_is_still_running(monkeypatch):
    gate, finished = asyncio.Event(), asyncio.Event()

    async def slow_run(company, files, emit=None) -> ClassifyResponse:
        await emit(StageEvent(stage="ingest", status="running"))
        await gate.wait()
        await emit(StageEvent(stage="ingest", status="done"))
        finished.set()
        return response_for(company)

    monkeypatch.setattr(main, "run_pipeline", slow_run)
    stream = main._stream(main.Submission(CompanyInput(name="Acme"), []))
    first = json.loads(await asyncio.wait_for(stream.__anext__(), 1))
    assert (first["stage"], first["status"]) == ("ingest", "running") and not finished.is_set()
    gate.set()
    rest = [json.loads(line) async for line in stream]
    assert [e["event"] for e in rest] == ["stage", "result"]


async def test_stream_cancels_the_pipeline_when_the_client_disconnects(monkeypatch):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def forever(company, files, emit=None) -> ClassifyResponse:
        await emit(StageEvent(stage="ingest", status="running"))
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    monkeypatch.setattr(main, "run_pipeline", forever)
    stream = main._stream(main.Submission(CompanyInput(name="Acme"), []))
    await stream.__anext__()
    await started.wait()
    await stream.aclose()
    await asyncio.wait_for(cancelled.wait(), 1)


# --------------------------------------------------------------------------- real pipeline, no keys, no network


@pytest.fixture
def keyless(monkeypatch):
    for module in (pipeline, decide, scoring, main):
        monkeypatch.setattr(module, "get_settings", lambda: settings())
    monkeypatch.setattr(pipeline, "get_gemini", lambda: None)
    monkeypatch.setattr(pipeline, "_runtime", None)


BOLTS = (
    "Acme Fasteners manufactures stainless steel hex bolts, socket head cap screws, hex nuts and flat washers "
    "for oil field and marine customers. We also stock threaded rod and machine screws. " * 3
)


def test_keyless_end_to_end_with_the_real_catalog(keyless):
    with TestClient(main.app) as client:  # lifespan builds the real catalog + BM25 index
        health = client.get("/api/health").json()
        assert health["gemini"] is False and health["decider"] == "retrieval" and health["fsc_count"] > 500
        assert health["retrieval"] == ["bm25"]

        r = client.post("/api/classify", data=form(), files=[("files", ("catalog.txt", BOLTS.encode(), "text/plain"))])
        assert r.status_code == 200, r.text
        body = r.json()
        codes = [m["code"] for m in body["fsc_codes"]]
        assert 3 <= len(codes) <= 10
        assert {"5305", "5306", "5310"} & set(codes[:5]), codes  # screws / bolts / nuts surface near the top
        assert body["backend"] == {
            "extractor": "heuristic",
            "decider": "retrieval",
            "decider_model": None,
            "retrieval": ["bm25"],
        }
        assert [s["stage"] for s in body["stages"]] == ["ingest", "research", "profile", "retrieve", "decide", "rank"]
        assert any("GEMINI_API_KEY" in w for w in body["warnings"]) and any(
            "TYPESAFE_API_KEY" in w for w in body["warnings"]
        )
        assert all(m["confidence"] != "high" for m in body["fsc_codes"])

        stream = client.post(
            "/api/classify/stream", data=form(), files=[("files", ("catalog.txt", BOLTS.encode(), "text/plain"))]
        )
        events = [json.loads(line) for line in stream.text.splitlines()]
        assert events[0] == {
            "event": "stage",
            "stage": "ingest",
            "status": "running",
            "detail": None,
            "elapsed_ms": None,
        }
        assert events[-1]["event"] == "result" and [m["code"] for m in events[-1]["data"]["fsc_codes"]] == codes
