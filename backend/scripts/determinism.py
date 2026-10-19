"""Repeatability test: run the full pipeline N times on ONE company and measure how the results differ.

    cd backend
    uv run python -m scripts.determinism run --name "Chicago Nut & Bolt" --url https://cnb-inc.com/ --runs 500
    uv run python -m scripts.determinism report --dir .cache/determinism/cnb

The website is crawled ONCE and the text frozen (evidence.json), so the runs differ only through the models and our own code,
not through the site changing and not through 500x6 page loads on the company's server. Every run records the profile, the
candidate pool and the Jev probabilities, so the report can say WHICH stage introduces the variance.

The repeatability settings are the app's own and come from the environment, so the same script measures any setup:
    LLM_CACHE=false                 measure the raw model variance (no memoised outputs)
    GEMINI_SEED=none|<int>          fixed sampling seed on every Gemini call (default 7)
    GEMINI_TEMPERATURE=<float>      force a temperature, even for gemini-3 models (default: model default)
Use a different --label per setup; each label is its own directory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import statistics
import time
from collections import Counter
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from app import pipeline
from app.config import BACKEND_DIR, get_settings
from app.fsc import load_catalog
from app.ingest.web import fetch_site
from app.schemas import CompanyInput, SourceDoc

OUT_ROOT = BACKEND_DIR / ".cache" / "determinism"
_REC: ContextVar[dict[str, Any]] = ContextVar("rec")  # per-run scratch dict, filled by the stage wrappers below


# --------------------------------------------------------------------------- instrumentation


def _install_recorders() -> None:
    """Wrap three pipeline stages so each run leaves a trace of its intermediate results (module globals are patched)."""
    build_profile, retrieve, decide = pipeline.build_profile, pipeline._retrieve, pipeline._decide

    async def profile_w(*args: Any, **kwargs: Any):
        out = await build_profile(*args, **kwargs)
        _REC.get()["profile"] = out.model_dump()
        return out

    async def retrieve_w(*args: Any, **kwargs: Any):
        cands, channels, routing = await retrieve(*args, **kwargs)
        _REC.get()["pool"] = [{"code": c.code, "rs": round(c.retrieval_score, 4), "src": c.sources} for c in cands]
        return cands, channels, routing

    async def decide_w(*args: Any, **kwargs: Any):
        decider, decisions = await decide(*args, **kwargs)
        _REC.get()["decider"] = {
            "name": decider.name,
            "model": decider.model,
            "p": {d.code: round(d.probability, 4) for d in decisions},
            "group_prior": {k: round(v, 4) for k, v in decider.group_prior.items() if v >= 0.01},
        }
        return decider, decisions

    pipeline.build_profile, pipeline._retrieve, pipeline._decide = profile_w, retrieve_w, decide_w


def _freeze_site(docs: list[SourceDoc]) -> None:
    async def frozen(url: str, **_: Any) -> list[SourceDoc]:
        return [d.model_copy(deep=True) for d in docs]

    pipeline.fetch_site = frozen  # type: ignore[assignment]


# --------------------------------------------------------------------------- run


def _key(items: list[str]) -> str:
    return hashlib.sha1("\n".join(sorted(i.strip().lower() for i in items)).encode()).hexdigest()[:10]


async def _one(i: int, company: CompanyInput, timeout: float) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    _REC.set(rec)
    started = time.perf_counter()
    try:
        res = await asyncio.wait_for(pipeline.run_pipeline(company, []), timeout)
    except Exception as exc:  # a failed run is data too, kept apart from "different answer"
        return {"i": i, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}", "ms": int((time.perf_counter() - started) * 1000)}
    products = [p["name"] for p in rec.get("profile", {}).get("products", [])]
    return {
        "i": i,
        "ok": True,
        "ms": res.total_ms,
        "codes": [
            {"code": m.code, "score": m.score, "conf": m.confidence, "dp": m.decider_probability, "rs": m.retrieval_score}
            for m in res.fsc_codes
        ],
        "products": products,
        "profile_key": _key(products),
        "pool": rec.get("pool", []),
        "decider": rec.get("decider", {}),
        "warnings": res.warnings,
        "backend": res.backend.model_dump(),
    }


async def cmd_run(args: argparse.Namespace) -> None:
    out = OUT_ROOT / args.label
    out.mkdir(parents=True, exist_ok=True)
    company = CompanyInput(name=args.name, website_url=args.url)
    settings = get_settings()

    # 1. freeze the evidence (one polite crawl) and check that the crawl itself is stable
    evidence = out / "evidence.json"
    if evidence.exists():
        docs = [SourceDoc(**d) for d in json.loads(evidence.read_text())]
        print(f"evidence: reusing {evidence} ({sum(len(d.text) for d in docs):,} chars)")
    else:
        docs = await fetch_site(args.url, max_pages=settings.max_site_pages)
        evidence.write_text(json.dumps([d.model_dump() for d in docs], indent=1))
        digests = {hashlib.sha1("".join(d.text for d in docs).encode()).hexdigest()[:10]}
        for _ in range(args.crawl_check):
            again = await fetch_site(args.url, max_pages=settings.max_site_pages)
            digests.add(hashlib.sha1("".join(d.text for d in again).encode()).hexdigest()[:10])
        print(f"evidence: crawled {len(docs)} pages, {sum(len(d.text) for d in docs):,} chars; "
              f"{args.crawl_check + 1} crawls gave {len(digests)} distinct text digest(s)")
        (out / "crawl_check.json").write_text(json.dumps({"crawls": args.crawl_check + 1, "distinct_digests": len(digests)}))
    _freeze_site(docs)
    _install_recorders()

    # 2. the runs
    (out / "config.json").write_text(json.dumps({
        "name": args.name, "url": args.url, "runs": args.runs, "concurrency": args.concurrency,
        "gemini_model": settings.gemini_model, "decider": settings.decider, "gemini_seed": settings.gemini_seed,
        "gemini_temperature": settings.gemini_temperature, "llm_cache": settings.llm_cache,
        "min_score": settings.min_score, "jev_weight": settings.jev_weight, "web_research": settings.enable_web_research,
    }, indent=1))
    runs_file = out / "runs.jsonl"
    done_ids = {json.loads(line)["i"] for line in runs_file.read_text().splitlines()} if runs_file.exists() else set()
    todo = [i for i in range(args.runs) if i not in done_ids]
    print(f"runs: {len(done_ids)} already done, {len(todo)} to go, concurrency {args.concurrency}")
    await pipeline.startup()  # builds BM25 + embedding cache once, before the timer

    sem, lock, stats = asyncio.Semaphore(args.concurrency), asyncio.Lock(), Counter()
    t0 = time.perf_counter()

    async def worker(i: int) -> None:
        async with sem:
            if stats["abort"]:
                return
            row = await _one(i, company, args.timeout)
            async with lock:
                with runs_file.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                stats["ok" if row["ok"] else "err"] += 1
                if not row["ok"] and stats["ok"] == 0 and stats["err"] >= args.abort_after:
                    stats["abort"] = 1
                    print(f"ABORT: first {stats['err']} runs all failed, last error: {row['error']}")
                n = stats["ok"] + stats["err"]
                if n % args.progress == 0 or n == len(todo):
                    rate = n / (time.perf_counter() - t0)
                    print(f"  {n}/{len(todo)}  ok={stats['ok']} err={stats['err']}  {rate:.2f} runs/s  eta {(len(todo) - n) / rate / 60:.1f} min", flush=True)

    await asyncio.gather(*(worker(i) for i in todo))
    print(f"finished in {(time.perf_counter() - t0) / 60:.1f} min -> {runs_file}")
    print(report(out))


# --------------------------------------------------------------------------- report


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def report(directory: Path) -> str:
    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines() if line.strip()]
    good = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    catalog, cfg = load_catalog(), json.loads((directory / "config.json").read_text())
    desc = lambda c: (catalog.get(c).description if catalog.get(c) else "?")  # noqa: E731
    out: list[str] = []
    p = out.append

    p(f"# Repeatability report: {cfg['name']}  ({cfg['url']})")
    seed = cfg.get("gemini_seed", cfg.get("seed"))  # older runs used the key "seed"
    temp = cfg.get("gemini_temperature", cfg.get("temperature_override"))
    p(f"model={cfg['gemini_model']} decider={cfg['decider']} seed={seed} temperature={'model default' if temp is None else temp} "
      f"llm_cache={cfg.get('llm_cache', False)}")
    p(f"\n## Runs\n- {len(rows)} runs: **{len(good)} ok**, {len(bad)} failed")
    for err, n in Counter(r["error"][:90] for r in bad).most_common(5):
        p(f"  - {n}x {err}")
    if not good:
        return "\n".join(out)
    ms = sorted(r["ms"] for r in good)
    p(f"- latency: median {statistics.median(ms) / 1000:.1f}s, p95 {ms[int(0.95 * len(ms)) - 1] / 1000:.1f}s")
    p(f"- decider used: {dict(Counter(r['backend']['decider'] for r in good))}; "
      f"with warnings: {sum(1 for r in good if r['warnings'])}")
    crawl = directory / "crawl_check.json"
    if crawl.exists():
        c = json.loads(crawl.read_text())
        p(f"- website crawl stability: {c['crawls']} crawls -> {c['distinct_digests']} distinct text digest(s)")

    sets = [frozenset(c["code"] for c in r["codes"]) for r in good]
    orders = [tuple(c["code"] for c in r["codes"]) for r in good]
    n = len(good)
    set_counts, order_counts = Counter(sets), Counter(orders)
    modal_set, modal_n = set_counts.most_common(1)[0]
    p("\n## Is the final answer the same every time?")
    p(f"- distinct code SETS: **{len(set_counts)}** across {n} runs; the most common set appears in **{_pct(modal_n, n)}** of runs")
    p(f"- distinct ORDERED lists: {len(order_counts)}; most common order: {_pct(order_counts.most_common(1)[0][1], n)}")
    top1 = Counter(o[0] if o else "-" for o in orders)
    p(f"- top-1 code: {', '.join(f'{c} {_pct(k, n)}' for c, k in top1.most_common(5))}")
    jac = [len(a & b) / len(a | b) for a, b in itertools.combinations(list(set_counts.elements())[:400], 2) if a | b]
    p(f"- pairwise Jaccard of code sets (first 400 runs): mean {statistics.mean(jac):.3f}, min {min(jac):.3f}" if jac else "")
    p("\nTop 6 distinct sets:")
    for s, k in set_counts.most_common(6):
        p(f"  - {_pct(k, n):>6}  {', '.join(sorted(s))}")

    p("\n## Per-code stability (inclusion rate across runs)")
    p("| code | description | in % runs | mean score | score sd | mean rank | confidence mix |")
    p("|---|---|---|---|---|---|---|")
    per: dict[str, list[tuple[int, dict]]] = {}
    for r in good:
        for rank, c in enumerate(r["codes"], 1):
            per.setdefault(c["code"], []).append((rank, c))
    for code, hits in sorted(per.items(), key=lambda kv: -len(kv[1]))[:20]:
        sc = [c["score"] for _, c in hits]
        mix = Counter(c["conf"] for _, c in hits)
        p(f"| {code} | {desc(code)[:42]} | {_pct(len(hits), n)} | {statistics.mean(sc):.2f} | "
          f"{statistics.pstdev(sc):.3f} | {statistics.mean(r for r, _ in hits):.1f} | {dict(mix)} |")
    core = [c for c, h in per.items() if len(h) / n >= 0.95]
    flicker = [c for c, h in per.items() if 0.05 <= len(h) / n < 0.95]
    p(f"\ncore (>=95% of runs): {sorted(core)}\nflickering (5-95%): {sorted(flicker)}\nrare (<5%): {sum(1 for h in per.values() if len(h) / n < 0.05)} codes")

    p("\n## Where does the variance come from?")
    prof = Counter(r["profile_key"] for r in good)
    modal_prof, mp_n = prof.most_common(1)[0]
    p(f"- **profile** (Gemini): {len(prof)} distinct product-line sets; the most common appears in {_pct(mp_n, n)} of runs; "
      f"lines per run: {dict(sorted(Counter(len(r['products']) for r in good).items()))}")
    name_counts = Counter(x.strip().lower() for r in good for x in r["products"])
    p("  - product-line names seen: " + "; ".join(f"{k} ({_pct(v, n)})" for k, v in name_counts.most_common(8)))
    same_prof = [r for r in good if r["profile_key"] == modal_prof]
    pools = {r["i"]: frozenset(x["code"] for x in r["pool"]) for r in good}
    pool_counts = Counter(pools[r["i"]] for r in same_prof)
    modal_pool, mpl_n = pool_counts.most_common(1)[0]
    p(f"- **candidate pool**, among the {len(same_prof)} runs with the modal profile: {len(pool_counts)} distinct pools; "
      f"modal pool in {_pct(mpl_n, len(same_prof))}")
    by_src: dict[str, Counter] = {s: Counter() for s in ("bm25", "embedding", "llm")}
    for r in same_prof:
        for s in by_src:
            by_src[s][frozenset(x["code"] for x in r["pool"] if s in x["src"])] += 1
    p("  - distinct pools per channel (modal profile only): " + ", ".join(f"{s}={len(c)}" for s, c in by_src.items()))
    same_pool = [r for r in same_prof if pools[r["i"]] == modal_pool]
    fin = Counter(frozenset(c["code"] for c in r["codes"]) for r in same_pool)
    p(f"- **final set**, among the {len(same_pool)} runs with modal profile AND modal pool: {len(fin)} distinct final sets "
      f"(modal {_pct(fin.most_common(1)[0][1], len(same_pool)) if fin else 'n/a'})")
    # is Jev itself repeatable? same profile + same candidate in pool -> same probability?
    probs: dict[tuple[str, str], list[float]] = {}
    for r in same_prof:
        for code, pr in (r["decider"].get("p") or {}).items():
            probs.setdefault((r["profile_key"], code), []).append(pr)
    multi = [v for v in probs.values() if len(v) >= 3]
    if multi:
        sds = [statistics.pstdev(v) for v in multi]
        p(f"- **Jev** (same profile, same candidate): {len(multi)} code/profile pairs with >=3 runs; "
          f"sd of probability: median {statistics.median(sds):.4f}, max {max(sds):.4f}, "
          f"{sum(1 for s in sds if s > 1e-6)} pairs not bit-identical")
    off = [(r, s) for r, s in zip(good, sets, strict=True) if s != modal_set]
    if off:
        cause = Counter(
            "profile differs" if r["profile_key"] != modal_prof
            else "pool differs (LLM recall channel)" if pools[r["i"]] != modal_pool
            else "decisions/scoring differ"
            for r, _ in off
        )
        p(f"\nOf the {len(off)} runs whose final set differs from the modal set, first divergence at:")
        for k, v in cause.most_common():
            p(f"  - {k}: {v} ({_pct(v, len(off))})")
    verdict = "REPEATABLE" if len(set_counts) == 1 else "NOT repeatable"
    p(f"\n**Verdict: {verdict}** - {_pct(modal_n, n)} of runs return the modal set; {len(set_counts)} distinct sets in {n} runs.")
    text = "\n".join(x for x in out if x is not None)
    (directory / "report.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--url", required=True)
    r.add_argument("--runs", type=int, default=500)
    r.add_argument("--concurrency", type=int, default=6)
    r.add_argument("--label", default="cnb")
    r.add_argument("--timeout", type=float, default=240.0)
    r.add_argument("--crawl-check", type=int, default=4, help="extra crawls to test that the site text itself is stable")
    r.add_argument("--abort-after", type=int, default=8, help="stop if this many runs fail before any succeeds")
    r.add_argument("--progress", type=int, default=25)
    rep = sub.add_parser("report")
    rep.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        asyncio.run(cmd_run(args))
    else:
        print(report(args.dir))


if __name__ == "__main__":
    main()
