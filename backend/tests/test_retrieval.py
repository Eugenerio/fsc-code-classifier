from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import get_settings
from app.fsc import FscCatalog, load_catalog
from app.retrieval import SUMMARY_LABEL, Query, Retriever, Routing, build_queries, catalog_relevance, merge_candidates, tokenize
from app.schemas import Candidate, CompanyProfile, ProductLine

DIM = 4096  # larger than the catalog vocabulary, so stems never share a dimension
CONCEPTS = {"notebook": "comput", "ultrabook": "comput"}  # words the catalog lacks, in stemmed space
VOCAB: dict[str, int] = {}  # shared, like a real model's embedding space: every instance maps a stem to the same axis


class FakeEmbedder:
    """Deterministic bag-of-stems embedder: texts sharing stems are close, so the embedding channel is testable."""

    def __init__(self, *, fail_docs: bool = False, fail_queries: bool = False, short_docs: bool = False) -> None:
        self.calls: list[tuple[str, int]] = []
        self.fail_docs, self.fail_queries, self.short_docs = fail_docs, fail_queries, short_docs

    @property
    def doc_calls(self) -> int:
        return sum(task == "RETRIEVAL_DOCUMENT" for task, _ in self.calls)

    async def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        self.calls.append((task_type, len(texts)))
        if (task_type == "RETRIEVAL_DOCUMENT" and self.fail_docs) or (
            task_type == "RETRIEVAL_QUERY" and self.fail_queries
        ):
            raise RuntimeError("embedding backend down")
        if self.short_docs and task_type == "RETRIEVAL_DOCUMENT":
            texts = texts[:-1]
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                out[row, VOCAB.setdefault(CONCEPTS.get(tok, tok), len(VOCAB))] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


@pytest.fixture(scope="module")
def catalog() -> FscCatalog:
    return load_catalog()


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cache"
    monkeypatch.setattr(get_settings(), "cache_dir", path)
    return path


async def top_codes(retriever: Retriever, text: str, n: int = 10) -> list[str]:
    return [c.code for c in await retriever.search([Query(label="q", text=text)], k_per_query=n)]


# --------------------------------------------------------------------------- tokenisation


def test_tokenize_basics() -> None:
    assert tokenize("Nuts and Washers") == ["nut", "wash"]
    assert tokenize("the equipment, and miscellaneous items") == []  # catalog filler carries no signal
    assert tokenize("Guns, thru 30 mm") == ["gun", "30", "mm"]


def test_tokenize_conflates_inflections() -> None:
    assert tokenize("tubing") == tokenize("tubes") == tokenize("tube")
    assert tokenize("fasteners") == tokenize("Fastening")
    assert tokenize("batteries") == tokenize("battery")
    assert tokenize("drilling") == ["drill"]
    assert tokenize("fittings") == tokenize("fitting") == tokenize("fitter")
    assert tokenize("gases") == tokenize("gas")


def test_tokenize_expands_catalog_abbreviations() -> None:
    assert tokenize("Elect Arc Welding Equipment") == tokenize("electrical arc welding")
    assert tokenize("Maintenance and Rep Shop") == tokenize("maintenance and repair shop")
    assert "comput" in tokenize("ADP Software")
    assert tokenize("machinery") == tokenize("machines")


def test_tokenize_hyphenated_compounds_meet_run_together_spellings() -> None:
    assert "oring" in tokenize("O-Rings")
    assert tokenize("o-rings") == tokenize("O-RINGS")
    assert "nonpower" in tokenize("Non-powered") and "nonpower" in tokenize("nonpowered")


# --------------------------------------------------------------------------- build_queries


def test_build_queries() -> None:
    profile = CompanyProfile(
        name="Acme",
        summary="Makes fasteners.",
        products=[
            ProductLine(name="hex bolts", description="Steel bolts.", keywords=["cap screws", "studs"]),
            ProductLine(name="gaskets"),
        ],
        keywords=["hardware"],
        industries=["aerospace"],
    )
    queries = build_queries(profile)
    assert [q.label for q in queries] == ["hex bolts", "gaskets", SUMMARY_LABEL]
    assert queries[0].text == "hex bolts. Steel bolts. cap screws, studs"
    assert queries[1].text == "gaskets"
    assert queries[2].text == "Makes fasteners. hardware. aerospace"
    assert "Acme" not in " ".join(q.text for q in queries)


def test_build_queries_skips_empty_text() -> None:
    assert build_queries(CompanyProfile(name="Nobody")) == []
    only_products = build_queries(CompanyProfile(name="X", products=[ProductLine(name="valves")]))
    assert [q.label for q in only_products] == ["valves"]


# --------------------------------------------------------------------------- BM25 on the real catalog


async def test_keyless_retriever_is_bm25_only(catalog: FscCatalog, cache_dir: Path) -> None:
    retriever = Retriever(catalog, None)
    await retriever.build()
    assert retriever.active_channels == ["bm25"]
    assert not cache_dir.exists()


# Expectations below were checked by eye against the real 580-class catalog (see tokenisation tuning notes).
async def test_fasteners_rank_hardware_group_first(catalog: FscCatalog) -> None:
    codes = await top_codes(Retriever(catalog, None), "hex bolts, socket cap screws, nuts, washers, fasteners")
    assert {c[:2] for c in codes[:4]} == {"53"}
    assert set(codes[:3]) == {"5310", "5305", "5306"}


async def test_centrifugal_pumps_surface_pumps_group(catalog: FscCatalog) -> None:
    codes = await top_codes(Retriever(catalog, None), "centrifugal pumps")
    assert {c[:2] for c in codes[:3]} == {"43"}
    assert {"4320", "4330"} <= set(codes)


async def test_tires_surface_tires_group(catalog: FscCatalog) -> None:
    assert (await top_codes(Retriever(catalog, None), "tires"))[0] == "2620"
    assert (await top_codes(Retriever(catalog, None), "tyre"))[0] == "2620"  # British spelling alias


async def test_oilfield_vocabulary_reaches_drilling_and_petroleum_classes(catalog: FscCatalog) -> None:
    codes = await top_codes(Retriever(catalog, None), "oil well drilling tools downhole")
    assert codes[0] == "3820"  # Mining, Rock Drilling, Earth Boring
    assert "3835" in codes  # Petroleum Production and Distribution Equipment
    assert "5133" in codes  # Drill Bits


@pytest.mark.parametrize(
    ("phrase", "expected_first", "expected_in_top5"),
    [
        ("ball bearings and roller bearings", None, {"3110", "3120", "3130"}),
        ("gate valves, ball valves, check valves", None, {"4810", "4820"}),
        ("wire rope and chain slings", "4010", {"4010", "4030", "4020"}),
        ("o-rings and gaskets seals", "5331", {"5330", "5331"}),
        ("hydraulic hoses and fittings", "4730", {"4720", "4730"}),
        ("welding supplies", "3439", {"3438", "3431", "3432"}),
        ("diesel fuel and gasoline", "2815", {"9110", "9130", "9140"}),
        ("electronic components resistors capacitors connectors", None, {"5905", "5910", "5935", "5999"}),
        ("computers", "7020", {"7020", "7021", "7022"}),
        ("industrial chemicals solvents", "6810", {"6810", "6850"}),
    ],
)
async def test_product_phrases_find_the_obvious_classes(
    catalog: FscCatalog, phrase: str, expected_first: str | None, expected_in_top5: set[str]
) -> None:
    codes = await top_codes(Retriever(catalog, None), phrase)
    if expected_first:
        assert codes[0] == expected_first
    assert expected_in_top5 <= set(codes[:5])


async def test_group_title_lifts_whole_group(catalog: FscCatalog) -> None:
    medical = await top_codes(Retriever(catalog, None), "medical supplies surgical gloves bandages")
    assert sum(c.startswith("65") for c in medical[:8]) >= 7
    clothing = await top_codes(Retriever(catalog, None), "uniforms and work clothing")
    assert {c[:2] for c in clothing[:3]} == {"84"}  # "uniform" is not in the catalog; "clothing" is in the group title


# --------------------------------------------------------------------------- search mechanics


async def test_search_normalises_and_orders(catalog: FscCatalog) -> None:
    retriever = Retriever(catalog, None)
    queries = [
        Query(label="bolts", text="bolts"),
        Query(label="pumps", text="centrifugal pumps"),
        Query(label=SUMMARY_LABEL, text="hardware and pumps"),
    ]
    result = await retriever.search(queries, k_per_query=12, k_total=8)
    scores = [c.retrieval_score for c in result]
    assert len(result) <= 8 and len({c.code for c in result}) == len(result)
    assert scores[0] == 1.0 and scores == sorted(scores, reverse=True)
    assert all(0 < s <= 1 for s in scores)
    assert all(c.sources == ["bm25"] for c in result)
    assert [c.code for c in result] == [c.code for c in await retriever.search(queries, k_per_query=12, k_total=8)]


async def test_search_matched_products_come_from_labels_not_summary(catalog: FscCatalog) -> None:
    retriever = Retriever(catalog, None)
    result = {
        c.code: c
        for c in await retriever.search(
            [
                Query(label="bolts", text="bolts"),
                Query(label="washers", text="bolts and washers"),
                Query(label=SUMMARY_LABEL, text="bolts"),
            ]
        )
    }
    # "washers" only matches 5306 through its tail words, far below the 5306 hit of the "bolts" line: not a match
    assert result["5306"].matched_products == ["bolts"]
    assert result["5310"].matched_products == ["washers"]
    assert all(SUMMARY_LABEL not in c.matched_products for c in result.values())


async def test_every_product_line_keeps_its_own_best_match(catalog: FscCatalog) -> None:
    queries = [
        Query(label="a", text="bolts"),
        Query(label="b", text="bolts and washers"),
        Query(label="c", text="washers"),
    ]
    result = await Retriever(catalog, None).search(queries)
    assert [c.code for c in result] == ["5306", "5310"]
    # "washers" is the whole point of line c: the specialist hit is not drowned by the code two lines agree on
    assert result[0].retrieval_score == 1.0 and result[1].retrieval_score > 0.7
    single = await Retriever(catalog, None).search(queries[:1])
    assert [c.code for c in single] == ["5306"]


async def test_summary_query_backs_up_but_never_outvotes_product_lines(catalog: FscCatalog) -> None:
    retriever = Retriever(catalog, None)
    result = await retriever.search([Query(label="valves", text="valves"), Query(label=SUMMARY_LABEL, text="bolts")])
    scores = {c.code: c.retrieval_score for c in result}
    assert scores["4820"] == 1.0 or scores["4810"] == 1.0  # the product line leads
    assert 0 < scores["5306"] < 1.0  # found only by the summary: weaker evidence


async def test_thesaurus_bridges_everyday_words_to_catalog_vocabulary(catalog: FscCatalog) -> None:
    retriever = Retriever(catalog, None)
    for phrase, expected in [
        ("combat boots", {"8430", "8435"}),
        ("laptops", {"7021", "7010", "7042"}),
        ("spark plugs", {"2920", "2925"}),
        ("fire extinguishers", {"4210"}),
        ("ball bearings", {"3110"}),
    ]:
        result = await retriever.search([Query(label=phrase, text=phrase)])
        assert expected & {c.code for c in result[:4]}, (phrase, [c.code for c in result])


async def test_thesaurus_leaves_words_the_catalog_already_uses_alone(catalog: FscCatalog) -> None:
    retriever = Retriever(catalog, None)
    await retriever.build()
    assert tokenize("capacitors") and retriever._split(Query(label="capacitors", text="capacitors"))[0] == ["capacitor"]
    head, _ = retriever._split(Query(label="boots", text="boots"))
    assert "footwear" in head  # "boot" is not a catalog word: bridged
    head, _ = retriever._split(Query(label="drills", text="drills"))
    assert "driven" in head  # forced entry: "drill" is a catalog word but names the wrong class


async def test_search_edge_cases(catalog: FscCatalog) -> None:
    retriever = Retriever(catalog, None)
    assert await retriever.search([]) == []
    assert await retriever.search([Query(label="x", text="   ")]) == []
    assert await retriever.search([Query(label="x", text="zzzqqq xylophonics")]) == []
    assert len(await retriever.search([Query(label="x", text="tools")], k_per_query=3, k_total=2)) <= 2


# --------------------------------------------------------------------------- embedding channel & cache


async def test_embedding_channel_adds_codes_bm25_cannot_find(catalog: FscCatalog, cache_dir: Path) -> None:
    embedder = FakeEmbedder()
    retriever = Retriever(catalog, embedder)
    await retriever.build()
    assert retriever.active_channels == ["bm25", "embedding"]

    assert await Retriever(catalog, None).search([Query(label="notebooks", text="notebook")]) == []  # no lexical hit
    result = await retriever.search([Query(label="notebooks", text="notebook")])
    assert result and result[0].code.startswith("70")
    assert all(c.sources == ["embedding"] for c in result)
    assert result[0].matched_products == ["notebooks"]


async def test_channels_agree_on_exact_terms(catalog: FscCatalog, cache_dir: Path) -> None:
    retriever = Retriever(catalog, FakeEmbedder())
    result = await retriever.search([Query(label="screws", text="screws")])
    assert result[0].code == "5305"
    assert result[0].sources == ["bm25", "embedding"] and result[0].retrieval_score == 1.0


async def test_embedding_cache_round_trip(catalog: FscCatalog, cache_dir: Path) -> None:
    first = FakeEmbedder()
    r1 = Retriever(catalog, first)
    await r1.build()
    await r1.build()  # idempotent
    assert first.doc_calls == 1
    cache_file = cache_dir / "fsc_embeddings.npz"
    with np.load(cache_file) as z:
        assert z["vectors"].shape == (580, DIM)
        assert str(z["key"]).startswith(f"{get_settings().gemini_embed_model}:")

    second = FakeEmbedder()
    r2 = Retriever(catalog, second)
    await r2.build()
    assert second.doc_calls == 0, "second start-up must reuse the cache"
    queries = [Query(label="q", text="notebook screws")]
    assert await r2.search(queries) == await r1.search(queries)


async def test_embedding_cache_invalidated_by_model_or_catalog(
    catalog: FscCatalog, cache_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await Retriever(catalog, FakeEmbedder()).build()

    other_model = FakeEmbedder()
    monkeypatch.setattr(get_settings(), "gemini_embed_model", "some-other-embedding-model")
    await Retriever(catalog, other_model).build()
    assert other_model.doc_calls == 1

    smaller = FakeEmbedder()
    await Retriever(FscCatalog(catalog.codes[:50]), smaller).build()
    assert smaller.doc_calls == 1
    again = FakeEmbedder()
    await Retriever(FscCatalog(catalog.codes[:50]), again).build()
    assert again.doc_calls == 0


async def test_corrupt_cache_is_recomputed(catalog: FscCatalog, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True)
    (cache_dir / "fsc_embeddings.npz").write_bytes(b"not a zip file")
    embedder = FakeEmbedder()
    retriever = Retriever(catalog, embedder)
    await retriever.build()
    assert embedder.doc_calls == 1 and retriever.active_channels == ["bm25", "embedding"]
    with np.load(cache_dir / "fsc_embeddings.npz") as z:
        assert z["vectors"].shape[0] == 580


@pytest.mark.parametrize("embedder", [FakeEmbedder(fail_docs=True), FakeEmbedder(short_docs=True)])
async def test_embedding_failure_degrades_to_bm25(catalog: FscCatalog, cache_dir: Path, embedder: FakeEmbedder) -> None:
    retriever = Retriever(catalog, embedder)
    await retriever.build()  # must not raise
    assert retriever.active_channels == ["bm25"]
    assert not (cache_dir / "fsc_embeddings.npz").exists()
    result = await retriever.search([Query(label="screws", text="screws")])
    assert result[0].code == "5305" and result[0].sources == ["bm25"]


async def test_query_embedding_failure_degrades_for_that_search(catalog: FscCatalog, cache_dir: Path) -> None:
    embedder = FakeEmbedder(fail_queries=True)
    retriever = Retriever(catalog, embedder)
    await retriever.build()
    assert retriever.active_channels == ["bm25", "embedding"]
    result = await retriever.search([Query(label="screws", text="screws")])
    assert result[0].code == "5305" and all(c.sources == ["bm25"] for c in result)


async def test_unwritable_cache_dir_is_not_fatal(
    catalog: FscCatalog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the cache dir should be")
    monkeypatch.setattr(get_settings(), "cache_dir", blocker / "cache")
    retriever = Retriever(catalog, FakeEmbedder())
    await retriever.build()
    assert retriever.active_channels == ["bm25", "embedding"]


# --------------------------------------------------------------------------- merge_candidates


def cand(
    code: str, score: float, sources: list, products: list[str] | None = None, reason: str | None = None
) -> Candidate:
    return Candidate(
        code=code, retrieval_score=score, sources=sources, matched_products=products or [], llm_reason=reason
    )


def test_merge_candidates_unions_and_renormalises() -> None:
    retrieved = [cand("5305", 1.0, ["bm25"], ["screws"]), cand("5306", 0.5, ["bm25", "embedding"], ["bolts"])]
    proposed = [
        cand("5306", 0.9, ["llm"], ["bolts", "studs"], "sells bolts"),
        cand("4010", 0.4, ["llm"], [], "wire rope"),
    ]
    merged = merge_candidates(retrieved, proposed)

    assert [c.code for c in merged] == ["5305", "5306", "4010"]
    by_code = {c.code: c for c in merged}
    assert by_code["5306"].retrieval_score == pytest.approx(0.9)  # best of 0.5 / 0.9, then / max
    assert by_code["5306"].sources == ["bm25", "embedding", "llm"]
    assert by_code["5306"].matched_products == ["bolts", "studs"]
    assert by_code["5306"].llm_reason == "sells bolts"
    assert by_code["4010"].llm_reason == "wire rope" and by_code["4010"].sources == ["llm"]
    assert by_code["5305"].retrieval_score == 1.0


def test_merge_candidates_renormalises_to_best_one() -> None:
    merged = merge_candidates([cand("5305", 0.5, ["bm25"]), cand("5306", 0.25, ["bm25"])])
    assert [c.retrieval_score for c in merged] == [1.0, 0.5]


def test_merge_candidates_is_deterministic_and_pure() -> None:
    a = [cand("5306", 1.0, ["bm25"], ["bolts"])]
    b = [cand("5305", 1.0, ["llm"])]
    merged = merge_candidates(a, b)
    assert [c.code for c in merged] == ["5305", "5306"]  # equal scores tie-break by code
    merged[1].matched_products.append("mutated")
    assert a[0].matched_products == ["bolts"] and a[0].retrieval_score == 1.0


def test_merge_candidates_empty() -> None:
    assert merge_candidates() == []
    assert merge_candidates([], []) == []


# --------------------------------------------------------------------------- hierarchy routing (group / SOS RIC)


async def test_route_puts_fastener_queries_on_hardware_and_general_industrial(catalog: FscCatalog) -> None:
    retriever = Retriever(catalog, None)
    await retriever.build()
    routing = retriever.route([Query(label="hex bolts", text="hex bolts, socket cap screws, nuts and washers")])
    assert max(routing.group_prior, key=routing.group_prior.get) == "53"
    assert max(routing.domain_prior, key=routing.domain_prior.get) == "S9I/KZ"  # General & Industrial
    assert all(0 < v <= 1 for v in [*routing.group_prior.values(), *routing.domain_prior.values()])
    assert max(routing.group_prior.values()) == max(routing.domain_prior.values()) == 1.0


async def test_route_follows_the_product_not_the_word_overlap(catalog: FscCatalog) -> None:
    retriever = Retriever(catalog, None)
    await retriever.build()
    meds = retriever.route([Query(label="surgical gloves", text="surgical gloves, bandages, syringes")])
    assert max(meds.group_prior, key=meds.group_prior.get) == "65"
    assert max(meds.domain_prior, key=meds.domain_prior.get) == "S9M/KX"  # Medical
    assert retriever.route([]) == Routing()
    assert retriever.route([Query(label="zzz", text="qqqxx")]) == Routing()  # nothing matched: no prior, not a flat one


async def test_search_routed_returns_routing_and_search_remembers_it(catalog: FscCatalog) -> None:
    retriever = Retriever(catalog, None)
    queries = [Query(label="bolts", text="bolts and nuts")]
    candidates, routing = await retriever.search_routed(queries)
    assert routing.group_prior["53"] > 0.9 and candidates[0].code in {"5306", "5310"}
    assert retriever.last_routing == Routing()  # search_routed is the concurrency-safe form: no shared state written
    assert await retriever.search(queries) == candidates
    assert retriever.last_routing == routing
    assert (await retriever.search_routed([]))[1] == Routing()


async def test_cluster_coherence_lifts_a_supported_group_over_an_isolated_hit(catalog: FscCatalog) -> None:
    retriever = Retriever(catalog, None)
    await retriever.build()
    fused = {"5305": 1.0, "5306": 0.95, "5310": 0.9, "5320": 0.85, "9999": 1.0}  # four hardware hits, one stray
    lexical = Routing(group_prior={"53": 0.5, "99": 0.5}, domain_prior={"S9I/KZ": 0.5})
    out = retriever._coherent_routing(lexical, fused)
    assert out.group_prior["53"] > out.group_prior["99"]
    assert out.domain_prior["S9I/KZ"] > 0.5  # 5305.. and 9999 share the domain, so it is the best supported
    assert all(0 <= v <= 1 for v in [*out.group_prior.values(), *out.domain_prior.values()])


async def test_routing_nudges_near_ties_but_not_clear_lexical_wins(catalog: FscCatalog) -> None:
    retriever = Retriever(catalog, None)
    result = await retriever.search([Query(label="wire rope", text="wire rope and chain slings")])
    assert result[0].code == "4010"  # the exact class name wins however the priors fall


def test_catalog_relevance() -> None:
    assert catalog_relevance("hex bolts") > 0.4
    assert catalog_relevance("combat boots") > 0  # bridged through the thesaurus
    assert catalog_relevance("kansas usa") == 0 and catalog_relevance("") == 0
