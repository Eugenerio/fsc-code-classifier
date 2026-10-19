"""Recall channels over the FSC catalog: BM25 + Gemini embeddings, fused with reciprocal-rank fusion."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from functools import lru_cache
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.fsc import FscCatalog, catalog_text, load_catalog
from app.schemas import Candidate, CandidateSource, CompanyProfile
from app.vocab import BRIDGE

if TYPE_CHECKING:
    from app.llm.gemini import GeminiClient

log = logging.getLogger(__name__)

SUMMARY_LABEL = "__summary__"
RRF_K = 60
GROUP_WEIGHT = 0.15  # group-title BM25 is a hint, not evidence: it lifts siblings of a good hit
DOMAIN_WEIGHT = 0.1  # same for the supply-domain label ("Medical", "Electronics", "GSA Furniture")
ROUTE_STRENGTH = 0.35  # retrieval_score *= 1 + ROUTE_STRENGTH * (0.5 * group_prior + 0.5 * domain_prior)
COHERENCE_TOP = 25  # how many best candidates vote on which groups / domains are well supported
TEMPLATE_VERSION = "v2"  # bump when fsc.catalog_text() changes: it is part of the embedding cache key
TAIL_WEIGHT = 0.5  # query words beyond the product-line name (description, keywords) count for less
BM25_K1, BM25_B = 1.2, 0.4  # descriptions are 1-12 tokens: length normalisation should be gentle
REL_CUTOFF = 0.3  # drop hits far weaker than the query's best
COVERAGE_WEIGHT = 0.35  # lexical-only hits lose up to this share if the evidence never uses most of the class's words
SUMMARY_WEIGHT = 0.5  # the summary query backs the product lines up; it must not outvote them
SUPPORT_BONUS = (0.10, 0.05)  # a code's 2nd and 3rd best product line add this share of their score
MATCH_SHARE = 0.6  # a product line "matches" a code when its score is within this share of the code's best
SOURCE_ORDER: tuple[CandidateSource, ...] = ("bm25", "embedding", "llm")

# --------------------------------------------------------------------------- tokenisation

_WORD = re.compile(r"[a-z0-9]+")
_HYPHENATED = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")
# Function words plus catalog/company-blurb filler ("Equipment", "Miscellaneous", "industrial customers")
_STOPWORDS = (
    "a an and or of for the to in on with by from as at is are be per thru over up down not more non other except "
    "related miscellaneous equipment item used use type size high low full industrial commercial "
    "product manufacturer manufacturing supplier"
).split()
# The sheet abbreviates freely ("Elect Arc Welding", "Maintenance and Rep Shop"); expand so queries match.
_ALIASES = {
    "elect": "electric",
    "electrical": "electric",
    "mach": "machine",
    "machinery": "machine",
    "rep": "repair",
    "spec": "special",
    "med": "medical",
    "temp": "temperature",
    "trans": "transmission",
    "ammo": "ammunition",
    "prep": "preparation",
    "info": "information",
    "cent": "central",
    "assoc": "associated",
    "adp": "automatic data processing computer",
    "adpe": "automatic data processing computer",
    "part": "component",
    "tyre": "tire",
    # vocabulary the catalog lacks: oil-field and IT-hardware words mapped onto the classes that cover them
    "oilfield": "petroleum production drilling",
    "downhole": "drilling earth boring petroleum",
    "wellhead": "petroleum production",
    "laptop": "computer",
    "server": "computer",
    "desktop": "computer",
}


def _stem(word: str) -> str:
    """Light stemmer: plurals, -ing/-ed/-er and a trailing e, so tube/tubes/tubing and fastener/fastening agree."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith(("sses", "shes", "ches", "xes", "zes")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for suffix, min_stem in (("ing", 3), ("ed", 3), ("er", 4)):
        stem = word[: -len(suffix)]
        if word.endswith(suffix) and len(stem) >= min_stem:
            word = stem[:-1] if stem[-1] == stem[-2] and stem[-1] not in "lsz" else stem  # fitt -> fit
            break
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


_STOP_STEMS = frozenset(_stem(w) for w in _STOPWORDS)


def _base_tokens(text: str) -> list[str]:
    text = text.lower()
    # "o-rings" also yields "oring" so hyphenated and run-together spellings meet
    words = _WORD.findall(text) + [m.replace("-", "") for m in _HYPHENATED.findall(text)]
    expanded = (part for w in words for part in (_ALIASES.get(w) or _ALIASES.get(w.removesuffix("s")) or w).split())
    stems = (_stem(w) for w in expanded if len(w) > 1)
    return [t for t in stems if t not in _STOP_STEMS]


_RAW_STOP = frozenset(_STOPWORDS)


def _surface(text: str) -> list[str]:
    """Lowercase words with plurals folded, nothing more: bridge keys must not collide through stemming
    ("locker" and "lock" share a stem but not a meaning)."""
    out = []
    for w in _WORD.findall(text.lower()):
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith(("sses", "shes", "ches", "xes")):
            w = w[:-2]
        elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
            w = w[:-1]
        if w not in _RAW_STOP:
            out.append(w)
    return out


def _parse_bridge(source: str) -> dict[tuple[str, ...], tuple[bool, list[str]]]:
    """`a | !b c => x y` lines -> {("a",): (False, [x, y]), ("b", "c"): (True, [x, y])}.

    Keys are surface words; the expansion is in token space. A leading `!` forces the expansion even for a
    single word the catalog itself uses (see `bridge_tokens`).
    """
    table: dict[tuple[str, ...], tuple[bool, list[str]]] = {}
    for line in source.splitlines():
        if "=>" not in line or line.lstrip().startswith("#"):
            continue
        keys, _, expansion = line.partition("=>")
        extra = _base_tokens(expansion)
        for key in keys.split("|"):
            force = key.strip().startswith("!")
            if words := tuple(_surface(key.replace("!", " "))):
                _, known = table.get(words, (False, []))
                table[words] = (force, [*known, *(t for t in extra if t not in known)])
    return table


_BRIDGE = _parse_bridge(BRIDGE)
_BRIDGE_N = max(map(len, _BRIDGE))


def bridge_tokens(text: str, vocabulary: AbstractSet[str]) -> list[str]:
    """Catalog words for everyday product words the catalog does not use itself, to append to a query's tokens.

    A single word that already occurs in some class description ("capacitor", "valve") needs no help and would only
    dilute the exact hit, so it expands only when the thesaurus line forces it with `!`. Phrases always expand.
    """
    words, extra, i = _surface(text), [], 0
    while i < len(words):  # longest match first: "surgical glove" is not just a "glove"
        for n in range(min(_BRIDGE_N, len(words) - i), 0, -1):
            hit = _BRIDGE.get(tuple(words[i : i + n]))
            if hit is None:
                continue
            force, tokens = hit
            if n > 1 or force or not any(tok in vocabulary for tok in _base_tokens(words[i])):
                extra += tokens
            i += n - 1
            break
        i += 1
    return list(dict.fromkeys(extra))


def tokenize(text: str) -> list[str]:
    """Normalised search tokens: lowercase, abbreviations expanded, stemmed, stop words removed."""
    return _base_tokens(text)


@lru_cache
def _catalog_vocabulary() -> frozenset[str]:
    return frozenset(t for c in load_catalog().codes for t in tokenize(c.description))


def catalog_relevance(text: str) -> float:
    """0..1: how much of this phrase names something in the FSC catalog, directly or through the thesaurus."""
    vocabulary, tokens = _catalog_vocabulary(), tokenize(text)
    if not tokens:
        return 0.0
    direct = sum(t in vocabulary for t in tokens) / len(tokens)
    return max(direct, 0.5 if bridge_tokens(text, vocabulary) else 0.0)


# --------------------------------------------------------------------------- queries


class Query(BaseModel):
    label: str  # ProductLine.name or SUMMARY_LABEL
    text: str


class Routing(BaseModel):
    """Where in the hierarchy the company's products live: 0..1 priors per FSC group and per SOS RIC domain."""

    group_prior: dict[str, float] = {}
    domain_prior: dict[str, float] = {}  # keyed by SOS RIC, e.g. "S9I/KZ"


def _join(*parts: str) -> str:
    return ". ".join(p.strip().rstrip(".") for p in parts if p.strip())


def build_queries(profile: CompanyProfile) -> list[Query]:
    names = [p.name.lower() for p in profile.products]
    queries = []
    for i, p in enumerate(profile.products):
        # A description that names other product lines (a copied list sentence) is not evidence for this one.
        crosstalk = sum(name in p.description.lower() for j, name in enumerate(names) if j != i and name != names[i])
        description = "" if crosstalk >= 2 else p.description
        queries.append(Query(label=p.name, text=_join(p.name, description, ", ".join(p.keywords))))
    queries.append(
        Query(
            label=SUMMARY_LABEL,
            text=_join(profile.summary, ", ".join(profile.keywords), ", ".join(profile.industries)),
        )
    )
    return [q for q in queries if q.text]


def merge_candidates(*lists: list[Candidate]) -> list[Candidate]:
    """Union by code: keep the best score, union sources/products, keep any LLM reason; best candidate = 1.0."""
    merged: dict[str, Candidate] = {}
    for cand in (c for lst in lists for c in lst):
        cur = merged.get(cand.code)
        if cur is None:
            merged[cand.code] = cand.model_copy(deep=True)
            continue
        cur.retrieval_score = max(cur.retrieval_score, cand.retrieval_score)
        cur.sources = [s for s in SOURCE_ORDER if s in {*cur.sources, *cand.sources}]
        cur.matched_products += [p for p in cand.matched_products if p not in cur.matched_products]
        cur.llm_reason = cur.llm_reason or cand.llm_reason
    top = max((c.retrieval_score for c in merged.values()), default=0.0)
    for c in merged.values():
        c.retrieval_score = c.retrieval_score / top if top > 0 else 0.0
    return sorted(merged.values(), key=lambda c: (-c.retrieval_score, c.code))


# --------------------------------------------------------------------------- retriever


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def _unit_max(values: dict[str, float]) -> dict[str, float]:
    """Scale to 0..1 by the maximum; drops non-positive entries (a prior of 0 and a missing one mean the same)."""
    top = max(values.values(), default=0.0)
    return {k: float(v) / top for k, v in values.items() if top > 0 and v > 0}


def _average(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: (a.get(k, 0.0) + b.get(k, 0.0)) / 2 for k in a.keys() | b.keys()}


def _top_hits(scores: np.ndarray, k: int) -> list[int]:
    order = np.argsort(-scores, kind="stable")[:k]  # stable: ties resolve to the lower code
    floor = max(REL_CUTOFF * scores[order[0]], 1e-9)
    return [int(i) for i in order if scores[i] >= floor]


def _strengths(channel: CandidateSource, scores: np.ndarray, hits: list[int], k: int) -> list[float]:
    """How strong each hit is relative to the query's best, 0..1 (RRF alone is flat over 12 ranks).

    BM25 scores are ratio-scaled, so a hit's strength is its share of the best score. Cosine similarities are
    not (unrelated text still sits near 0.4), so they are rescaled between the best and the 3k-th best.
    """
    top = float(scores[hits[0]])
    if channel == "bm25":
        return [float(scores[i]) / top for i in hits]
    floor = float(np.sort(scores)[-min(3 * k, len(scores))])
    span = top - floor
    return [1.0 if span <= 1e-9 else max(0.0, (float(scores[i]) - floor) / span) for i in hits]


class Retriever:
    def __init__(self, catalog: FscCatalog, gemini: GeminiClient | None) -> None:
        self.catalog = catalog
        self.gemini = gemini
        self.active_channels: list[CandidateSource] = []
        self._built = False
        self.last_routing = Routing()  # of the most recent search(); concurrent callers use search_routed()
        self._vectors: np.ndarray | None = None  # (n_codes, d), unit length; None = embedding channel off
        self._desc_index: BM25Okapi
        self._title_index: BM25Okapi  # one doc per group: its title
        self._label_index: BM25Okapi  # one doc per SOS RIC: its supply-domain label
        self._group_docs: BM25Okapi  # one doc per group: title + every class description (routing prior)
        self._domain_docs: BM25Okapi  # one doc per SOS RIC: label + every class description (routing prior)
        self._groups: list[str] = []
        self._rics: list[str] = []
        self._vocab: frozenset[str] = frozenset()  # every token used by a class description
        self._doc_tokens: dict[str, frozenset[str]] = {}  # code -> tokens of its description
        self._group_of_row: np.ndarray  # code row -> row of its group in _groups
        self._domain_of_row: np.ndarray  # code row -> row of its SOS RIC in _rics

    async def build(self) -> None:
        if self._built:
            return
        codes = self.catalog.codes
        docs = [tokenize(c.description) for c in codes]
        self._vocab = frozenset(t for doc in docs for t in doc)
        self._desc_index = BM25Okapi(docs, k1=BM25_K1, b=BM25_B)
        self._doc_tokens = {c.code: frozenset(doc) for c, doc in zip(codes, docs, strict=True)}
        # Titles and labels are indexed once per group / domain (not per code) so big ones don't dilute their own IDF.
        self._groups = list(self.catalog.groups)
        self._rics = list(self.catalog.domains)
        group_ids = {g: i for i, g in enumerate(self._groups)}
        ric_ids = {r: i for i, r in enumerate(self._rics)}
        titles = [self.catalog.groups[g] for g in self._groups]
        labels = [self.catalog.domains[r].label for r in self._rics]
        self._title_index = BM25Okapi([tokenize(t) for t in titles], k1=BM25_K1, b=BM25_B)
        self._label_index = BM25Okapi([tokenize(t) for t in labels], k1=BM25_K1, b=BM25_B)
        self._group_of_row = np.array([group_ids[c.group] for c in codes])
        self._domain_of_row = np.array([ric_ids[c.sos_ric] for c in codes])
        # Routing indexes: a class description is terse, but the pooled vocabulary of its group / buying centre is not.
        group_text = [[titles[i], *(c.description for c in self.catalog.in_group(g))] for i, g in enumerate(self._groups)]
        domain_text = [
            [labels[i], *(self.catalog.by_code[c].description for c in self.catalog.domains[r].codes)]
            for i, r in enumerate(self._rics)
        ]
        self._group_docs = BM25Okapi([tokenize(" ".join(t)) for t in group_text], k1=BM25_K1, b=BM25_B)
        self._domain_docs = BM25Okapi([tokenize(" ".join(t)) for t in domain_text], k1=BM25_K1, b=BM25_B)
        self.active_channels = ["bm25"]
        if self.gemini is not None:
            try:
                self._vectors = _unit(await self._load_or_embed(self.gemini))
                self.active_channels.append("embedding")
            except Exception as exc:  # any embedding failure must degrade to BM25-only
                log.warning("embedding channel disabled: %s", exc)
        self._built = True

    def _doc_texts(self) -> list[str]:
        return [catalog_text(c) for c in self.catalog.codes]

    async def _load_or_embed(self, gemini: GeminiClient) -> np.ndarray:
        settings = get_settings()
        texts = self._doc_texts()
        digest = hashlib.sha1("\n".join(texts).encode()).hexdigest()
        key = f"{settings.gemini_embed_model}:{TEMPLATE_VERSION}:{digest}"
        path = settings.cache_dir / "fsc_embeddings.npz"
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["key"]) == key:
                    return z["vectors"]
        except Exception:  # missing, truncated or foreign file: just recompute
            pass
        vectors = np.asarray(await gemini.embed(texts, task_type="RETRIEVAL_DOCUMENT"), dtype=np.float32)
        if vectors.shape[0] != len(texts):
            raise ValueError(f"expected {len(texts)} embeddings, got {vectors.shape[0]}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, key=key, vectors=vectors)
            tmp.replace(path)  # atomic: a concurrent reader never sees a half-written file
        except OSError as exc:
            log.warning("could not write embedding cache: %s", exc)
        return vectors

    def _score(self, tokens: list[str]) -> np.ndarray:
        if not tokens:
            return np.zeros(len(self.catalog.codes))
        titles = self._title_index.get_scores(tokens)[self._group_of_row]
        labels = self._label_index.get_scores(tokens)[self._domain_of_row]
        return self._desc_index.get_scores(tokens) + GROUP_WEIGHT * titles + DOMAIN_WEIGHT * labels

    def _split(self, query: Query) -> tuple[list[str], list[str]]:
        """(head, tail) tokens. For product lines the label is the name: the high-precision part, and the only
        one the thesaurus bridges (free text such as a description would make every line expand every product)."""
        if query.label == SUMMARY_LABEL:
            head: list[str] = []
        else:
            head = list(dict.fromkeys([*tokenize(query.label), *bridge_tokens(query.label, self._vocab)]))
        tail = [t for t in dict.fromkeys(tokenize(query.text)) if t not in head]
        return head, tail

    def _bm25_scores(self, query: Query) -> np.ndarray:
        head, tail = self._split(query)
        return self._score(head) + TAIL_WEIGHT * self._score(tail)

    def route(self, queries: list[Query]) -> Routing:
        """Heuristic hierarchy priors, no API: which FSC groups and supply domains do these queries point at?"""
        group_mass = np.zeros(len(self._groups))
        domain_mass = np.zeros(len(self._rics))
        for query in queries:
            head, tail = self._split(query)
            for tokens, weight in ((head, 1.0), (tail, TAIL_WEIGHT)):
                if tokens:
                    group_mass += weight * self._group_docs.get_scores(tokens)
                    domain_mass += weight * self._domain_docs.get_scores(tokens)
        return Routing(
            group_prior=_unit_max(dict(zip(self._groups, group_mass, strict=True))),
            domain_prior=_unit_max(dict(zip(self._rics, domain_mass, strict=True))),
        )

    async def _embedding_scores(self, queries: list[Query]) -> np.ndarray | None:
        if self._vectors is None or self.gemini is None:
            return None
        try:
            vectors = await self.gemini.embed([q.text for q in queries], task_type="RETRIEVAL_QUERY")
            qvec = _unit(np.asarray(vectors, dtype=np.float32))
            if qvec.shape[0] != len(queries):
                raise ValueError(f"expected {len(queries)} query embeddings, got {qvec.shape[0]}")
            return qvec @ self._vectors.T
        except Exception as exc:  # a flaky query embedding must not fail the whole search
            log.warning("query embedding failed, BM25 only for this search: %s", exc)
            return None

    async def search(
        self, queries: list[Query], k_per_query: int | None = None, k_total: int | None = None
    ) -> list[Candidate]:
        candidates, routing = await self.search_routed(queries, k_per_query, k_total)
        self.last_routing = routing
        return candidates

    async def search_routed(
        self, queries: list[Query], k_per_query: int | None = None, k_total: int | None = None
    ) -> tuple[list[Candidate], Routing]:
        """Fuse channels per product line, pool across lines, lift well-supported groups / domains. Concurrency-safe."""
        await self.build()
        settings = get_settings()
        k_q, k_t = k_per_query or settings.k_per_query, k_total or settings.k_total
        queries = [q for q in queries if q.text.strip()]
        if not queries:
            return [], Routing()
        channels: dict[CandidateSource, np.ndarray] = {"bm25": np.stack([self._bm25_scores(q) for q in queries])}
        if (embedding := await self._embedding_scores(queries)) is not None:
            channels["embedding"] = embedding

        # One score per (query, code): channels agree -> strong. Across queries: best product line plus a little
        # for each further one, so a code that is THE match for one product line is not drowned by popular codes.
        per_query: dict[str, list[tuple[float, str]]] = defaultdict(list)
        sources: dict[str, set[CandidateSource]] = defaultdict(set)
        for qi, query in enumerate(queries):
            weight = SUMMARY_WEIGHT if query.label == SUMMARY_LABEL else 1.0
            score: dict[str, float] = defaultdict(float)
            for channel, matrix in channels.items():
                hits = _top_hits(matrix[qi], k_q)
                if not hits:
                    continue
                for rank, (i, strength) in enumerate(zip(hits, _strengths(channel, matrix[qi], hits, k_q)), start=1):
                    code = self.catalog.codes[i].code
                    score[code] += (RRF_K + 1) / (RRF_K + rank) * strength / len(channels)
                    sources[code].add(channel)
            for code, value in score.items():
                per_query[code].append((value * weight, query.label))
        if not per_query:
            return [], Routing()

        fused: dict[str, float] = {}
        products: dict[str, list[str]] = {}
        for code, found in per_query.items():
            found.sort(key=lambda f: -f[0])
            fused[code] = found[0][0] + sum(b * f[0] for b, f in zip(SUPPORT_BONUS, found[1:], strict=False))
            products[code] = [label for value, label in found if label != SUMMARY_LABEL and value >= MATCH_SHARE * found[0][0]]

        evidence = {tok for q in queries for part in self._split(q) for tok in part}
        for code in fused:
            if sources[code] == {"bm25"}:  # an embedding hit is vouched for semantically; a lexical one must use the words
                fused[code] *= 1 - COVERAGE_WEIGHT + COVERAGE_WEIGHT * self._coverage(code, evidence)

        routing = self._coherent_routing(self.route(queries), fused)
        for code in fused:
            fsc = self.catalog.by_code[code]
            prior = 0.5 * routing.group_prior.get(fsc.group, 0.0) + 0.5 * routing.domain_prior.get(fsc.sos_ric, 0.0)
            fused[code] *= 1 + ROUTE_STRENGTH * prior
        top = max(fused.values())
        best = sorted(fused, key=lambda c: (-fused[c], c))[:k_t]
        candidates = [
            Candidate(
                code=code,
                retrieval_score=fused[code] / top,
                sources=[ch for ch in SOURCE_ORDER if ch in sources[code]],
                matched_products=products[code],
            )
            for code in best
        ]
        return candidates, routing

    def _coverage(self, code: str, evidence: set[str]) -> float:
        """IDF-weighted share of the class description's words that the evidence uses (1.0 = all of them)."""
        idf = self._desc_index.idf
        words = self._doc_tokens[code]
        total = sum(idf[w] for w in words)
        return sum(idf[w] for w in words if w in evidence) / total if total > 0 else 1.0

    def _coherent_routing(self, routing: Routing, fused: dict[str, float]) -> Routing:
        """Average the lexical routing prior with the share of top-candidate mass each group / domain holds.

        A class sitting in a group or buying centre that several other strong candidates also sit in is likelier
        right than an isolated one-off lexical hit.
        """
        best = sorted(fused, key=lambda c: (-fused[c], c))[:COHERENCE_TOP]
        top = fused[best[0]]
        group_mass: dict[str, float] = defaultdict(float)
        domain_mass: dict[str, float] = defaultdict(float)
        for code in best:
            fsc = self.catalog.by_code[code]
            group_mass[fsc.group] += fused[code] / top
            domain_mass[fsc.sos_ric] += fused[code] / top
        return Routing(
            group_prior=_average(routing.group_prior, _unit_max(group_mass)),
            domain_prior=_average(routing.domain_prior, _unit_max(domain_mass)),
        )
