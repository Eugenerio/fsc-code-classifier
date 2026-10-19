"""Blend retrieval and decider scores, add hierarchy priors, apply the selection thresholds, label confidence."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

from app.config import Settings, get_settings
from app.schemas import Candidate, CompanyProfile, Confidence, Decision, FscCode, FscMatch

if TYPE_CHECKING:
    from app.fsc import FscCatalog
    from app.retrieval import Routing

BOOST_RETRIEVAL_ONLY = 0.10  # max score added by the group / domain priors when nothing verified the candidates
BOOST_JUDGED = 0.05  # ... and when Jev or Gemini ran (their probabilities already carry most of the signal)
SUBSISTENCE_GUARD = 0.6  # score multiplier for food / live-animal classes when the company shows no food cue
HIGH, MEDIUM = 0.75, 0.55

# A `*` class (subsistence, live animals) only fits a company that deals in food or animals. Without this guard,
# words like "stock", "oil" or "cereal" in a machine-shop profile pull the food classes in.
_FOOD_CUES = re.compile(
    r"\b(food|foods|foodservice|beverage|beverages|meat|meats|poultry|seafood|fresh produce|dairy|bakery|bread|"
    r"grocery|groceries|catering|caterer|coffee|farm|farms|farming|farmer|livestock|cattle|fish|fruit|fruits|"
    r"vegetable|vegetables|egg|eggs|snack|snacks|confectionery|candy|restaurant|restaurants|subsistence|ration|"
    r"rations|animal|animals|hog|hogs|kennel|ranch)\b",
    re.IGNORECASE,
)


class _Row(NamedTuple):
    score: float
    fsc: FscCode
    candidate: Candidate
    decision: Decision | None  # None = retrieval-only mode, or the decider returned nothing for this code
    signals: dict[str, float]


def _label(row: _Row) -> Confidence:
    # A bare retrieval rank is never "high": the best candidate always normalises to 1.0.
    if row.score >= HIGH and row.decision is not None:
        return "high"
    return "medium" if row.score >= MEDIUM else "low"


def _rationale(row: _Row) -> str | None:
    c = row.candidate
    if row.decision and row.decision.rationale:
        return row.decision.rationale
    if c.llm_reason:
        return c.llm_reason
    if c.matched_products:
        return "matched: " + ", ".join(c.matched_products)
    return f"retrieved by {'+'.join(c.sources)}" if c.sources else None


def _match(row: _Row, confidence: Confidence) -> FscMatch:
    return FscMatch(
        code=row.fsc.code,
        description=row.fsc.description,
        sos_ric=row.fsc.sos_ric,
        group=row.fsc.group,
        group_name=row.fsc.group_name,
        score=row.score,
        confidence=confidence,
        decider_probability=row.decision.probability if row.decision else None,
        retrieval_score=row.candidate.retrieval_score,
        matched_products=row.candidate.matched_products,
        rationale=_rationale(row),
        agency=row.fsc.agency,
        supply_domain=row.fsc.supply_domain,
        marker_note=row.fsc.marker_note,
        signals=row.signals,
    )


def has_food_cue(profile: CompanyProfile) -> bool:
    text = " ".join(
        [
            profile.summary,
            *profile.business_types,
            *profile.keywords,
            *profile.industries,
            *(f"{p.name} {p.description} {' '.join(p.keywords)}" for p in profile.products),
        ]
    )
    return _FOOD_CUES.search(text) is not None


def _scaled(prior: dict[str, float] | None) -> dict[str, float]:
    """A decider prior is a probability distribution (peaked, sums to 1): put it on the heuristic 0..1 scale."""
    top = max((prior or {}).values(), default=0.0)
    return {k: v / top for k, v in (prior or {}).items()} if top > 0 else {}


def _axis(key: str, heuristic: dict[str, float] | None, judged: dict[str, float]) -> float | None:
    """Average of the heuristic and the decider prior for one axis; None when neither exists (axis is skipped)."""
    values = [prior.get(key, 0.0) for prior in (heuristic, judged) if prior]
    return sum(values) / len(values) if values else None


def rank_matches(
    candidates: list[Candidate],
    decisions: list[Decision],
    catalog: FscCatalog,
    *,
    group_prior: dict[str, float] | None = None,
    decider_name: str,
    domain_prior: dict[str, float] | None = None,
    routing: Routing | None = None,
    profile: CompanyProfile | None = None,
    settings: Settings | None = None,
) -> list[FscMatch]:
    """final = clamp((w*decider + (1-w)*retrieval + boost) * marker_guard); keep >= min_score in [min_results, max_results].

    `group_prior` / `domain_prior` (domain keyed by SOS RIC) are the decider's; `routing` holds the retriever's
    heuristic priors. An axis with both averages them, with one uses it, with neither is left out of the boost.
    Without a `profile` the subsistence guard cannot tell whether the company deals in food, so it stays off.
    """
    s = settings or get_settings()
    by_code = {} if decider_name == "retrieval" else {d.code: d for d in decisions}
    judged_group, judged_domain = _scaled(group_prior), _scaled(domain_prior)
    heur_group = routing.group_prior if routing else None
    heur_domain = routing.domain_prior if routing else None
    k = BOOST_RETRIEVAL_ONLY if decider_name == "retrieval" else BOOST_JUDGED
    food = profile is None or has_food_cue(profile)

    rows: list[_Row] = []
    for c in candidates:
        if (fsc := catalog.get(c.code)) is None:
            continue
        decision = by_code.get(c.code)
        base = c.retrieval_score
        if decision is not None:
            base = s.jev_weight * decision.probability + (1 - s.jev_weight) * c.retrieval_score
        g = _axis(fsc.group, heur_group, judged_group)
        d = _axis(fsc.sos_ric, heur_domain, judged_domain)
        axes = [v for v in (g, d) if v is not None]
        boost = k * sum(axes) / len(axes) if axes else 0.0  # <= k, so priors alone never rescue a rejected class
        guard = 1.0 if fsc.marker_kind != "subsistence" or food else SUBSISTENCE_GUARD
        score = round(min(1.0, max(0.0, (base + boost) * guard)), 4)
        signals = {
            "group_prior": round(g or 0.0, 3),
            "domain_prior": round(d or 0.0, 3),
            "boost": round(boost, 4),
            "marker_guard": guard,
        }
        rows.append(_Row(score, fsc, c, decision, signals))
    rows.sort(key=lambda r: (-r.score, r.fsc.code))  # rounded score + code => deterministic ties

    kept = [r for r in rows if r.score >= s.min_score][: s.max_results]
    padding = rows[len(kept) : min(s.min_results, s.max_results)]  # best of the rest: always give something to review
    return [_match(r, _label(r)) for r in kept] + [_match(r, "low") for r in padding]
