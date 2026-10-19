from __future__ import annotations

from app.config import Settings
from app.schemas import Candidate, Decision, FscCode
from app.scoring import rank_matches


class FakeCatalog:
    def __init__(self, *rows: tuple[str, str]) -> None:
        self.by_code = {
            code: FscCode(code=code, description=desc, sos_ric="S9C/AX", group=code[:2], group_name=f"Group {code[:2]}")
            for code, desc in rows
        }

    def get(self, code: str) -> FscCode | None:
        return self.by_code.get(code)


CATALOG = FakeCatalog(
    ("5305", "Screws"), ("5306", "Bolts"), ("5310", "Nuts and Washers"), ("5315", "Nails"), ("4730", "Fittings")
)
SETTINGS = Settings(_env_file=None, jev_weight=0.75, min_score=0.45, max_results=3, min_results=2)


def cand(code: str, score: float, **kw) -> Candidate:
    return Candidate(code=code, retrieval_score=score, sources=["bm25"], **kw)


def dec(code: str, p: float, rationale: str | None = None) -> Decision:
    return Decision(code=code, probability=p, rationale=rationale)


def rank(candidates, decisions, *, prior=None, decider="jev", settings=SETTINGS):
    return rank_matches(candidates, decisions, CATALOG, group_prior=prior, decider_name=decider, settings=settings)


def test_blend_uses_jev_weight_and_sorts_desc():
    out = rank([cand("5305", 1.0), cand("5306", 0.4)], [dec("5305", 0.6), dec("5306", 0.95)])
    assert [m.code for m in out] == ["5306", "5305"]
    assert out[0].score == round(0.75 * 0.95 + 0.25 * 0.4, 4)
    assert out[0].decider_probability == 0.95
    assert out[0].retrieval_score == 0.4
    assert (out[0].group, out[0].group_name, out[0].sos_ric) == ("53", "Group 53", "S9C/AX")


def test_confidence_labels_at_boundaries():
    out = rank(
        [cand("5305", 0.0), cand("5306", 0.0), cand("5310", 0.0)],
        [dec("5305", 1.0), dec("5306", 0.8), dec("5310", 0.62)],
        settings=Settings(_env_file=None, jev_weight=1.0, min_score=0.1, max_results=5, min_results=1),
    )
    assert [(m.code, m.score, m.confidence) for m in out] == [
        ("5305", 1.0, "high"),
        ("5306", 0.8, "high"),
        ("5310", 0.62, "medium"),
    ]
    exact = rank(
        [cand("5305", 0.0), cand("5306", 0.0)],
        [dec("5305", 0.75), dec("5306", 0.55)],
        settings=Settings(_env_file=None, jev_weight=1.0, min_score=0.1, max_results=5, min_results=1),
    )
    assert [m.confidence for m in exact] == ["high", "medium"]


def test_nothing_above_min_score_still_returns_min_results_flagged_low():
    out = rank([cand("5305", 0.3), cand("5306", 0.2), cand("5310", 0.1)], [dec("5305", 0.2), dec("5306", 0.1)])
    assert [m.code for m in out] == ["5305", "5306"]  # min_results=2
    assert {m.confidence for m in out} == {"low"}
    assert all(m.score < SETTINGS.min_score for m in out)


def test_padding_is_forced_low_even_if_min_score_is_above_medium():
    settings = Settings(_env_file=None, jev_weight=1.0, min_score=0.95, max_results=5, min_results=2)
    out = rank([cand("5305", 0.0), cand("5306", 0.0)], [dec("5305", 0.8), dec("5306", 0.7)], settings=settings)
    assert [m.confidence for m in out] == ["low", "low"]


def test_max_results_caps_and_min_results_pads_only_the_gap():
    cands = [cand(c, 0.9) for c in ("5305", "5306", "5310", "5315")]
    decisions = [dec(c, 0.9) for c in ("5305", "5306", "5310", "5315")]
    assert len(rank(cands, decisions)) == SETTINGS.max_results
    one_good = rank([cand("5305", 0.9), cand("5306", 0.1), cand("5310", 0.01)], [dec("5305", 0.9), dec("5306", 0.0)])
    assert [(m.code, m.confidence) for m in one_good] == [("5305", "high"), ("5306", "low")]


def test_ties_break_deterministically_by_code():
    cands = [cand("5310", 0.8), cand("5305", 0.8), cand("5306", 0.8)]
    decisions = [dec("5310", 0.8), dec("5305", 0.8), dec("5306", 0.8)]
    assert [m.code for m in rank(cands, decisions)] == ["5305", "5306", "5310"]
    assert [m.code for m in rank(list(reversed(cands)), decisions)] == ["5305", "5306", "5310"]


def test_retrieval_only_ignores_decisions_caps_confidence_and_hides_probability():
    cands = [cand("5305", 1.0, matched_products=["hex bolts"]), cand("5306", 0.6), cand("5310", 0.2)]
    decisions = [dec("5305", 0.0), dec("5306", 0.0)]  # must be ignored in retrieval mode
    out = rank(cands, decisions, decider="retrieval")
    assert [(m.code, m.score) for m in out] == [("5305", 1.0), ("5306", 0.6)]
    assert [m.confidence for m in out] == ["medium", "medium"]  # an unverified rank is never "high"
    assert all(m.decider_probability is None for m in out)
    assert out[0].rationale == "matched: hex bolts"


def test_candidate_without_a_decision_falls_back_to_retrieval_score():
    out = rank([cand("5305", 0.9), cand("5306", 0.9)], [dec("5306", 0.9)])
    assert {m.code: m.decider_probability for m in out} == {"5305": None, "5306": 0.9}
    assert next(m for m in out if m.code == "5305").confidence == "medium"


def test_group_prior_boosts_top_group_and_is_clamped():
    cands = [cand("5305", 0.5), cand("4730", 0.5)]
    decisions = [dec("5305", 0.6), dec("4730", 0.6)]
    plain = {m.code: m.score for m in rank(cands, decisions)}
    boosted = {m.code: m.score for m in rank(cands, decisions, prior={"53": 0.7, "47": 0.1, "99": 0.2})}
    assert boosted["5305"] == round(plain["5305"] + 0.05, 4)
    assert boosted["4730"] == round(plain["4730"] + 0.05 * 0.1 / 0.7, 4)
    top = rank([cand("5305", 1.0)], [dec("5305", 1.0)], prior={"53": 1.0})
    assert top[0].score == 1.0


def test_rationale_precedence_and_unknown_codes_dropped():
    out = rank(
        [
            cand("5305", 0.9, llm_reason="reason from llm", matched_products=["screws"]),
            cand("5306", 0.9, llm_reason="reason from llm"),
            cand("5310", 0.9, matched_products=["nuts", "washers"]),
            cand("0000", 1.0),
        ],
        [dec("5305", 0.9, "decider says"), dec("5306", 0.9), dec("5310", 0.9)],
        settings=Settings(_env_file=None, min_score=0.1, max_results=10, min_results=1),
    )
    assert {m.code: m.rationale for m in out} == {
        "5305": "decider says",
        "5306": "reason from llm",
        "5310": "matched: nuts, washers",
    }


def test_empty_candidates():
    assert rank([], []) == []
